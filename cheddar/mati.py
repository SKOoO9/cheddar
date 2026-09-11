from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


MATI_ACQUISITION_FIELDS = (
    "TE",
    "TR",
    "FA",
    "TI",
    "TD",
    "B0",
    "df",
    "FSL",
    "TSL",
    "delta",
    "Delta",
    "shape",
    "b",
    "G",
    "n",
    "trise",
    "gdir",
)

DEFAULT_ENCODING_GROUP_FIELDS = ("shape", "n", "delta", "Delta", "trise", "TE", "TR", "FA")


class MatiPulseError(ValueError):
    """Raised when a MATI diffusion pulse is missing or inconsistent."""


@dataclass(frozen=True)
class MatiEncodingGroup:
    label: str
    indices: tuple[int, ...]
    encoding: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _nacq(payload: Mapping[str, Any]) -> int:
    try:
        value = int(payload["Nacq"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MatiPulseError("MATI pulse JSON must contain an integer Nacq.") from exc
    if value <= 0:
        raise MatiPulseError("MATI pulse Nacq must be positive.")
    return value


def acquisition_array(payload: Mapping[str, Any], field: str, *, nacq: int | None = None) -> np.ndarray:
    n = _nacq(payload) if nacq is None else int(nacq)
    if field not in payload:
        raise MatiPulseError(f"MATI pulse is missing required field {field!r}.")

    value = payload[field]
    if field == "gdir":
        arr = np.asarray(value, dtype=float)
        if arr.shape == (3,):
            return np.tile(arr.reshape(1, 3), (n, 1))
        if arr.shape == (1, 3):
            return np.tile(arr, (n, 1))
        if arr.shape != (n, 3):
            raise MatiPulseError(f"MATI pulse gdir must be length 3 or Nacq x 3; got {arr.shape}.")
        return arr

    dtype = str if field == "shape" else float
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0 or arr.size == 1:
        return np.full(n, arr.reshape(-1)[0], dtype=dtype)
    arr = arr.reshape(-1)
    if arr.size != n:
        raise MatiPulseError(f"MATI pulse field {field!r} must be scalar or length Nacq; got {arr.size}.")
    return arr


def validate_mati_pulse(
    payload: Mapping[str, Any],
    *,
    expected_nacq: int | None = None,
    required_fields: Sequence[str] = (),
) -> int:
    n = _nacq(payload)
    if expected_nacq is not None and n != int(expected_nacq):
        raise MatiPulseError(f"MATI pulse Nacq={n} does not match the DWI volume count {expected_nacq}.")

    for field in required_fields:
        arr = acquisition_array(payload, field, nacq=n)
        if field != "shape" and not np.all(np.isfinite(arr.astype(float))):
            raise MatiPulseError(f"MATI pulse field {field!r} contains non-finite values.")
        if field == "shape" and any(not str(item).strip() for item in arr):
            raise MatiPulseError("MATI pulse shape contains an empty value.")
    return n


def load_mati_pulse(
    path: str | Path,
    *,
    expected_nacq: int | None = None,
    required_fields: Sequence[str] = (),
) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MatiPulseError(f"MATI pulse JSON not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise MatiPulseError(f"Invalid MATI pulse JSON {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MatiPulseError(f"MATI pulse JSON must contain an object: {source}")
    validate_mati_pulse(payload, expected_nacq=expected_nacq, required_fields=required_fields)
    return dict(payload)


def validate_mati_bvalues(
    payload: Mapping[str, Any],
    bvals_smm2: np.ndarray,
    *,
    absolute_tolerance: float = 50.0,
    relative_tolerance: float = 0.05,
) -> None:
    if "b" not in payload:
        return
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise MatiPulseError("MATI b-value tolerances must be non-negative.")

    n = validate_mati_pulse(payload)
    observed = np.asarray(bvals_smm2, dtype=float).reshape(-1)
    if observed.size != n:
        raise MatiPulseError(f"DWI gradients contain {observed.size} b-values; expected {n}.")
    expected = acquisition_array(payload, "b", nacq=n).astype(float) * 1000.0
    if not np.all(np.isfinite(expected)) or np.any(expected < 0):
        raise MatiPulseError("MATI pulse b-values must be finite and non-negative.")

    matches = np.isclose(
        observed,
        expected,
        atol=float(absolute_tolerance),
        rtol=float(relative_tolerance),
    )
    if np.all(matches):
        return

    mismatches = np.flatnonzero(~matches)
    details = ", ".join(
        f"volume {int(index)}: DWI={observed[index]:g}, MATI={expected[index]:g} s/mm^2"
        for index in mismatches[:5]
    )
    suffix = "" if mismatches.size <= 5 else f" (and {mismatches.size - 5} more)"
    raise MatiPulseError(
        "MATI pulse b-values do not match the DWI volume order within the configured tolerances: "
        f"{details}{suffix}."
    )


def _group_key(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise MatiPulseError("MATI encoding fields used for grouping must be finite.")
        return round(number, 10)
    if isinstance(value, np.integer):
        return int(value)
    return str(value).strip().lower()


def encoding_groups(
    payload: Mapping[str, Any],
    *,
    fields: Sequence[str] = DEFAULT_ENCODING_GROUP_FIELDS,
) -> list[MatiEncodingGroup]:
    n = validate_mati_pulse(payload, required_fields=fields)
    arrays = {field: acquisition_array(payload, field, nacq=n) for field in fields}
    grouped: OrderedDict[tuple[Any, ...], list[int]] = OrderedDict()
    for index in range(n):
        key = tuple(_group_key(arrays[field][index]) for field in fields)
        grouped.setdefault(key, []).append(index)

    out: list[MatiEncodingGroup] = []
    for number, (key, indices) in enumerate(grouped.items(), start=1):
        encoding = {field: _json_ready(value) for field, value in zip(fields, key, strict=True)}
        shape = re.sub(r"[^A-Za-z0-9]+", "", str(encoding.get("shape", "encoding"))) or "encoding"
        out.append(MatiEncodingGroup(f"enc{number:02d}{shape.lower()}", tuple(indices), encoding))
    return out


def subset_mati_pulse(payload: Mapping[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    n = validate_mati_pulse(payload)
    selected = np.asarray(indices, dtype=int).reshape(-1)
    if selected.size == 0:
        raise MatiPulseError("Cannot create an empty MATI pulse subset.")
    if np.any(selected < 0) or np.any(selected >= n):
        raise MatiPulseError("MATI pulse subset contains an out-of-range volume index.")

    out = deepcopy(dict(payload))
    out["Nacq"] = int(selected.size)
    for field in MATI_ACQUISITION_FIELDS:
        if field not in payload:
            continue
        out[field] = _json_ready(acquisition_array(payload, field, nacq=n)[selected])
    return out


def corrected_mati_pulse(
    source_payload: Mapping[str, Any],
    output_order: Sequence[int],
    *,
    bvecs: np.ndarray,
    bvals_smm2: np.ndarray,
) -> dict[str, Any]:
    payload = subset_mati_pulse(source_payload, output_order)
    n = int(payload["Nacq"])
    vectors = np.asarray(bvecs, dtype=float)
    if vectors.shape == (3, n):
        vectors = vectors.T
    if vectors.shape != (n, 3):
        raise MatiPulseError(f"Corrected b-vectors must be 3xN or Nx3; got {vectors.shape}.")
    values = np.asarray(bvals_smm2, dtype=float).reshape(-1)
    if values.size != n:
        raise MatiPulseError(f"Corrected b-values contain {values.size} entries; expected {n}.")
    if not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(values)):
        raise MatiPulseError("Corrected diffusion gradients contain non-finite values.")

    payload["b"] = (values / 1000.0).tolist()
    payload["gdir"] = vectors.tolist()
    payload.pop("G", None)
    return payload


def write_mati_pulse(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_ready(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target
