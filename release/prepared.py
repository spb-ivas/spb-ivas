"""Audited county/year preparation for the corrected SPB-IVAS workflow.

``prepare(data_dir, revision)`` returns an untransformed feature table and a
JSON-serializable audit. It never fills spots, estimates model state, or writes
inputs. Missing observations remain NaN. Duplicate conflicting observations
are quarantined, not averaged. Seasonal ET and precipitation require all eight
months (March--October). WaSSI ET is expected in mm per month; no 25.4 conversion
is applied. The legacy ET header does not itself prove upstream unit provenance.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["FIPS", "YEAR"]
FEATURES = ["TMAX_C", "ET_growing", "PINE_DENSITY", "STD_1", "PINE_ACRES"]
SEASON = tuple(range(3, 11))
FIA_FIELDS = ["PINE_DENSITY", "STD_1", "PINE_ACRES"]
CLIMATE_FIELDS = ["TMAX_C", "TMEAN_C", "TMIN_C", "PPT_MM"]
CLIMATE_ALIASES = {
    "TMAX_C": "TMAX_C", "TMAX": "TMAX_C", "TMEAN_C": "TMEAN_C",
    "TAVG_C": "TMEAN_C", "TMEAN": "TMEAN_C", "TAVG": "TMEAN_C",
    "TMIN_C": "TMIN_C", "TMIN": "TMIN_C", "PPT_MM": "PPT_MM", "PPT": "PPT_MM",
}


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(frame, columns):
    """Represent source keys/values without NumPy scalars or NaN JSON literals."""
    return frame.loc[:, columns].astype(object).where(frame[columns].notna(), None).to_dict("records")


def _load(directory, prefix, revision, audit, *, optional=False, wassi=False):
    names = ([f"{prefix}_{revision}_p0.csv", f"{prefix}_{revision}.csv"] if wassi
             else [f"{prefix}_{revision}.csv"])
    # Unsuffixed schema/demo files are supported only when no tagged file exists.
    names.append(f"{prefix}.csv")
    path = next((directory / name for name in names if (directory / name).is_file()), None)
    if path is None:
        if not optional:
            raise FileNotFoundError(f"Missing {prefix} input; expected one of {names}")
        audit["missing_optional_inputs"].append(prefix)
        return pd.DataFrame(columns=KEYS + (["MONTH"] if prefix.startswith("prism") else []))
    audit["input_sha256"][path.name] = _hash(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    header = [name.strip() for name in header]
    if not header:
        raise ValueError(f"{path.name}: CSV header is missing")
    if len(header) != len(set(header)):
        raise ValueError(f"{path.name}: duplicated column names")
    try:
        frame = pd.read_csv(path, dtype={"FIPS": "string"})
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"{path.name}: CSV header is missing") from error
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError(f"{path.name}: duplicated column names")
    audit["inputs"][prefix] = {"filename": path.name, "rows": len(frame)}
    if frame.empty:
        audit["empty_inputs"].append(prefix)
    return frame


def _keys(frame, table, *, monthly=False):
    needed = KEYS + (["MONTH"] if monthly else [])
    absent = [name for name in needed if name not in frame]
    if absent:
        raise ValueError(f"{table}: missing key columns {absent}")
    frame = frame.copy()
    text = frame.FIPS.astype("string").str.strip()
    # Integral decimal serialization is accepted; nonnumeric/partial keys are not.
    text = text.str.replace(r"\.0+$", "", regex=True)
    valid = text.str.fullmatch(r"\d{1,5}", na=False)
    text = text.str.zfill(5)
    valid &= text.ne("00000") & text.str[-3:].ne("000")
    if not valid.all():
        raise ValueError(f"{table}: invalid FIPS at source rows {list(frame.index[~valid][:10] + 2)}")
    frame["FIPS"] = text.astype(str)
    for name, lower, upper in [("YEAR", 1, 9999)] + ([("MONTH", 1, 12)] if monthly else []):
        number = pd.to_numeric(frame[name], errors="coerce")
        valid = number.notna() & np.isfinite(number) & number.between(lower, upper) & number.eq(number.round())
        if not valid.all():
            raise ValueError(f"{table}: invalid {name} at source rows {list(frame.index[~valid][:10] + 2)}")
        frame[name] = number.astype("int64")
    return frame


def _unit_check(frame, table):
    allowed = {
        "ET": {"mm", "mm/month", "mmpermonth", "mmmonth-1", "mm/monthly"},
        "AET": {"mm", "mm/month", "mmpermonth", "mmmonth-1", "mm/monthly"},
        "PPT": {"mm", "mm/month", "mmpermonth"},
        "TMAX": {"c", "degc", "°c", "celsius"},
        "TMEAN": {"c", "degc", "°c", "celsius"},
        "TMIN": {"c", "degc", "°c", "celsius"},
        "PINE_ACRES": {"acre", "acres"},
        "PINE_DENSITY": {"fraction", "proportion", "1"},
        "STD_1": {"fraction", "proportion", "1"},
        "SPOTS": {"count", "counts", "spots"},
    }
    for name in frame:
        match = re.fullmatch(r"(.+)_UNITS?", name)
        if not match:
            continue
        field = match[1]
        field = {"ET_MM": "ET", "AET_MM": "AET", "PPT_MM": "PPT",
                 "TMAX_C": "TMAX", "TMEAN_C": "TMEAN", "TAVG_C": "TMEAN", "TMIN_C": "TMIN"}.get(field, field)
        if field in allowed:
            values = set(frame[name].dropna().astype(str).str.lower().str.replace(" ", "", regex=False)) - {""}
            bad = values - allowed[field]
            if bad:
                raise ValueError(f"{table}: unsupported units for {field}: {sorted(bad)}")


def _numeric(frame, field, table, audit, *, minimum=None, maximum=None, integer=False):
    original = frame[field]
    value = pd.to_numeric(original, errors="coerce")
    present = original.notna() & original.astype(str).str.strip().ne("")
    valid = value.notna() & np.isfinite(value)
    if minimum is not None:
        valid &= value.ge(minimum)
    if maximum is not None:
        valid &= value.le(maximum)
    if integer:
        valid &= value.eq(value.round())
    bad = present & ~valid
    if bad.any():
        bad_frame = frame.loc[bad, KEYS + (["MONTH"] if "MONTH" in frame else [])].copy()
        bad_frame["field"] = field
        bad_frame["value"] = original.loc[bad].astype(str)
        audit["invalid_values"].setdefault(table, []).extend(_records(bad_frame, list(bad_frame)))
    frame[field] = value.where(valid, np.nan).astype(float)


def _unique(frame, fields, table, audit, *, monthly=False):
    """Resolve known/missing duplicates; conflicting known values become NaN."""
    keys = KEYS + (["MONTH"] if monthly else [])
    frame = frame[keys + fields].copy()
    exact = frame.duplicated(keys + fields)
    audit["duplicates"][table] = {"identical_rows_removed": int(exact.sum()), "duplicate_key_excess": int(frame.duplicated(keys).sum())}
    frame = frame.loc[~exact]
    if frame.empty:
        return frame
    grouped = frame.groupby(keys, sort=True, dropna=False)
    result = grouped[fields].first()
    conflicts = []
    for field in fields:
        counts = grouped[field].nunique(dropna=True)
        for key in counts[counts.gt(1)].index:
            key = key if isinstance(key, tuple) else (key,)
            values = sorted(float(x) for x in grouped.get_group(key)[field].dropna().unique())
            item = dict(zip(keys, key))
            item.update(field=field, values=values)
            conflicts.append(item)
            result.loc[key, field] = np.nan
    audit["conflicts"][table] = conflicts
    return result.reset_index()


def _monthly(frame, table, audit):
    frame = _keys(frame, table, monthly=True)
    _unit_check(frame, table)
    found = {}
    for name, canonical in CLIMATE_ALIASES.items():
        if name in frame:
            if canonical in found:
                raise ValueError(f"{table}: competing aliases for {canonical}")
            found[canonical] = name
    result = frame[KEYS + ["MONTH"]].copy()
    for field in CLIMATE_FIELDS:
        result[field] = frame[found[field]] if field in found else np.nan
        _numeric(result, field, table, audit, minimum=0 if field == "PPT_MM" else None)
    return _unique(result, CLIMATE_FIELDS, table, audit, monthly=True)


def _cpc(frame, audit):
    if "MONTH" in frame:
        return _monthly(frame, "cpc", audit)
    frame = _keys(frame, "cpc")
    _unit_check(frame, "cpc")
    pieces = []
    for name in frame:
        match = re.fullmatch(r"(TMAX|TMEAN|TAVG|TMIN|PPT)(?:_(?:C|MM))?_(\d{1,2})", name)
        if not match:
            continue
        month = int(match[2])
        if month not in range(1, 13):
            raise ValueError(f"cpc: invalid month in field {name}")
        field = CLIMATE_ALIASES[match[1]]
        piece = frame[KEYS].copy()
        piece["MONTH"] = month
        piece[field] = frame[name]
        pieces.append(piece)
    wide = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=KEYS + ["MONTH"])
    # Different fields from a wide row are coalesced at a monthly key. Duplicate
    # contradictory rows remain conflicts rather than pivot_table averages.
    return _monthly(wide, "cpc", audit)


def _season(frame, field, output):
    available = frame.loc[frame.MONTH.isin(SEASON)]
    grouped = available.groupby(KEYS)[field]
    result = grouped.mean() if output == "ET_growing" else grouped.sum(min_count=8)
    result = result.where(grouped.count().eq(8))
    return result.rename(output).reset_index()


def prepare(data_dir, revision):
    """Return unique raw county/year features and a JSON-safe preparation audit.

    Stable PRISM values supersede provisional values field by field. Available
    explicit CPC monthly fields then supersede PRISM, including TMAX only if a
    CPC TMAX field actually exists. Within-source conflicts block that field.
    FIA uses the most recent same-county snapshot at or before the output year;
    an incomplete newer snapshot does not borrow a future or older field.
    """
    directory = Path(data_dir).expanduser().resolve()
    revision = str(revision)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", revision):
        raise ValueError("Revision must contain only letters, digits, dots, underscores, or hyphens")
    audit = {
        "schema_version": 1, "revision": revision, "input_sha256": {}, "inputs": {},
        "missing_optional_inputs": [], "empty_inputs": [], "duplicates": {},
        "conflicts": {}, "invalid_values": {},
        "policies": {
            "climate_precedence": "valid CPC field > stable PRISM field > provisional PRISM field; within-source conflicting fields quarantined",
            "fia": "most recent same-county snapshot at or before year; missing snapshot fields stay missing",
            "spots": "raw zero preserved; missing stays missing; identical duplicates collapse; conflicting known counts quarantined",
            "previous_spots": "exact same-county calendar year minus one, with no fill",
            "growing_season": "March through October; all eight monthly values required",
            "domain": "union of county/year keys in supplied source tables; missing model features retained",
        },
        "units": {"TMAX_C": "degC", "ET_growing": "mm/month", "PPT_growing": "mm", "PINE_ACRES": "acre", "PINE_DENSITY": "fraction", "STD_1": "fraction", "SPOTS": "count", "PREV_SPOTS": "count"},
        "unit_assumptions": ["Unsuffixed WaSSI ET is interpreted as monthly millimetres; legacy headers cannot independently verify upstream provenance."],
        "std_1_definition": "Opaque FIA-derived fraction: upstream category definition not established by supplied local evidence; not climate standard deviation.",
    }
    stable = _monthly(_load(directory, "prism_stable", revision, audit), "prism_stable", audit)
    provisional = _monthly(_load(directory, "prism_provisional", revision, audit, optional=True), "prism_provisional", audit)
    cpc = _cpc(_load(directory, "cpc", revision, audit, optional=True), audit)
    monthly_keys = KEYS + ["MONTH"]
    climate = provisional
    # A within-source conflict blocks lower-priority fallback. A valid field in
    # a documented higher-priority source can supersede that conflict.
    for table, source in (("prism_stable", stable), ("cpc", cpc)):
        climate = source.set_index(monthly_keys).combine_first(climate.set_index(monthly_keys)).reset_index()
        for conflict in audit["conflicts"].get(table, []):
            mask = np.ones(len(climate), dtype=bool)
            for key in monthly_keys:
                mask &= climate[key].eq(conflict[key]).to_numpy()
            climate.loc[mask, conflict["field"]] = np.nan

    wassi = _keys(_load(directory, "wassi", revision, audit, wassi=True), "wassi", monthly=True)
    _unit_check(wassi, "wassi")
    et_columns = [name for name in ("ET", "ET_MM", "AET_MM") if name in wassi]
    if len(et_columns) != 1:
        raise ValueError("wassi: need exactly one monthly ET field: ET, ET_MM, or AET_MM")
    wassi = wassi.rename(columns={et_columns[0]: "ET"})
    _numeric(wassi, "ET", "wassi", audit, minimum=0)
    wassi = _unique(wassi, ["ET"], "wassi", audit, monthly=True)

    fia = _keys(_load(directory, "fia", revision, audit), "fia")
    _unit_check(fia, "fia")
    missing = [name for name in FIA_FIELDS if name not in fia]
    if missing:
        raise ValueError(f"fia: missing required feature columns {missing}")
    for field in FIA_FIELDS:
        _numeric(fia, field, "fia", audit, minimum=0, maximum=None if field == "PINE_ACRES" else 1)
    fia = _unique(fia, FIA_FIELDS, "fia", audit)

    spots = _keys(_load(directory, "spots", revision, audit), "spots")
    _unit_check(spots, "spots")
    if "SPOTS" not in spots:
        raise ValueError("spots: missing required SPOTS column")
    _numeric(spots, "SPOTS", "spots", audit, minimum=0, integer=True)
    spots = _unique(spots, ["SPOTS"], "spots", audit)

    domain = pd.concat([table[KEYS] for table in (climate, wassi, fia, spots)], ignore_index=True).drop_duplicates()
    domain = domain.sort_values(["YEAR", "FIPS"]).reset_index(drop=True)
    if domain.empty:
        raise ValueError("No county/year records are available in the supplied inputs")
    if fia.empty:
        prepared = domain.assign(**{field: np.nan for field in FIA_FIELDS}, FIA_YEAR=np.nan)
    else:
        prepared = pd.merge_asof(domain, fia.rename(columns={"YEAR": "FIA_YEAR"}).sort_values(["FIA_YEAR", "FIPS"]),
                                 left_on="YEAR", right_on="FIA_YEAR", by="FIPS", direction="backward")
    audit["fia_carried_rows"] = int(prepared.FIA_YEAR.lt(prepared.YEAR).sum())
    audit["counties_without_fia_records"] = sorted(set(domain.FIPS) - set(fia.FIPS))
    april = climate.loc[climate.MONTH.eq(4), KEYS + ["TMAX_C"]]
    prepared = prepared.merge(april, on=KEYS, how="left", validate="one_to_one")
    prepared = prepared.merge(_season(wassi, "ET", "ET_growing"), on=KEYS, how="left", validate="one_to_one")
    prepared = prepared.merge(_season(climate, "PPT_MM", "PPT_growing"), on=KEYS, how="left", validate="one_to_one")
    prepared = prepared.merge(spots, on=KEYS, how="left", validate="one_to_one")
    previous = spots.rename(columns={"SPOTS": "PREV_SPOTS"}).copy()
    previous["YEAR"] += 1
    prepared = prepared.merge(previous, on=KEYS, how="left", validate="one_to_one")
    columns = KEYS + FEATURES + ["PREV_SPOTS", "SPOTS", "PPT_growing"]
    prepared = prepared[columns].sort_values(KEYS).reset_index(drop=True)
    prepared["FIPS"] = prepared.FIPS.astype(str).str.zfill(5)
    prepared["YEAR"] = prepared.YEAR.astype("int64")
    audit["rows"] = len(prepared)
    audit["counties"] = int(prepared.FIPS.nunique())
    audit["years"] = [int(prepared.YEAR.min()), int(prepared.YEAR.max())]
    audit["missing_values"] = {name: int(prepared[name].isna().sum()) for name in columns if name not in KEYS}
    audit["rows_missing_model_inputs"] = int(prepared[FEATURES + ["PREV_SPOTS"]].isna().any(axis=1).sum())
    prepared.attrs["input_revision"] = revision
    payload = json.dumps(audit["input_sha256"], sort_keys=True, separators=(",", ":")).encode()
    prepared.attrs["input_manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    return prepared, audit
