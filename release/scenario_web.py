"""Offline Dash workspace for a validated fitted model and county-year inputs.

All comparisons use county/year keys. Model uploads and browser-held model state
are validated on the server. Scenario settings apply to the selected year; the
county history uses that scenario in that year and the baseline elsewhere.
"""
from __future__ import annotations

import base64
import binascii
import copy
import html as html_escape
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from plotly import graph_objects as go
from plotly.colors import sample_colorscale

try:
    from . import fitted
except ImportError:
    import fitted

KEYS = ["FIPS", "YEAR"]
MAX_MODEL_BYTES = 2 * 1024 * 1024
EXTRA_PREDICTION_FIELDS = {
    "Risk_category": "risk", "Max April Temperature": "temperature",
    "Evap_Transporation": "et", "Pine_density": "pine_fraction",
    "Rain": "rain", "Actual spots": "observed_spots",
}
FIELDS = {
    "scenario_risk": ("Relative risk class", "relative class"),
    "scenario_spots": ("Scenario spots", "spots"),
    "delta_spots": ("Change from baseline", "spots"),
    "baseline_spots": ("Baseline spots", "spots"),
    "scenario_density": ("Scenario spots per 1,000 pine acres", "spots / 1,000 pine acres"),
    "delta_density": ("Change in spots per 1,000 pine acres", "spots / 1,000 pine acres"),
    "scenario_et": ("Seasonal mean ET", "mm / month"),
    "scenario_temperature": ("April maximum temperature", "°C"),
    "scenario_pine_fraction": ("Pine area fraction", "fraction"),
    "stand_fraction": ("FIA fraction (STD_1)", "fraction"),
    "scenario_rain": ("Growing-season precipitation", "mm"),
}
COLORS = {"ink": "#153238", "teal": "#007b79", "muted": "#698087", "missing": "#e2e8e8"}


def county_id(value):
    """Normalize an integer-like county identifier without dropping leading zeros."""
    if pd.isna(value):
        raise ValueError("County FIPS identifiers cannot be missing.")
    if isinstance(value, (int, np.integer)):
        value = str(int(value))
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = str(int(value))
    else:
        value = str(value).strip()
    if not value.isdigit() or not 1 <= len(value) <= 5:
        raise ValueError("County FIPS identifiers must contain one to five digits.")
    return value.zfill(5)


def checked_frame(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Provide a nonempty prepared county-year DataFrame.")
    if any(key not in frame for key in KEYS):
        raise ValueError("Prepared inputs need FIPS and YEAR columns.")
    result = frame.copy(deep=True)
    result["FIPS"] = result["FIPS"].map(county_id)
    years = pd.to_numeric(result["YEAR"], errors="raise")
    if not np.isfinite(years).all() or not (years == years.astype(int)).all():
        raise ValueError("YEAR values must be finite integers.")
    result["YEAR"] = years.astype(int)
    if result.duplicated(KEYS).any():
        raise ValueError("Prepared inputs contain duplicate FIPS/YEAR keys.")
    return result


def validated_model(model):
    """Treat JSON held in a browser store as untrusted on every execution."""
    if isinstance(model, str):
        if len(model.encode("utf-8")) > MAX_MODEL_BYTES:
            raise ValueError("The stored model exceeds 2 MiB.")
        try:
            model = json.loads(model)
        except json.JSONDecodeError as exc:
            raise ValueError("The stored fitted model is not valid JSON.") from exc
    if not isinstance(model, dict):
        raise ValueError("A fitted model must be a JSON object.")
    candidate = copy.deepcopy(model)
    validated = fitted.validate_model(candidate)
    if isinstance(validated, dict):
        candidate = validated
    # A validator may return None after checking the supplied object.
    json.dumps(candidate, allow_nan=False)
    return candidate


def model_json(model):
    """Preserve canonical JSON number spelling across JavaScript round trips.

    A browser-held object turns 1.0 into 1 before returning it to Python. Storing
    the validated artifact as a string preserves its canonical bytes and hence
    the same server-computed model fingerprint used by command-line exports.
    """
    return json.dumps(validated_model(model), sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode_model_upload(contents):
    if not isinstance(contents, str) or "," not in contents:
        raise ValueError("Upload an exported fitted-model JSON file.")
    header, payload = contents.split(",", 1)
    if ";base64" not in header or len(payload) > MAX_MODEL_BYTES * 4 // 3 + 8:
        raise ValueError("The model upload must be base64 JSON of at most 2 MiB.")
    try:
        raw = base64.b64decode(payload, validate=True)
        if len(raw) > MAX_MODEL_BYTES:
            raise ValueError("The model upload exceeds 2 MiB.")
        candidate = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("The uploaded file is not valid UTF-8 JSON.") from exc
    return validated_model(candidate)


def _predictions(frame):
    result = checked_frame(frame)
    required = {"SPB_spots", "SPB_per_thousand_acre"}
    if not required.issubset(result.columns):
        raise ValueError("Prediction output lacks spot-count or spot-density columns.")
    for col in required:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    eligible = result["Eligible"].fillna(False).astype(bool) if "Eligible" in result else True
    result["supported"] = eligible & np.isfinite(result["SPB_spots"]) & np.isfinite(result["SPB_per_thousand_acre"])
    reason = result["Reason"].fillna("").astype(str) if "Reason" in result else pd.Series("", index=result.index)
    result["reason"] = reason
    result.loc[~result["supported"] & result["reason"].eq(""), "reason"] = "Required model inputs or prediction are unavailable."
    result["status"] = result["Prediction_status"].fillna("").astype(str) if "Prediction_status" in result else ""
    for source, target in EXTRA_PREDICTION_FIELDS.items():
        result[target] = pd.to_numeric(result[source], errors="coerce") if source in result else np.nan
    return result[KEYS + ["SPB_spots", "SPB_per_thousand_acre", "supported", "reason", "status"] + list(EXTRA_PREDICTION_FIELDS.values())]


def keyed_comparison(baseline, scenario, expected_fips=None, year=None):
    """Outer join predictions by FIPS/YEAR; unsupported rows never acquire a delta."""
    b = _predictions(baseline).rename(columns={
        "SPB_spots": "baseline_spots", "SPB_per_thousand_acre": "baseline_density",
        "supported": "baseline_supported", "reason": "baseline_reason", "status": "baseline_status",
        **{field: "baseline_" + field for field in EXTRA_PREDICTION_FIELDS.values()},
    })
    s = _predictions(scenario).rename(columns={
        "SPB_spots": "scenario_spots", "SPB_per_thousand_acre": "scenario_density",
        "supported": "scenario_supported", "reason": "scenario_reason", "status": "scenario_status",
        **{field: "scenario_" + field for field in EXTRA_PREDICTION_FIELDS.values()},
    })
    result = b.merge(s, on=KEYS, how="outer", validate="one_to_one")
    if expected_fips is not None:
        if year is None:
            raise ValueError("A comparison year is needed for the expected county roster.")
        roster = pd.DataFrame({"FIPS": sorted({county_id(v) for v in expected_fips}), "YEAR": int(year)})
        result = roster.merge(result, on=KEYS, how="outer", validate="one_to_one")
    for col in ["baseline_supported", "scenario_supported"]:
        result[col] = result[col].eq(True)
    result["supported"] = result["baseline_supported"] & result["scenario_supported"]
    for mode in ["baseline", "scenario"]:
        result[mode + "_reason"] = result[mode + "_reason"].fillna("No prepared county-year record or model prediction.")
        result[mode + "_status"] = result[mode + "_status"].fillna("unavailable")
        for measure in ["spots", "density"]:
            result.loc[~result[mode + "_supported"], mode + "_" + measure] = np.nan
    result["delta_spots"] = (result["scenario_spots"] - result["baseline_spots"]).where(result["supported"])
    result["delta_density"] = (result["scenario_density"] - result["baseline_density"]).where(result["supported"])
    return result.sort_values(KEYS).reset_index(drop=True)


def _finite_setting(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + " must be a finite number, not a boolean.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + " must be finite.")
    return value


def comparison_view(frame, model, year, tmax=0.0, et=0.0, pine=0.0):
    model = validated_model(model)
    numeric_year = _finite_setting(year, "Year")
    if numeric_year != int(numeric_year):
        raise ValueError("Year must be an integer present in the prepared inputs.")
    year = int(numeric_year)
    if year not in set(frame["YEAR"]):
        raise ValueError("Select a year present in the prepared inputs.")
    settings = {name: _finite_setting(value, name) for name, value in [("tmax", tmax), ("et", et), ("pine", pine)]}
    selected_year = frame.loc[frame["YEAR"].eq(year)].copy()
    baseline = fitted.predict(selected_year, model, year, tmax=0.0, et=0.0, pine=0.0)
    scenario = fitted.predict(selected_year, model, year, **settings)
    baseline = baseline.loc[pd.to_numeric(baseline["YEAR"], errors="coerce").eq(year)].copy()
    scenario = scenario.loc[pd.to_numeric(scenario["YEAR"], errors="coerce").eq(year)].copy()
    compared = keyed_comparison(baseline, scenario, frame["FIPS"].unique(), year)
    if set(compared["YEAR"]) != {year}:
        raise ValueError("The prediction backend returned records outside the requested year.")
    if "STD_1" in frame:
        stand = frame.loc[frame["YEAR"].eq(year), KEYS + ["STD_1"]].rename(columns={"STD_1": "stand_fraction"})
        compared = compared.merge(stand, on=KEYS, how="left", validate="one_to_one")
    else:
        compared["stand_fraction"] = np.nan
    return {"year": year, "settings": settings, "identity": fitted.export_identity(frame, model), "records": json.loads(compared.to_json(orient="records", double_precision=15))}


def county_history(frame, model, fips, scenario_year, settings):
    """Use the selected scenario for its year and baseline predictions elsewhere."""
    model = validated_model(model)
    fips = county_id(fips)
    county = frame.loc[frame["FIPS"].eq(fips)].copy()
    if county.empty:
        return pd.DataFrame()
    has_target = county["YEAR"].eq(int(scenario_year)).any()
    baseline_year = int(scenario_year) if has_target else int(county["YEAR"].max())
    baseline = fitted.predict(county, model, baseline_year, tmax=0.0, et=0.0, pine=0.0)
    if not has_target:
        # Keep the numerical history and insert an explicit gap at the missing
        # scenario year; never substitute a neighbouring year's prediction.
        return keyed_comparison(baseline, baseline.copy(), [fips], int(scenario_year))
    scenario = fitted.predict(county, model, int(scenario_year), **settings)
    return keyed_comparison(baseline, scenario)


def local_geometry(geojson):
    if geojson is None:
        return None
    if isinstance(geojson, (str, Path)):
        geojson = json.loads(Path(geojson).read_text())
    geometry = copy.deepcopy(geojson)
    if not isinstance(geometry, dict) or geometry.get("type") != "FeatureCollection":
        raise ValueError("Local geometry must be a GeoJSON FeatureCollection.")
    seen = set()
    for feature in geometry.get("features", []):
        props = feature.get("properties") or {}
        fips = feature.get("id", props.get("FIPS", props.get("GEOID")))
        feature["id"] = county_id(fips)
        if feature["id"] in seen:
            raise ValueError("Local geometry has duplicate county identifiers.")
        seen.add(feature["id"])
    return geometry


def county_names(frame, geojson=None):
    names = {fips: fips for fips in frame["FIPS"].unique()}
    for col in ["NAME", "County", "COUNTYNAME", "County_name"]:
        if col in frame:
            for fips, group in frame.groupby("FIPS", sort=False):
                values = group[col].dropna()
                if not values.empty:
                    names[fips] = str(values.iloc[0])
            break
    if geojson:
        for feature in geojson.get("features", []):
            props = feature.get("properties") or {}
            name = props.get("NAME", props.get("County"))
            if name and feature["id"] in names:
                names[feature["id"]] = str(name)
    return names


def _number(value, digits=2):
    return "Unavailable" if value is None or not np.isfinite(value) else f"{value:,.{digits}f}"


def _empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper", font={"color": COLORS["muted"], "size": 15})
    fig.update_layout(template="plotly_white", margin={"l": 20, "r": 20, "t": 20, "b": 20}, xaxis={"visible": False}, yaxis={"visible": False}, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def map_figure(records, field, geojson=None, names=None, reset=0):
    if field not in FIELDS:
        raise ValueError("Unknown map variable.")
    names = names or {}
    values = pd.DataFrame(records)
    if values.empty:
        return _empty_figure("No county-year records are available.")
    label, unit = FIELDS[field]
    numeric = pd.to_numeric(values[field], errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    delta = field.startswith("delta_")
    lower = -max(float(finite.abs().max()), 1e-9) if delta and len(finite) else (float(finite.min()) if len(finite) else 0.0)
    upper = -lower if delta else (float(finite.max()) if len(finite) else 1.0)
    if field == "scenario_risk":
        lower, upper = 0.0, 3.0
    if field in {"scenario_pine_fraction", "stand_fraction"}:
        lower, upper = 0.0, 1.0
    if upper <= lower:
        upper = lower + 1.0
    scale = "RdBu_r" if delta else "YlOrRd"
    if not geojson:
        ranked = values.copy()
        ranked["_order"] = pd.to_numeric(ranked[field], errors="coerce").abs()
        ranked = ranked.sort_values("_order", ascending=False).head(40)
        fig = go.Figure(go.Bar(
            x=ranked[field], y=[names.get(fips, fips) + " · " + fips for fips in ranked["FIPS"]], orientation="h",
            customdata=[[fips] for fips in ranked["FIPS"]],
            marker={"color": ranked[field], "colorscale": scale, "cmin": lower, "cmax": upper},
            hovertemplate="%{y}<br>%{x:.3f} " + unit + "<extra></extra>",
        ))
        fig.update_layout(yaxis={"autorange": "reversed"}, xaxis_title=label, title={"text": "County comparison · local boundaries not supplied", "font": {"size": 14}})
    else:
        # Cartesian county polygons avoid geographic basemap/TopoJSON downloads.
        # A regional equirectangular projection preserves a useful map aspect.
        county_set = set(values["FIPS"])
        features = [f for f in geojson.get("features", []) if f["id"] in county_set]
        latitudes = []
        for feature in features:
            geom = feature.get("geometry") or {}
            polygons = [geom.get("coordinates", [])] if geom.get("type") == "Polygon" else geom.get("coordinates", [])
            for polygon in polygons:
                if polygon:
                    latitudes.extend(float(p[1]) for p in polygon[0])
        xfactor = math.cos(math.radians(float(np.mean(latitudes)))) if latitudes else 1.0
        indexed = values.set_index("FIPS")
        fig = go.Figure()
        for feature in features:
            fips = feature["id"]
            record = indexed.loc[fips]
            val = record[field]
            supported = val is not None and pd.notna(val) and np.isfinite(float(val))
            fill = sample_colorscale(scale, [min(1.0, max(0.0, (float(val) - lower) / (upper - lower)))])[0] if supported else COLORS["missing"]
            geom = feature.get("geometry") or {}
            if geom.get("type") not in {"Polygon", "MultiPolygon"}:
                continue
            polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
            xs, ys = [], []
            for polygon in polygons:
                for ring in polygon:
                    xs.extend(float(p[0]) * xfactor for p in ring); xs.append(None)
                    ys.extend(float(p[1]) for p in ring); ys.append(None)
            name = html_escape.escape(names.get(fips, fips))
            detail = _number(float(val), 3) + " " + unit if supported else "No model support"
            reason = str(record.get("scenario_reason", ""))
            tooltip = f"<b>{name} · {fips}</b><br>{label}: {detail}"
            if not supported and reason:
                tooltip += "<br>" + html_escape.escape(reason)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", fill="toself", fillcolor=fill,
                line={"color": "#ffffff", "width": 0.7}, name=fips,
                meta={"fips": fips}, customdata=[[fips] for _ in xs],
                hoveron="fills", hovertemplate=tooltip + "<extra></extra>", showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=[None, None], y=[None, None], mode="markers", hoverinfo="skip", showlegend=False,
            marker={"color": [lower, upper], "cmin": lower, "cmax": upper, "colorscale": scale, "showscale": True,
                    "colorbar": {"title": {"text": unit}, "thickness": 12, "len": 0.72, "outlinewidth": 0}},
        ))
        fig.update_xaxes(visible=False, constrain="domain")
        fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        template="plotly_white", margin={"l": 12, "r": 24, "t": 35, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        uirevision=f"county-map-{reset}", font={"family": "Arial, sans-serif", "color": COLORS["ink"]},
    )
    return fig


def event_county(event, figure=None):
    points = event.get("points", []) if isinstance(event, dict) else []
    if not points:
        return None
    point = points[0]
    value = point.get("location")
    custom = point.get("customdata")
    if value is None and isinstance(custom, (list, tuple)) and custom:
        value = custom[0]
    if value is None and figure is not None:
        index = point.get("curveNumber")
        traces = figure.get("data", []) if isinstance(figure, dict) else []
        if isinstance(index, int) and 0 <= index < len(traces):
            value = (traces[index].get("meta") or {}).get("fips")
    try:
        return county_id(value) if value is not None else None
    except ValueError:
        return None


def history_figure(history, name, scenario_year, through_year=None):
    if history.empty:
        return _empty_figure("No history is available for this county.")
    fig = go.Figure()
    for col, label, color, dash in [
        ("baseline_spots", "Baseline", "#819397", "dash"),
        ("scenario_spots", "Selected scenario", COLORS["teal"], "solid"),
    ]:
        fig.add_trace(go.Scatter(
            x=history["YEAR"], y=history[col], name=label, mode="lines+markers", connectgaps=False,
            line={"color": color, "width": 2.5, "dash": dash}, marker={"size": 6},
            hovertemplate="%{x}<br>" + label + ": %{y:.3f} spots<extra></extra>",
        ))
    fig.add_vline(x=int(scenario_year), line_width=1, line_dash="dot", line_color="#becbcb")
    if through_year is not None:
        try:
            through_year = int(through_year)
            if history["YEAR"].min() <= through_year <= history["YEAR"].max():
                fig.add_vrect(x0=history["YEAR"].min() - 0.25, x1=through_year + 0.25, fillcolor="#e9eeee", opacity=0.35, line_width=0, layer="below")
        except (TypeError, ValueError):
            pass
    fig.update_layout(
        template="plotly_white", margin={"l": 52, "r": 16, "t": 18, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": "Year", "dtick": 1 if len(history) < 9 else None}, yaxis_title="Modeled spots",
        legend={"orientation": "h", "y": 1.16, "x": 0}, uirevision="history-" + str(name),
        font={"family": "Arial, sans-serif", "color": COLORS["ink"]},
    )
    return fig


_CSS = """
:root{color-scheme:light;--ink:#153238;--teal:#007b79;--muted:#698087;--line:#dce6e5}
*{box-sizing:border-box}body{margin:0;background:#f3f7f6;color:var(--ink);font-family:Arial,Helvetica,sans-serif}
.shell{max-width:1500px;margin:auto;padding:26px 36px 32px}.topbar,.row,.panel-heading,.brand{display:flex;align-items:center;justify-content:space-between;gap:16px}
.topbar{border-bottom:1px solid var(--line);padding-bottom:21px}.brand{justify-content:flex-start;font-size:22px;font-weight:800;letter-spacing:-.7px}.brand-mark{display:grid;place-items:center;width:35px;height:39px;background:var(--teal);color:white;border-radius:12px 12px 4px 4px;font-size:23px}
.eyebrow{font-size:10px;letter-spacing:1.8px;font-weight:800;color:var(--teal);text-transform:uppercase}.model-status{font-size:12px;color:var(--muted);text-align:right;max-width:540px}.hero{padding:25px 0 23px}h1{font-size:31px;letter-spacing:-1.1px;line-height:1.2;margin:10px 0}h2{font-size:16px;margin:0}p{line-height:1.5}.lede{margin:0;font-size:14px;color:var(--muted);max-width:800px}.workspace{display:grid;grid-template-columns:290px minmax(0,1fr);gap:22px;align-items:start}.panel{background:white;border:1px solid var(--line);border-radius:15px;box-shadow:0 4px 15px #17393a03}.controls{padding:22px}.control-block{padding:18px 0;border-bottom:1px solid #e8eeee}.control-block:last-child{border-bottom:0;padding-bottom:0}.control-label{font-size:12px;font-weight:700;display:block;margin-bottom:12px}.hint{font-size:11px;line-height:1.5;color:var(--muted);margin:9px 0 0}.button{background:white;border:1px solid #cbd9d7;border-radius:8px;padding:10px 14px;font-size:12px;color:var(--ink);font-weight:700;cursor:pointer}.button:hover{border-color:var(--teal);color:var(--teal)}.button-primary{background:var(--teal);border-color:var(--teal);color:white}.button-primary:hover{background:#006765;color:white}.upload-button{width:100%;margin-top:16px}.slider-number{font-size:12px;font-weight:700;color:var(--teal);white-space:nowrap}.controls .dash-slider-root .dash-slider-range{background:var(--teal)}.controls .dash-slider-root .dash-slider-thumb{background:var(--teal)}.controls .dash-slider-mark{font-size:10px}.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:17px}.stat{background:white;border:1px solid var(--line);border-radius:12px;padding:16px 18px}.stat-label{font-size:10px;letter-spacing:.9px;text-transform:uppercase;color:var(--muted);font-weight:700}.stat-value{font-size:27px;font-weight:700;letter-spacing:-.7px;margin:7px 0 3px}.stat-note{font-size:11px;color:var(--muted)}.map-panel{padding:20px 21px 8px}.map-controls{display:flex;gap:10px;align-items:center}.variable{width:230px;font-size:12px}.map-graph{height:380px}.legend-note{font-size:10px;color:var(--muted);padding:0 0 8px}.dot{display:inline-block;width:9px;height:9px;border-radius:3px;background:#e2e8e8;margin-right:5px}.bottom{display:grid;grid-template-columns:minmax(0,1fr) 230px;gap:20px;margin-top:18px}.history-panel{padding:20px 20px 4px;min-width:0}.history-graph{height:255px}.county-picker{font-size:12px;margin-top:15px}.scenario-detail{padding:20px;font-size:12px}.scenario-detail dl{margin:15px 0 0}.scenario-detail dt{color:var(--muted);font-size:11px;margin-top:15px}.scenario-detail dd{margin:5px 0 0;font-weight:700}.notice{font-size:11px;line-height:1.5;color:var(--muted);margin-top:12px;overflow-wrap:anywhere}.notice.error{color:#a23b2b}.support{margin-top:18px;font-size:11px;color:var(--muted)}.support summary{cursor:pointer}.support table{width:100%;text-align:left;border-collapse:collapse;margin-top:12px}.support th,.support td{padding:8px;border-bottom:1px solid var(--line)}.footer{font-size:10px;color:var(--muted);padding-top:22px}.footer strong{font-weight:700;color:#436066}.js-plotly-plot .plotly .modebar{opacity:.6}
@media(max-width:950px){.shell{padding:20px}.workspace{grid-template-columns:1fr}.controls{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.controls>.panel-heading,.controls>#model-upload,.controls>#model-message{grid-column:1/-1}.control-block{border:0;padding:8px}.bottom{grid-template-columns:1fr}.model-status{max-width:50%}}@media(max-width:600px){.controls,.stats{grid-template-columns:1fr}.map-controls{flex-wrap:wrap}.map-graph{height:320px}h1{font-size:26px}.topbar{align-items:flex-start}.variable{width:180px}.shell{padding:14px}}
"""


def _stat(label, value, note):
    return html.Div([html.Div(label, className="stat-label"), html.Div(value, className="stat-value"), html.Div(note, className="stat-note")], className="stat")


def _summary(view):
    data = pd.DataFrame(view["records"])
    support = data["supported"].eq(True)
    scenario = pd.to_numeric(data.loc[support, "scenario_spots"], errors="coerce")
    delta = pd.to_numeric(data.loc[support, "delta_spots"], errors="coerce")
    return [
        _stat("Scenario total", _number(scenario.sum()) if support.any() else "Unavailable", "Modeled spots · supported counties"),
        _stat("Change from baseline", f"{delta.sum():+,.2f}" if support.any() else "Unavailable", "Same county / year comparison"),
        _stat("Model support", f"{int(support.sum()):,} / {len(data):,}", "Counties with both predictions"),
    ]


def _support_details(view, names):
    unsupported = [r for r in view["records"] if not r["supported"]]
    if not unsupported:
        return html.Div("All counties have baseline and scenario model support.", className="support")
    rows = [html.Tr([html.Td(names.get(r["FIPS"], r["FIPS"]) + " · " + r["FIPS"]), html.Td(r["baseline_reason"] if not r["baseline_supported"] else r["scenario_reason"])]) for r in unsupported[:40]]
    return html.Details([
        html.Summary(f"{len(unsupported):,} counties without complete model support"),
        html.Table([html.Thead(html.Tr([html.Th("County"), html.Th("Reason")])), html.Tbody(rows)]),
        html.P("First 40 shown. The CSV contains every county and its support status.") if len(unsupported) > 40 else None,
    ], className="support")


def _scenario_details(view):
    settings = view["settings"]
    return html.Div([
        html.Div("Selected scenario", className="eyebrow"), html.H2(str(view["year"])),
        html.Dl([
            html.Dt("April temperature"), html.Dd(f"{settings['tmax']:+g} °C"),
            html.Dt("ET input sensitivity"), html.Dd(f"{settings['et']:+g}%"),
            html.Dt("Pine fraction sensitivity"), html.Dd(f"{settings['pine']:+g}%"),
        ]),
        html.P("History uses these settings in the selected year and baseline inputs in other years.", className="hint"),
    ])


def create_app(frame, model, geojson=None):
    """Create an independent Dash app without shared mutable scenario state."""
    prepared = checked_frame(frame)
    initial_model = validated_model(model)
    geometry = local_geometry(geojson)
    names = county_names(prepared, geometry)
    years = sorted(int(year) for year in prepared["YEAR"].unique())
    latest_year = years[-1]
    first_county = sorted(names)[0]
    initial_view = comparison_view(prepared, initial_model, latest_year)
    app = Dash(__name__, title="SPB-IVAS · Scenario workspace", update_title="Updating scenario…")
    app.index_string = "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}<style>" + _CSS + "</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"
    app.layout = html.Main([
        dcc.Store(id="active-model", data=model_json(initial_model), storage_type="memory"),
        dcc.Store(id="scenario-view", data=initial_view, storage_type="memory"), dcc.Store(id="map-binding"), dcc.Download(id="scenario-download"),
        html.Header([
            html.Div([html.Div("♧", className="brand-mark"), "SPB-IVAS"], className="brand"),
            html.Div(id="model-summary", className="model-status"),
        ], className="topbar"),
        html.Section([html.H1("County scenario explorer"), html.P(str(prepared.attrs["display_context"]), className="lede") if prepared.attrs.get("display_context") else None], className="hero"),
        html.Div([
            html.Aside([
                html.Div(html.H2("Scenario settings"), className="panel-heading"),
                html.Div([html.Label("Analysis year", className="control-label", htmlFor="analysis-year"), dcc.Dropdown(id="analysis-year", options=[{"label": str(y), "value": y} for y in years], value=latest_year, clearable=False)], className="control-block"),
                html.Div([html.Div([html.Label("April temperature", className="control-label"), html.Span(id="tmax-value", className="slider-number")], className="row"), dcc.Slider(id="tmax", min=-2, max=7, step=0.5, value=0, marks={-2: "−2 °C", 0: "0", 7: "+7 °C"}), html.P("Change in monthly mean daily maximum temperature.", className="hint")], className="control-block"),
                html.Div([html.Div([html.Label("ET sensitivity", className="control-label"), html.Span(id="et-value", className="slider-number")], className="row"), dcc.Slider(id="et", min=-40, max=40, step=5, value=0, marks={-40: "−40%", 0: "0", 40: "+40%"}), html.P("Direct percentage change to the evapotranspiration input.", className="hint")], className="control-block"),
                html.Div([html.Div([html.Label("Pine fraction sensitivity", className="control-label"), html.Span(id="pine-value", className="slider-number")], className="row"), dcc.Slider(id="pine", min=-20, max=20, step=5, value=0, marks={-20: "−20%", 0: "0", 20: "+20%"}), html.P("Relative change to pine area fraction; +10% multiplies it by 1.10.", className="hint")], className="control-block"),
                dcc.Upload(id="model-upload", children=html.Button("Load fitted model JSON", className="button upload-button"), multiple=False, accept=".json,application/json"),
                html.Div("Use an exported model with its saved preprocessing and coefficients.", id="model-message", className="notice"),
            ], className="panel controls"),
            html.Section([
                html.Div(_summary(initial_view), id="summary-stats", className="stats"),
                html.Div([
                    html.Div([html.Div([html.Div("County comparison", className="eyebrow"), html.H2(id="map-title", children=str(latest_year))]), html.Div([dcc.Dropdown(id="map-variable", options=[{"label": title, "value": key} for key, (title, _) in FIELDS.items()], value="scenario_risk", clearable=False, className="variable"), html.Button("Recenter", id="recenter", n_clicks=0, className="button"), html.Button("Export CSV", id="download-button", n_clicks=0, className="button button-primary")], className="map-controls")], className="panel-heading"),
                    dcc.Loading(dcc.Graph(id="county-map", figure=map_figure(initial_view["records"], "scenario_risk", geometry, names), className="map-graph", config={"displaylogo": False, "scrollZoom": True}), type="circle", color=COLORS["teal"]),
                    html.Div([html.Span(className="dot"), "Gray counties have no value for the displayed variable. Hover to inspect; click to select."], className="legend-note"),
                    html.Div(_support_details(initial_view, names), id="support-details"),
                ], className="panel map-panel"),
                html.Div([
                    html.Div([html.Div([html.Div("County history", className="eyebrow"), html.H2(id="history-title", children="Select a county")]), dcc.Dropdown(id="selected-county", options=[{"label": names[f] + " · " + f, "value": f} for f in sorted(names)], value=first_county, clearable=False, className="county-picker"), dcc.Loading(dcc.Graph(id="county-history", figure=_empty_figure("Loading county history…"), className="history-graph", config={"displaylogo": False}), type="circle", color=COLORS["teal"])], className="panel history-panel"),
                    html.Div(_scenario_details(initial_view), id="scenario-details", className="panel scenario-detail"),
                ], className="bottom"),
                html.Div(id="run-message", className="notice error"),
            ]),
        ], className="workspace"),
        html.Footer([html.Strong("Scenario interpretation. "), "Values describe the fitted model's response to input changes. They are not calibrated outbreak probabilities or evidence of a causal treatment effect."], className="footer"),
    ], className="shell")

    # Filled Plotly polygons have no pointNumber. Dash's default Graph event
    # serializer drops their metadata, so relay their explicit county identity.
    # This bridge does not compute predictions or accept model state.
    app.clientside_callback(
        """function(figure) {
            let tries = 0;
            function bind() {
                const plot = document.querySelector('#county-map .js-plotly-plot');
                if (!plot || typeof plot.on !== 'function') {
                    if (++tries < 60) window.setTimeout(bind, 50);
                    return;
                }
                if (plot.__spbIvasCountyEvents) return;
                plot.__spbIvasCountyEvents = true;
                function relay(event, property) {
                    const point = event && event.points && event.points[0];
                    const fips = point && point.data && point.data.meta && point.data.meta.fips;
                    if (!fips || point.pointNumber != null) return;
                    const update = {};
                    update[property] = {points: [{location: String(fips), curveNumber: point.curveNumber}]};
                    window.dash_clientside.set_props('county-map', update);
                }
                plot.on('plotly_hover', event => relay(event, 'hoverData'));
                plot.on('plotly_click', event => relay(event, 'clickData'));
            }
            window.setTimeout(bind, 0);
            return window.dash_clientside.no_update;
        }""",
        Output("map-binding", "data"), Input("county-map", "figure"),
    )

    @app.callback(Output("active-model", "data"), Output("model-message", "children"), Output("model-message", "className"), Input("model-upload", "contents"), State("model-upload", "filename"), prevent_initial_call=True)
    def upload_model(contents, filename):
        try:
            uploaded = decode_model_upload(contents)
            return model_json(uploaded), f"Loaded {filename or 'exported model'}; server validation passed.", "notice"
        except (ValueError, TypeError, KeyError) as exc:
            return no_update, "Model not loaded: " + str(exc), "notice error"

    @app.callback(Output("model-summary", "children"), Input("active-model", "data"))
    def describe_model(model_state):
        try:
            artifact = validated_model(model_state)
            through = artifact.get("training", {}).get("through_year")
            label = artifact.get("name") or artifact.get("model_name") or "Fitted regression model"
            return str(label) + (f" · training through {through}" if through is not None else " · saved preprocessing")
        except (ValueError, TypeError, KeyError):
            return "Model validation required"

    @app.callback(Output("scenario-view", "data"), Output("run-message", "children"), Input("active-model", "data"), Input("analysis-year", "value"), Input("tmax", "value"), Input("et", "value"), Input("pine", "value"))
    def run_scenario(model_state, year, tmax, et, pine):
        try:
            return comparison_view(prepared, model_state, year, tmax, et, pine), ""
        except (ValueError, TypeError, KeyError) as exc:
            return no_update, "Scenario could not be evaluated: " + str(exc)

    @app.callback(Output("county-map", "figure"), Output("map-title", "children"), Output("summary-stats", "children"), Output("support-details", "children"), Output("scenario-details", "children"), Output("tmax-value", "children"), Output("et-value", "children"), Output("pine-value", "children"), Input("scenario-view", "data"), Input("map-variable", "value"), Input("recenter", "n_clicks"))
    def render_view(view, field, reset):
        settings = view["settings"]
        return (map_figure(view["records"], field, geometry, names, reset), str(view["year"]), _summary(view), _support_details(view, names), _scenario_details(view), f"{settings['tmax']:+g} °C", f"{settings['et']:+g}%", f"{settings['pine']:+g}%")

    @app.callback(Output("selected-county", "value"), Input("county-map", "clickData"), prevent_initial_call=True)
    def select_county(click):
        selected = event_county(click)
        return selected if selected in names else no_update

    @app.callback(Output("county-history", "figure"), Output("history-title", "children"), Input("analysis-year", "value"), Input("tmax", "value"), Input("et", "value"), Input("pine", "value"), Input("selected-county", "value"), Input("county-map", "hoverData"), Input("active-model", "data"))
    def render_history(year, tmax, et, pine, selected, hover, model_state):
        hovered = event_county(hover) if ctx.triggered_id == "county-map" else None
        selected = hovered if hovered in names else selected
        if selected not in names:
            return _empty_figure("Select a county with prepared inputs."), "County history"
        try:
            history = county_history(prepared, model_state, selected, year, {"tmax": tmax, "et": et, "pine": pine})
            through = validated_model(model_state).get("training", {}).get("through_year")
            return history_figure(history, selected, year, through), names[selected] + " · " + selected
        except (ValueError, TypeError, KeyError) as exc:
            return _empty_figure(str(exc)), names[selected] + " · " + selected

    @app.callback(Output("scenario-download", "data"), Input("download-button", "n_clicks"), State("scenario-view", "data"), prevent_initial_call=True)
    def download_comparison(clicks, view):
        if not clicks:
            return no_update
        data = pd.DataFrame(view["records"])
        data["FIPS"] = data["FIPS"].map(county_id)
        for name, value in view["identity"].items():
            data[name] = value
        for name, value in view["settings"].items():
            data[name + "_adjustment"] = value
        return dcc.send_data_frame(data.to_csv, f"spb-ivas-{view['year']}-scenario.csv", index=False)

    return app
