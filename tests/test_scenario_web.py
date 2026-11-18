"""Regression coverage for the corrected browser's model and keyed view boundary."""
import base64
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "release"))
import fitted
from prepared import prepare
import scenario_web as web


def prediction(fips, spots, eligible=None):
    return pd.DataFrame({
        "FIPS": fips, "YEAR": [2025] * len(fips), "SPB_spots": spots,
        "SPB_per_thousand_acre": spots,
        "Eligible": [True] * len(fips) if eligible is None else eligible,
        "Reason": [""] * len(fips), "Prediction_status": ["predicted"] * len(fips),
        "Risk_category": [0.5] * len(fips),
    })


def upload_payload(model):
    return "data:application/json;base64," + base64.b64encode(json.dumps(model).encode()).decode()


class ScenarioWebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, _ = prepare(ROOT / "examples/demo", "20250101")
        cls.model = json.loads((ROOT / "examples/demo-model.json").read_text())
        cls.geometry = web.local_geometry(ROOT / "examples/demo/counties.geojson")
        cls.app = web.create_app(cls.frame, cls.model, cls.geometry)
        cls.client = cls.app.server.test_client()

    def callback(self, output_contains, values, changed):
        key = next(key for key in self.app.callback_map if output_contains in key)
        entry = self.app.callback_map[key]
        output = entry["output"]
        outputs = ([{"id": item.component_id, "property": item.component_property} for item in output]
                   if isinstance(output, list)
                   else {"id": output.component_id, "property": output.component_property})
        def items(group):
            return [{"id": item["id"], "property": item["property"], "value": values[item["id"] + "." + item["property"]]} for item in entry[group]]
        response = self.client.post("/_dash-update-component", json={
            "output": key, "outputs": outputs, "inputs": items("inputs"),
            "state": items("state"), "changedPropIds": [changed],
        })
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()["response"]

    def test_county_identifiers_keep_leading_zeros_and_reject_bad_keys(self):
        self.assertEqual(web.county_id(1001), "01001")
        self.assertEqual(web.county_id("01001"), "01001")
        for invalid in [None, "ABC", "123456", -1, "1001.5"]:
            with self.assertRaises(ValueError):
                web.county_id(invalid)
        duplicate = pd.concat([self.frame, self.frame.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            web.checked_frame(duplicate)

    def test_delta_is_county_year_joined_instead_of_row_aligned(self):
        base = prediction(["01001", "01003"], [3.0, 7.0])
        scenario = prediction(["01003", "01001"], [70.0, 30.0])
        view = web.keyed_comparison(base, scenario).set_index("FIPS")
        self.assertEqual(view.loc["01001", "delta_spots"], 27.0)
        self.assertEqual(view.loc["01003", "delta_spots"], 63.0)
        self.assertEqual(view.loc["01001", "scenario_risk"], 0.5)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            web.keyed_comparison(pd.concat([base, base.iloc[:1]]), scenario)

    def test_missing_support_is_missing_not_zero_or_a_false_delta(self):
        base = prediction(["01001", "01003"], [3.0, 7.0])
        scenario = prediction(["01003", "01001"], [999.0, 30.0], [False, True])
        result = web.keyed_comparison(base, scenario, ["01001", "01003", "01005"], 2025).set_index("FIPS")
        self.assertFalse(result.loc["01003", "supported"])
        self.assertTrue(np.isnan(result.loc["01003", "scenario_spots"]))
        self.assertTrue(np.isnan(result.loc["01003", "delta_spots"]))
        self.assertFalse(result.loc["01005", "supported"])
        self.assertIn("No prepared", result.loc["01005", "scenario_reason"])

    def test_uploaded_json_is_validated_by_the_fitted_model_contract(self):
        self.assertEqual(web.decode_model_upload(upload_payload(self.model)), self.model)
        bad = copy.deepcopy(self.model)
        del bad["coefficients"]["TMAX_C"]
        for payload in [upload_payload(bad), "data:application/json;base64,??", upload_payload([])]:
            with self.assertRaises(ValueError):
                web.decode_model_upload(payload)
        with self.assertRaisesRegex(ValueError, "2 MiB"):
            web.decode_model_upload("data:application/json;base64," + "A" * (web.MAX_MODEL_BYTES * 2))

    def test_model_store_string_preserves_cli_fingerprint_after_browser_roundtrip(self):
        expected = fitted.export_identity(self.frame, self.model)["model_sha256"]
        # JavaScript JSON serialization erases the distinction between these
        # integer-valued floats and integers when a model is held as an object.
        self.assertIs(type(self.model["transform"]["shift"]), float)
        response = self.callback("active-model.data", {
            "model-upload.contents": upload_payload(self.model),
            "model-upload.filename": "model.json",
        }, "model-upload.contents")
        stored = response["active-model"]["data"]
        self.assertIsInstance(stored, str)
        roundtrip = json.loads(json.dumps(stored))
        self.assertIs(type(json.loads(roundtrip)["transform"]["shift"]), float)
        response = self.callback("scenario-view.data", {
            "active-model.data": roundtrip, "analysis-year.value": 2025,
            "tmax.value": 2, "et.value": 0, "pine.value": 0,
        }, "active-model.data")
        view = response["scenario-view"]["data"]
        self.assertEqual(view["identity"]["model_sha256"], expected)
        response = self.callback("scenario-download.data", {
            "download-button.n_clicks": 1, "scenario-view.data": view,
        }, "download-button.n_clicks")
        table = pd.read_csv(io.StringIO(response["scenario-download"]["data"]["content"]))
        self.assertTrue(table.model_sha256.eq(expected).all())

    def test_string_model_store_is_revalidated_without_trusting_a_client_hash(self):
        tampered = copy.deepcopy(self.model)
        tampered["coefficients"]["TMAX_C"] = "not a coefficient"
        response = self.callback("scenario-view.data", {
            "active-model.data": json.dumps(tampered), "analysis-year.value": 2025,
            "tmax.value": 0, "et.value": 0, "pine.value": 0,
        }, "active-model.data")
        self.assertNotIn("scenario-view", response)
        self.assertIn("could not be evaluated", response["run-message"]["children"])
        for malformed in ['{', '[]']:
            with self.assertRaises(ValueError):
                web.validated_model(malformed)

    def test_target_year_view_and_history_use_the_current_scenario(self):
        before = copy.deepcopy(self.model)
        view = web.comparison_view(self.frame, self.model, 2025, tmax=2, et=10, pine=5)
        self.assertEqual({row["YEAR"] for row in view["records"]}, {2025})
        selected = next(row["FIPS"] for row in view["records"] if row["supported"])
        history = web.county_history(self.frame, self.model, selected, 2025, view["settings"])
        target = history.loc[history.YEAR.eq(2025)].iloc[0]
        mapped = next(row for row in view["records"] if row["FIPS"] == selected)
        self.assertAlmostEqual(target.scenario_spots, mapped["scenario_spots"], places=8)
        historic = history.loc[~history.YEAR.eq(2025)]
        np.testing.assert_allclose(historic.baseline_spots, historic.scenario_spots, equal_nan=True)
        self.assertGreater(abs(target.delta_spots), 0)
        self.assertEqual(self.model, before)
        self.assertIn("scenario_et", mapped)
        self.assertIn("scenario_temperature", mapped)
        self.assertIn("scenario_pine_fraction", mapped)
        self.assertIn("stand_fraction", mapped)

    def test_map_execution_uses_only_selected_year_but_preserves_county_roster(self):
        with patch.object(web.fitted, "predict", wraps=fitted.predict) as predict_spy:
            view = web.comparison_view(self.frame, self.model, 2025, tmax=2)
        self.assertEqual(predict_spy.call_count, 2)
        for call in predict_spy.call_args_list:
            self.assertEqual(set(call.args[0].YEAR), {2025})
        self.assertEqual(len(view["records"]), self.frame.FIPS.nunique())

    def test_county_missing_target_year_retains_history_and_explicit_gap(self):
        selected = self.frame.FIPS.iloc[0]
        frame = self.frame.loc[~(self.frame.FIPS.eq(selected) & self.frame.YEAR.eq(2025))].copy()
        history = web.county_history(frame, self.model, selected, 2025, {"tmax": 2, "et": 10, "pine": 5})
        missing = history.loc[history.YEAR.eq(2025)].iloc[0]
        self.assertFalse(missing.supported)
        self.assertTrue(np.isnan(missing.scenario_spots))
        self.assertTrue(np.isnan(missing.delta_spots))
        other = history.loc[history.YEAR.ne(2025)]
        self.assertEqual(len(other), frame.loc[frame.FIPS.eq(selected)].YEAR.nunique())
        self.assertTrue(other.scenario_spots.notna().any())
        np.testing.assert_allclose(other.baseline_spots, other.scenario_spots, equal_nan=True)
        view = web.comparison_view(frame, self.model, 2025, tmax=2)
        row = next(row for row in view["records"] if row["FIPS"] == selected)
        self.assertFalse(row["supported"])
        self.assertIsNone(row["scenario_spots"])

    def test_missing_model_input_is_reported_by_real_backend(self):
        frame = self.frame.copy()
        selected = frame.FIPS.iloc[0]
        frame.loc[frame.FIPS.eq(selected) & frame.YEAR.eq(2025), "ET_growing"] = np.nan
        view = web.comparison_view(frame, self.model, 2025)
        row = next(row for row in view["records"] if row["FIPS"] == selected)
        self.assertFalse(row["supported"])
        self.assertIsNone(row["delta_spots"])
        self.assertIn("ET_growing", row["scenario_reason"])

    def test_offline_map_supports_county_events_and_relative_risk(self):
        view = web.comparison_view(self.frame, self.model, 2025)
        figure = web.map_figure(view["records"], "scenario_risk", self.geometry).to_dict()
        self.assertTrue(all(trace["type"] in {"scatter", "bar"} for trace in figure["data"]))
        self.assertNotIn("mapbox", figure["layout"])
        self.assertEqual(web.event_county({"points": [{"curveNumber": 0}]}, figure), self.geometry["features"][0]["id"])
        self.assertEqual(web.event_county({"points": [{"customdata": ["01003"]}]}), "01003")
        self.assertIsNone(web.event_county({"points": [{"curveNumber": 9999}]}, figure))
        self.assertIn("FIA fraction", web.FIELDS["stand_fraction"][0])
        fallback = web.map_figure(view["records"], "delta_spots")
        self.assertEqual(fallback.data[0].type, "bar")

    def test_pointer_callbacks_do_not_send_geometry_or_all_county_records(self):
        for key, entry in self.app.callback_map.items():
            if "county-history.figure" in key or "selected-county.value" == key:
                payload_fields = [(item["id"], item["property"]) for item in entry["inputs"] + entry["state"]]
                self.assertNotIn(("county-map", "figure"), payload_fields)
                self.assertNotIn(("scenario-view", "data"), payload_fields)

    def test_http_layout_and_rejected_upload_keep_current_model(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        layout = self.client.get("/_dash-layout").get_data(as_text=True)
        self.assertIn("ET sensitivity", layout)
        self.assertIn("Pine fraction sensitivity", layout)
        bad = copy.deepcopy(self.model)
        bad["schema_version"] = -1
        response = self.callback("active-model.data", {
            "model-upload.contents": upload_payload(bad), "model-upload.filename": "bad.json",
        }, "model-upload.contents")
        self.assertNotIn("active-model", response)
        self.assertIn("Model not loaded", response["model-message"]["children"])

    def test_scenario_callback_revalidates_browser_model_and_download_is_current(self):
        bad = copy.deepcopy(self.model)
        bad["coefficients"]["TMAX_C"] = "invalid"
        response = self.callback("scenario-view.data", {
            "active-model.data": bad, "analysis-year.value": 2025,
            "tmax.value": 2, "et.value": 10, "pine.value": 5,
        }, "active-model.data")
        self.assertNotIn("scenario-view", response)
        self.assertIn("could not be evaluated", response["run-message"]["children"])
        view = web.comparison_view(self.frame, self.model, 2025, tmax=2)
        response = self.callback("scenario-download.data", {
            "download-button.n_clicks": 1, "scenario-view.data": view,
        }, "download-button.n_clicks")
        payload = response["scenario-download"]["data"]
        table = pd.read_csv(io.StringIO(payload["content"]), dtype={"FIPS": str})
        self.assertTrue(table.FIPS.str.fullmatch(r"\d{5}").all())
        self.assertEqual(set(table.YEAR), {2025})
        self.assertTrue(table.tmax_adjustment.eq(2).all())
        self.assertTrue(table.model_sha256.eq(view["identity"]["model_sha256"]).all())
        self.assertTrue(table.input_manifest_sha256.str.fullmatch("[0-9a-f]{64}").all())
        self.assertIn("scenario_risk", table)
        self.assertIn("baseline_risk", table)
        expected = pd.DataFrame(view["records"]).set_index("FIPS").scenario_spots
        np.testing.assert_allclose(table.set_index("FIPS").scenario_spots.sort_index(), expected.sort_index(), equal_nan=True)


if __name__ == "__main__":
    unittest.main()
