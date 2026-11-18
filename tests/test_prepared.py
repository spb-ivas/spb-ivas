"""Regression tests for data integrity at the model preparation boundary."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "release"))
from prepared import prepare


REVISION = "20250101"


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.climate = [{"FIPS": fips, "YEAR": year, "MONTH": month, "TMAX_C": 20.0, "PPT_MM": 10.0}
                        for fips in (1001, 1003) for year in (2020, 2021, 2022) for month in range(1, 13)]
        self.water = [{"FIPS": row["FIPS"], "YEAR": row["YEAR"], "MONTH": row["MONTH"], "ET": 40.0}
                      for row in self.climate]
        self.fia = [{"FIPS": 1001, "YEAR": 2020, "PINE_DENSITY": 0.4, "STD_1": 0.3, "PINE_ACRES": 10000.0}]
        self.spots = [{"FIPS": 1001, "YEAR": 2020, "SPOTS": 0}, {"FIPS": 1001, "YEAR": 2021, "SPOTS": 5}]
        self.write("prism_stable", self.climate)
        self.write("wassi", self.water, suffix="_p0")
        self.write("fia", self.fia)
        self.write("spots", self.spots)

    def write(self, prefix, rows, suffix=""):
        pd.DataFrame(rows).to_csv(self.folder / f"{prefix}_{REVISION}{suffix}.csv", index=False)

    def get(self):
        return prepare(self.folder, REVISION)

    def row(self, frame, fips="01001", year=2021):
        return frame.loc[frame.FIPS.eq(fips) & frame.YEAR.eq(year)].iloc[0]

    def test_zero_missing_and_exact_previous_calendar_year_remain_distinct(self):
        frame, audit = self.get()
        self.assertEqual(self.row(frame, year=2020).SPOTS, 0)
        self.assertEqual(self.row(frame, year=2021).PREV_SPOTS, 0)
        self.assertEqual(self.row(frame, year=2022).PREV_SPOTS, 5)
        self.assertTrue(np.isnan(self.row(frame, year=2022).SPOTS))
        self.assertTrue(np.isnan(self.row(frame, "01003").SPOTS))
        self.assertFalse(frame.duplicated(["FIPS", "YEAR"]).any())
        self.assertTrue(all(frame.FIPS.str.len().eq(5)))
        json.dumps(audit, allow_nan=False)

    def test_gap_is_not_previous_row_lag(self):
        self.write("spots", [self.spots[0], {"FIPS": 1001, "YEAR": 2022, "SPOTS": 9}])
        frame, _ = self.get()
        self.assertTrue(np.isnan(self.row(frame, year=2022).PREV_SPOTS))

    def test_identical_spots_deduplicate_but_conflicts_and_their_lags_quarantine(self):
        self.write("spots", self.spots + [self.spots[0], {"FIPS": 1001, "YEAR": 2021, "SPOTS": 6}])
        frame, audit = self.get()
        self.assertEqual(audit["duplicates"]["spots"]["identical_rows_removed"], 1)
        self.assertEqual(audit["conflicts"]["spots"], [{"FIPS": "01001", "YEAR": 2021, "field": "SPOTS", "values": [5.0, 6.0]}])
        self.assertTrue(np.isnan(self.row(frame).SPOTS))
        self.assertTrue(np.isnan(self.row(frame, year=2022).PREV_SPOTS))

    def test_fia_carry_is_same_county_and_past_only(self):
        self.write("fia", self.fia + [{"FIPS": 1003, "YEAR": 2022, "PINE_DENSITY": 0.7, "STD_1": 0.8, "PINE_ACRES": 70000.0}])
        frame, audit = self.get()
        self.assertEqual(self.row(frame, year=2022).PINE_ACRES, 10000)
        self.assertTrue(np.isnan(self.row(frame, "01003", 2021).PINE_ACRES))
        self.assertEqual(self.row(frame, "01003", 2022).PINE_ACRES, 70000)
        self.assertEqual(audit["fia_carried_rows"], 2)

    def test_incomplete_new_fia_snapshot_is_not_filled_from_another_snapshot(self):
        self.write("fia", self.fia + [{"FIPS": 1001, "YEAR": 2021, "PINE_DENSITY": 0.6, "STD_1": np.nan, "PINE_ACRES": 60000.0}])
        frame, _ = self.get()
        self.assertTrue(np.isnan(self.row(frame, year=2022).STD_1))
        self.assertEqual(self.row(frame, year=2022).PINE_ACRES, 60000)

    def test_all_eight_et_months_required_and_units_not_multiplied(self):
        self.write("wassi", [r for r in self.water if not (r["FIPS"] == 1001 and r["YEAR"] == 2021 and r["MONTH"] == 10)], suffix="_p0")
        frame, _ = self.get()
        self.assertTrue(np.isnan(self.row(frame).ET_growing))
        self.assertEqual(self.row(frame, year=2020).ET_growing, 40)
        self.assertEqual(self.row(frame, year=2020).PPT_growing, 80)

    def test_header_only_provisional_and_unsuffixed_wassi_fallback(self):
        pd.DataFrame(columns=["FIPS", "YEAR", "MONTH", "TMAX_C", "PPT_MM"]).to_csv(self.folder / f"prism_provisional_{REVISION}.csv", index=False)
        (self.folder / f"wassi_{REVISION}_p0.csv").rename(self.folder / f"wassi_{REVISION}.csv")
        frame, audit = self.get()
        self.assertEqual(self.row(frame).ET_growing, 40)
        self.assertIn("prism_provisional", audit["empty_inputs"])
        self.assertIn(f"wassi_{REVISION}.csv", audit["input_sha256"])
        self.assertNotIn("prism_provisional", audit["missing_optional_inputs"])

    def test_stable_then_cpc_overlay_never_converts_tmean_to_tmax(self):
        self.write("prism_provisional", [{"FIPS": 1001, "YEAR": 2021, "MONTH": 4, "TMAX_C": 99, "PPT_MM": 77}])
        self.write("cpc", [{"FIPS": 1001, "YEAR": 2021, "TMEAN_4": 33, "PPT_4": 30}])
        frame, _ = self.get()
        self.assertEqual(self.row(frame).TMAX_C, 20)
        self.assertEqual(self.row(frame).PPT_growing, 100)
        self.write("cpc", [{"FIPS": 1001, "YEAR": 2021, "TMEAN_4": 33, "TMAX_4": 27}])
        frame, _ = self.get()
        self.assertEqual(self.row(frame).TMAX_C, 27)

    def test_future_cpc_rows_retained_without_inventing_future_temperature(self):
        self.write("cpc", [{"FIPS": 1001, "YEAR": 2026, "TMEAN_4": 21, "PPT_4": 30}])
        frame, _ = self.get()
        self.assertTrue(np.isnan(self.row(frame, year=2026).TMAX_C))
        self.assertEqual(self.row(frame, year=2026).PINE_ACRES, 10000)

    def test_conflicting_month_is_not_averaged_or_completed_from_another_source(self):
        self.write("prism_stable", self.climate + [{"FIPS": 1001, "YEAR": 2021, "MONTH": 4, "TMAX_C": 30, "PPT_MM": 10}])
        self.write("prism_provisional", [{"FIPS": 1001, "YEAR": 2021, "MONTH": 4, "TMAX_C": 25, "PPT_MM": 10}])
        frame, audit = self.get()
        self.assertTrue(np.isnan(self.row(frame).TMAX_C))
        self.assertEqual(audit["conflicts"]["prism_stable"][0]["values"], [20.0, 30.0])

    def test_higher_priority_valid_sources_supersede_lower_priority_conflicts(self):
        self.write("prism_provisional", [{"FIPS": 1001, "YEAR": 2021, "MONTH": 4, "TMAX_C": 25},
                                         {"FIPS": 1001, "YEAR": 2021, "MONTH": 4, "TMAX_C": 26}])
        frame, audit = self.get()
        self.assertEqual(self.row(frame).TMAX_C, 20)
        self.assertEqual(len(audit["conflicts"]["prism_provisional"]), 1)
        self.write("prism_stable", self.climate + [{"FIPS": 1001, "YEAR": 2021, "MONTH": 4, "TMAX_C": 30, "PPT_MM": 10}])
        self.write("cpc", [{"FIPS": 1001, "YEAR": 2021, "TMAX_4": 27}])
        frame, _ = self.get()
        self.assertEqual(self.row(frame).TMAX_C, 27)
        self.write("cpc", [{"FIPS": 1001, "YEAR": 2021, "TMAX_4": 27},
                           {"FIPS": 1001, "YEAR": 2021, "TMAX_4": 28}])
        frame, _ = self.get()
        self.assertTrue(np.isnan(self.row(frame).TMAX_C))

    def test_no_unsubstantiated_temperature_et_or_modern_year_caps(self):
        self.write("prism_stable", [{"FIPS": 1001, "YEAR": 1750, "MONTH": 4, "TMAX_C": 75}])
        self.write("wassi", [{"FIPS": 1001, "YEAR": 1750, "MONTH": month, "ET": 2500} for month in range(3, 11)], suffix="_p0")
        frame, _ = self.get()
        self.assertEqual(self.row(frame, year=1750).TMAX_C, 75)
        self.assertEqual(self.row(frame, year=1750).ET_growing, 2500)

    def test_duplicate_headers_and_units_on_explicitly_named_columns(self):
        path = self.folder / f"prism_provisional_{REVISION}.csv"
        path.write_text("FIPS,YEAR,MONTH,TMAX_C,TMAX_C\n1001,2021,4,20,30\n")
        with self.assertRaisesRegex(ValueError, "duplicated column names"):
            self.get()
        self.write("prism_provisional", [{"FIPS": 1001, "YEAR": 2021, "MONTH": 4,
                                         "TMAX_C": 21, "TMAX_C_UNITS": "fahrenheit"}])
        with self.assertRaisesRegex(ValueError, "unsupported units"):
            self.get()
        path.unlink()
        self.write("wassi", [dict(r, ET_MM=r["ET"], ET_MM_UNITS="inches") for r in self.water], suffix="_p0")
        with self.assertRaisesRegex(ValueError, "unsupported units"):
            self.get()

    def test_invalid_values_are_audited_and_declared_wrong_units_rejected(self):
        self.write("spots", [{"FIPS": 1001, "YEAR": 2020, "SPOTS": -1}, {"FIPS": 1001, "YEAR": 2021, "SPOTS": 2.5}])
        frame, audit = self.get()
        self.assertTrue(self.row(frame).isna()["SPOTS"])
        self.assertEqual(len(audit["invalid_values"]["spots"]), 2)
        wrong = self.water.copy()
        wrong = [dict(r, ET_UNITS="inches") for r in wrong]
        self.write("wassi", wrong, suffix="_p0")
        with self.assertRaisesRegex(ValueError, "unsupported units"):
            self.get()

    def test_bad_keys_missing_required_input_and_missing_header_rejected(self):
        self.write("spots", [{"FIPS": "bad", "YEAR": 2020, "SPOTS": 1}])
        with self.assertRaisesRegex(ValueError, "invalid FIPS"):
            self.get()
        self.write("spots", self.spots)
        self.write("cpc", [{"FIPS": 1001, "YEAR": 2021.5, "PPT_4": 30}])
        with self.assertRaisesRegex(ValueError, "invalid YEAR"):
            self.get()
        (self.folder / f"cpc_{REVISION}.csv").unlink()
        (self.folder / f"fia_{REVISION}.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self.get()
        self.write("fia", self.fia)
        (self.folder / f"prism_provisional_{REVISION}.csv").write_text("")
        with self.assertRaisesRegex(ValueError, "header is missing"):
            self.get()


if __name__ == "__main__":
    unittest.main()
