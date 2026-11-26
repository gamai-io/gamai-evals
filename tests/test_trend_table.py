"""Pins for tools/trend_table.py: the table is computed, never typed.

Covers the Wilson interval invariants, the known 147/600 and 87/300 cases quoted in
the README, null/zero cells (which must read "not measured", never 0%), and the
`--check` gate that fails when the committed table drifts from the generator.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "trend_table.py"
SOURCE = ROOT / "results" / "cua-trend-2026-11-06.json"
COMMITTED = ROOT / "docs" / "TREND_TABLE.md"

sys.path.insert(0, str(ROOT / "tools"))
import trend_table  # noqa: E402


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


class Wilson(unittest.TestCase):
    def test_interval_contains_rate_and_stays_in_unit_range(self):
        for n in (1, 7, 10, 63, 86, 300, 599, 600, 5000):
            for hits in (0, 1, n // 3, n // 2, n - 1, n):
                low, high = trend_table.wilson_interval(hits, n)
                p = hits / n
                self.assertGreaterEqual(low, 0.0)
                self.assertLessEqual(high, 1.0)
                self.assertLessEqual(low, p, f"{hits}/{n}")
                self.assertGreaterEqual(high, p, f"{hits}/{n}")

    def test_interval_is_wider_for_smaller_n(self):
        widths = {
            n: (lambda ab: ab[1] - ab[0])(trend_table.wilson_interval(int(round(0.245 * n)), n))
            for n in (30, 300, 3000)
        }
        self.assertGreater(widths[30], widths[300])
        self.assertGreater(widths[300], widths[3000])

    def test_known_cases_from_the_readme(self):
        # 147/600 = 24.5%, the matched-budget ground figure.
        low, high = trend_table.wilson_interval(147, 600)
        self.assertAlmostEqual(low, 0.2122800252, places=9)
        self.assertAlmostEqual(high, 0.2809644423, places=9)
        # 87/300 = 29.0%, the matched-budget ScreenSpot figure.
        low, high = trend_table.wilson_interval(87, 300)
        self.assertAlmostEqual(low, 0.2415645461, places=9)
        self.assertAlmostEqual(high, 0.3437455018, places=9)

    def test_rejects_impossible_counts(self):
        with self.assertRaises(ValueError):
            trend_table.wilson_interval(601, 600)
        with self.assertRaises(ValueError):
            trend_table.wilson_interval(0, 0)


class Cells(unittest.TestCase):
    def test_null_cell_is_not_measured(self):
        row = trend_table.cell_row("a", "m", "screenspot", None)
        self.assertIn("not measured", row)
        self.assertNotIn("0.0%", row)

    def test_zero_denominator_is_not_measured(self):
        row = trend_table.cell_row("a", "m", "ground", {"hits": 0, "n": 0})
        self.assertIn("not measured", row)
        self.assertNotIn("0.0%", row)

    def test_odd_denominator_599_is_computed(self):
        row = trend_table.cell_row("a", "m", "ground", {"hits": 70, "n": 599})
        self.assertIn("11.7%", row)
        self.assertIn("9.4%\u201314.5%", row)

    def test_every_json_cell_appears_in_the_table(self):
        data = json.loads(SOURCE.read_text(encoding="utf-8"))
        text = trend_table.render(data, SOURCE)
        for entry in data["results"]:
            for name, cell in entry.items():
                if name in ("adapter", "mode"):
                    continue
                prefix = f"| {entry['adapter']} | {entry.get('mode', '-')} | {name} |"
                self.assertIn(prefix, text)
                if cell is None:
                    self.assertIn(f"{prefix} - | - | not measured", text)

    def test_caveats_are_carried_into_the_table(self):
        data = json.loads(SOURCE.read_text(encoding="utf-8"))
        text = trend_table.render(data, SOURCE)
        for caveat in data["caveats"]:
            self.assertIn(caveat, text)
        self.assertIn("## Caveats", text)
        self.assertIn("## Measurement correction", text)


class Readme(unittest.TestCase):
    def test_quoted_rates_match_the_json(self):
        """The two figures the README bolds must equal hits/n from the JSON."""
        data = json.loads(SOURCE.read_text(encoding="utf-8"))
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cells = {
            (entry["adapter"], name): cell
            for entry in data["results"]
            for name, cell in entry.items()
            if isinstance(cell, dict) and "hits" in cell
        }
        for key, quoted in (
            (("norm-ck4750", "ground_matched_budget"), "24.5%"),
            (("norm-ck4750", "screenspot_matched_budget"), "29.0%"),
            (("norm-ck4750", "ui_vision_matched_budget"), "20.0%"),
        ):
            cell = cells[key]
            self.assertEqual(f"{100.0 * cell['hits'] / cell['n']:.1f}%", quoted)
            self.assertIn(quoted, readme, f"README no longer quotes {quoted} ({key})")


class Check(unittest.TestCase):
    def test_committed_table_is_current(self):
        result = run("--check")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_check_fails_on_stale_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "TREND_TABLE.md"
            self.assertEqual(run("--out", str(target)).returncode, 0)
            self.assertEqual(run("--check", "--out", str(target)).returncode, 0)

            stale = target.read_text(encoding="utf-8").replace("24.5%", "25.5%")
            target.write_text(stale, encoding="utf-8")
            result = run("--check", "--out", str(target))
            self.assertEqual(result.returncode, 1)
            self.assertIn("stale", result.stderr)
            self.assertIn("first difference", result.stderr)

    def test_check_fails_on_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run("--check", "--out", str(pathlib.Path(tmp) / "absent.md"))
            self.assertEqual(result.returncode, 1)
            self.assertIn("missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
