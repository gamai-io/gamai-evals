"""phase1_eval_600 is exactly screenspot_300 + uivision_300.

The README lists three sets; two of them are halves of the third. Scoring all three
and presenting them as three independent measurements would count the same items
twice. This test pins the real relationship so the text cannot drift from it.
"""
from __future__ import annotations

import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"


def load(name: str) -> list[dict]:
    data = json.loads((EVALS / name).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("tasks", [])


def keys(items: list[dict]) -> set[str]:
    return {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in items}


class Overlap(unittest.TestCase):
    def test_the_two_300_sets_are_disjoint(self):
        self.assertEqual(len(keys(load("screenspot_300.json")) & keys(load("uivision_300.json"))), 0)

    def test_their_union_is_the_600_set(self):
        union = keys(load("screenspot_300.json")) | keys(load("uivision_300.json"))
        self.assertEqual(len(union), 600)
        self.assertEqual(union, keys(load("phase1_eval_600.json")))

    def test_readme_says_it(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        self.assertIn("union", text,
                      "the README must state that phase1_eval_600 is the union of the two 300-item sets")


if __name__ == "__main__":
    unittest.main()
