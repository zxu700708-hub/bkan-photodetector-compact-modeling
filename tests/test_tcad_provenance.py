from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "artifacts" / "results" / "tcad_provenance"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TcadProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = PROVENANCE / "tcad_provenance_manifest.json"
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))

    def test_primary_counts_and_packaged_hashes(self) -> None:
        primary = self.manifest["primary_campaign"]
        self.assertEqual(primary["condition_count"], 160)
        self.assertEqual(primary["raw_condition_file_count"], 160)
        for key in ("condition_table", "sampling_parameters"):
            item = primary[key]
            path = ROOT / item["packaged_path"]
            self.assertTrue(path.is_file())
            self.assertEqual(sha256(path), item["packaged_sha256"])
            self.assertEqual(item["source_sha256"], item["packaged_sha256"])

    def test_primary_native_archive_is_explicitly_partial(self) -> None:
        counts = self.manifest["native_session_recovery"]["primary_campaign_date_match_counts"]
        self.assertEqual(counts, {"fsp": 4, "ldev": 4, "log": 8})
        with (PROVENANCE / "native_session_inventory.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        primary_rows = [row for row in rows if row["campaign_relation"] == "primary_campaign_date_match"]
        self.assertEqual(len(primary_rows), 16)

    def test_unavailable_items_are_declared(self) -> None:
        unavailable = {
            item["item"] for item in self.manifest["recoverability"] if item["status"] == "unavailable"
        }
        self.assertIn("original random-sampling seed", unavailable)
        self.assertIn("exact source-code revision executed for the primary campaign", unavailable)
        self.assertIn("native sessions and logs for all 160 primary conditions", unavailable)

    def test_public_metadata_is_sanitized(self) -> None:
        forbidden = ("E:\\\\Data", "E:\\Data", "license host:", "Running on host:")
        for path in PROVENANCE.iterdir():
            if path.suffix.lower() not in {".csv", ".json", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in forbidden:
                self.assertNotIn(token, text, msg=f"{token!r} leaked in {path.name}")


if __name__ == "__main__":
    unittest.main()
