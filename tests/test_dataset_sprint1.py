"""Sprint 1 dataset audit and split tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dl_final.data.ham10000 import attach_image_paths, audit_metadata, load_ham10000_metadata
from dl_final.data.splits import check_lesion_leakage, create_lesion_aware_splits

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]


class DatasetSprint1Tests(unittest.TestCase):
    def test_load_metadata_normalizes_dx_to_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / "HAM10000_metadata.csv"
            pd.DataFrame(
                {"image_id": ["ISIC_0000001"], "lesion_id": ["HAM_0001"], "dx": ["nv"]}
            ).to_csv(metadata_path, index=False)

            metadata = load_ham10000_metadata(metadata_path, CLASS_NAMES)

            self.assertEqual(metadata.loc[0, "label"], "nv")
            self.assertEqual(metadata.loc[0, "image_id"], "ISIC_0000001")
            self.assertEqual(metadata.loc[0, "sample_id"], "ISIC_0000001")

    def test_attach_image_paths_and_audit_reports_missing_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            (raw_dir / "ISIC_0000001.jpg").write_bytes(b"path exists")
            metadata = pd.DataFrame(
                {
                    "sample_id": ["ISIC_0000001", "ISIC_0000002"],
                    "image_id": ["ISIC_0000001", "ISIC_0000002"],
                    "lesion_id": ["HAM_0001", "HAM_0002"],
                    "label": ["nv", "mel"],
                }
            )

            with_paths = attach_image_paths(metadata, raw_dir)
            audit = audit_metadata(
                with_paths,
                tmp_path / "metadata.csv",
                raw_dir,
                CLASS_NAMES,
                image_open_sample=0,
            )

            self.assertTrue(with_paths.loc[0, "image_path"].endswith("ISIC_0000001.jpg"))
            self.assertEqual(audit.missing_images, ["ISIC_0000002"])
            self.assertTrue(audit.has_blocking_errors)

    def test_lesion_aware_split_prevents_group_leakage(self) -> None:
        rows = []
        for label in CLASS_NAMES:
            for group_idx in range(6):
                lesion_id = f"{label}_{group_idx}"
                for image_idx in range(2):
                    image_id = f"{lesion_id}_{image_idx}"
                    rows.append(
                        {
                            "sample_id": image_id,
                            "image_id": image_id,
                            "lesion_id": lesion_id,
                            "label": label,
                            "image_path": f"/tmp/{image_id}.jpg",
                        }
                    )
        metadata = pd.DataFrame(rows)

        result = create_lesion_aware_splits(metadata, CLASS_NAMES, seed=7)
        leaks = check_lesion_leakage(result.splits)

        self.assertEqual(leaks, [])
        self.assertEqual(sum(len(frame) for frame in result.splits.values()), len(metadata))
        self.assertEqual(set(result.splits), {"train", "val", "test"})

    def test_unknown_label_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / "HAM10000_metadata.csv"
            pd.DataFrame(
                {"image_id": ["x"], "lesion_id": ["l1"], "dx": ["unknown"]}
            ).to_csv(metadata_path, index=False)

            with self.assertRaisesRegex(ValueError, "outside configured class_names"):
                load_ham10000_metadata(metadata_path, CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()

