import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.preprocessing.data_loader import FERDataLoader


class FERDataLoaderTests(unittest.TestCase):
    def write_dataset(self, directory, rows):
        csv_path = Path(directory) / "fer2013.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("emotion", "pixels", "Usage"))
            writer.writerows(rows)
        return csv_path

    def test_loads_pixels_labels_and_standard_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = self.write_dataset(
                directory,
                [
                    (0, "0 1 2 3", "Training"),
                    (3, "4 5 6 7", "PublicTest"),
                    (6, "8 9 10 11", "PrivateTest"),
                ],
            )

            images, labels, splits = FERDataLoader(csv_path, img_size=2).load()

        self.assertEqual(images.shape, (3, 2, 2, 1))
        self.assertEqual(images.dtype, np.uint8)
        np.testing.assert_array_equal(images[0, :, :, 0], [[0, 1], [2, 3]])
        np.testing.assert_array_equal(labels, [0, 3, 6])
        np.testing.assert_array_equal(splits, ["train", "validation", "test"])

    def test_rejects_rows_with_wrong_image_size(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = self.write_dataset(directory, [(0, "0 1 2", "Training")])

            with self.assertRaisesRegex(ValueError, "expected 4"):
                FERDataLoader(csv_path, img_size=2).load()

    def test_rejects_labels_outside_fer_class_range(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = self.write_dataset(directory, [(7, "0 1 2 3", "Training")])

            with self.assertRaisesRegex(ValueError, "between 0 and 6"):
                FERDataLoader(csv_path, img_size=2).load()


if __name__ == "__main__":
    unittest.main()
