import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.download_data import copy_fer_csv


FER_CSV = b"emotion,pixels,Usage\n0,0 1 2 3,Training\n"


class FERDownloadTests(unittest.TestCase):
    def test_extracts_csv_from_download_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "download.zip"
            output_path = Path(directory) / "fer2013.csv"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("fer2013.csv", FER_CSV)

            self.assertTrue(copy_fer_csv(archive_path, output_path))
            self.assertEqual(output_path.read_bytes(), FER_CSV)

    def test_extracts_csv_from_nested_gzip_tar(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "download.zip"
            output_path = Path(directory) / "fer2013.csv"
            compressed_tar = io.BytesIO()
            with tarfile.open(fileobj=compressed_tar, mode="w:gz") as archive:
                member = tarfile.TarInfo("fer2013.csv")
                member.size = len(FER_CSV)
                archive.addfile(member, io.BytesIO(FER_CSV))
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("fer2013.tar.gz", compressed_tar.getvalue())

            self.assertTrue(copy_fer_csv(archive_path, output_path))
            self.assertEqual(output_path.read_bytes(), FER_CSV)


if __name__ == "__main__":
    unittest.main()
