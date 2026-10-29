"""Offline regression tests for CERNBox downloads."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.sh"

FAKE_CURL = r'''#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

args = sys.argv[1:]
url = next(arg for arg in args if arg.startswith("https://"))
prefix = "/remote.php/dav/public-files/" + os.environ["SHARE_TOKEN"] + "/"
path = unquote(urlsplit(url).path)
assert path.startswith(prefix), path
rel = path[len(prefix):]
method = "PROPFIND" if "PROPFIND" in args else ("HEAD" if "-I" in args else "GET")
with open(os.environ["CURL_LOG"], "a", encoding="utf-8") as log:
    log.write(method + " " + rel + "\n")

if method == "PROPFIND":
    if os.environ.get("FAIL_PROPFIND"):
        sys.exit(22)
    particle = "mu-" if "/mu-/" in rel else "e-"
    base = "SIREN_files/water/" + particle + "/"
    dirs = [base, base + "cherenkov/", base + "cherenkov/trained_model/",
            base + "dedx/", base + "dedx/trained_model/"]
    files = [base + "cherenkov/trained_model/photonsim_siren_weights.npz",
             base + "cherenkov/config.json",
             base + "dedx/trained_model/dedx_siren_weights.npz"]
    entries = []
    for entry in dirs + files:
        is_dir = entry.endswith("/")
        resource_type = "<d:collection/>" if is_dir else ""
        entries.append(f'<d:response><d:href>{prefix}{entry}</d:href>'
                       f'<d:propstat><d:prop><d:resourcetype>{resource_type}'
                       f'</d:resourcetype><d:getcontentlength>0</d:getcontentlength>'
                       f'</d:prop></d:propstat></d:response>')
    print('<d:multistatus xmlns:d="DAV:">' + "".join(entries) + '</d:multistatus>')
    sys.exit(0)

if method == "HEAD":
    print("HTTP/1.1 200 OK\r\nContent-Length: 12\r\n")
    sys.exit(0)

if rel.endswith("/") or rel == os.environ.get("FAIL_FILE"):
    print("curl: (22) The requested URL returned error: 501", file=sys.stderr)
    sys.exit(22)
output = Path(args[args.index("-o") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_bytes(b"valid data!!")
'''


class DownloadDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        shutil.copy2(SCRIPT, self.repo / "scripts" / "download_data.sh")
        bindir = self.base / "bin"
        bindir.mkdir()
        curl = bindir / "curl"
        curl.write_text(FAKE_CURL, encoding="utf-8")
        curl.chmod(0o755)
        self.log = self.base / "curl.log"
        self.env = os.environ.copy()
        self.env.update(PATH=str(bindir) + os.pathsep + self.env["PATH"], CURL_LOG=str(self.log))

    def run_download(self, *args, **env):
        return subprocess.run(
            ["bash", str(self.repo / "scripts" / "download_data.sh"), *map(str, args)],
            env={**self.env, **env}, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )

    def test_skips_webdav_directories_and_links_models(self):
        result = self.run_download()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Download complete.", result.stdout)
        for particle in ("muon", "electron"):
            root = self.repo / "data" / "water" / particle
            self.assertTrue((root / "1000MeV_100events.root").is_file())
            self.assertTrue((root / "siren_training/trained_model/photonsim_siren_weights.npz").is_file())
            self.assertTrue((root / "dedx_siren_training/trained_model/dedx_siren_weights.npz").is_file())
            self.assertTrue((self.repo / "data/wbls" / particle / "siren_training").is_symlink())
        requests = self.log.read_text().splitlines()
        self.assertFalse([line for line in requests if line.startswith("GET ") and line.endswith("/")])

    def test_missing_auxiliary_file_causes_failure(self):
        result = self.run_download(FAIL_FILE="SIREN_files/water/mu-/cherenkov/config.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed downloads: 1", result.stdout)
        self.assertIn("MISS", result.stdout)
        self.assertNotIn("Download complete.", result.stdout)

    def test_existing_real_data_is_not_deleted(self):
        original = self.repo / "data/water/muon/siren_training"
        original.mkdir(parents=True)
        sentinel = original / "important.txt"
        sentinel.write_text("keep me", encoding="utf-8")
        result = self.run_download("--store-dir", self.base / "external")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to replace existing file/directory", result.stdout)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me")

    def test_repeated_download_keeps_valid_symlinks(self):
        first = self.run_download()
        self.assertEqual(first.returncode, 0, first.stdout)
        second = self.run_download()
        self.assertEqual(second.returncode, 0, second.stdout)
        self.assertIn("[skip]", second.stdout)
        self.assertTrue((self.repo / "data/wbls/muon/siren_training").is_symlink())

    def test_failed_listing_is_fatal(self):
        result = self.run_download(FAIL_PROPFIND="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Download complete.", result.stdout)


if __name__ == "__main__":
    unittest.main()
