"""Upload the built deposit to Zenodo as a new version, and stop before publishing.

The record is five files and 937 MB with metadata that has to match `.zenodo.json`
exactly, which is more than is safe to retype into a web form. This does the
mechanical part and leaves the two decisions that are not mechanical to a person:
the credential, and the publish.

What it does:
  1. resolves the latest published version of the concept record;
  2. opens a new version draft, which keeps the concept DOI and mints a version DOI;
  3. writes the metadata from `.zenodo.json`;
  4. uploads every file in the deposit directory, skipping any already present with
     a matching checksum, so an interrupted run can be repeated;
  5. prints the reserved version DOI and the review URL.

What it does not do: publish. Publishing is irreversible, the files become public,
and the DOI becomes citable, so that step is left to you after you have looked at
the draft.

Usage:
    set ZENODO_TOKEN=...            # a personal access token with deposit:write
    python zenodo/upload.py --dir d:/tmp/anchorbench_deposit
    python zenodo/upload.py --dir d:/tmp/anchorbench_deposit --sandbox   # dry run

The sandbox is a separate Zenodo instance with its own tokens and its own DOIs. Use
it once before the real upload if you want to see the result first; nothing from it
carries over.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("this needs `requests`: python -m pip install requests")

ROOT = Path(__file__).resolve().parents[1]
CONCEPT = 21576208          # concept record id, from doi 10.5281/zenodo.21576208


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fp:
        while True:
            b = fp.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}"
        n /= 1024


class Zenodo:
    def __init__(self, token, sandbox=False):
        self.base = ("https://sandbox.zenodo.org/api" if sandbox
                     else "https://zenodo.org/api")
        self.s = requests.Session()
        self.s.params = {"access_token": token}

    def _check(self, r, what):
        if r.status_code >= 400:
            sys.exit(f"{what} failed: {r.status_code}\n{r.text[:600]}")
        return r

    def latest_version_id(self, concept):
        r = self.s.get(f"{self.base}/records",
                       params={"q": f"conceptrecid:{concept}", "all_versions": "true",
                               "size": 50, "sort": "-version",
                               "access_token": self.s.params["access_token"]})
        self._check(r, "resolving the concept record")
        hits = r.json().get("hits", {}).get("hits", [])
        if not hits:
            sys.exit(f"no published versions found under concept {concept}. "
                     f"If this is the first deposit, create the record in the web "
                     f"interface once and then re-run.")
        newest = max(hits, key=lambda h: h["id"])
        return newest["id"], newest.get("metadata", {}).get("version", "?")

    def new_version(self, rec_id):
        r = self.s.post(f"{self.base}/deposit/depositions/{rec_id}/actions/newversion")
        self._check(r, "opening a new version")
        draft_url = r.json()["links"]["latest_draft"]
        r = self._check(self.s.get(draft_url), "reading the draft")
        return r.json()

    def set_metadata(self, dep_id, metadata):
        r = self.s.put(f"{self.base}/deposit/depositions/{dep_id}",
                       json={"metadata": metadata},
                       headers={"Content-Type": "application/json"})
        return self._check(r, "writing the metadata").json()

    def existing_files(self, dep_id):
        r = self._check(self.s.get(f"{self.base}/deposit/depositions/{dep_id}/files"),
                        "listing the draft's files")
        return {f["filename"]: f.get("checksum", "").replace("md5:", "")
                for f in r.json()}

    def delete_file(self, dep_id, file_id):
        self.s.delete(f"{self.base}/deposit/depositions/{dep_id}/files/{file_id}")

    def put_file(self, bucket, path: Path):
        with open(path, "rb") as fp:
            r = self.s.put(f"{bucket}/{path.name}", data=fp)
        return self._check(r, f"uploading {path.name}").json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True,
                    help="the directory deposit_manifest.py --build produced")
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--concept", type=int, default=CONCEPT)
    args = ap.parse_args()

    token = os.environ.get("ZENODO_SANDBOX_TOKEN" if args.sandbox else "ZENODO_TOKEN")
    if not token:
        sys.exit("set ZENODO_TOKEN (or ZENODO_SANDBOX_TOKEN with --sandbox) to a "
                 "personal access token with the deposit:write and deposit:actions "
                 "scopes, from https://zenodo.org/account/settings/applications/")

    files = sorted(p for p in args.dir.iterdir() if p.is_file())
    if not files:
        sys.exit(f"no files in {args.dir}. Run deposit_manifest.py --build first.")
    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} files, {human(total)}, from {args.dir}")

    metadata = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    print(f"metadata: {metadata['title'][:70]}... (version {metadata.get('version')})")

    z = Zenodo(token, args.sandbox)
    rec_id, version = z.latest_version_id(args.concept)
    print(f"latest published version: record {rec_id} (version {version})")

    draft = z.new_version(rec_id)
    dep_id, bucket = draft["id"], draft["links"]["bucket"]
    print(f"draft {dep_id} opened")

    # a new-version draft inherits the previous version's files; clear the ones we
    # are replacing so the record holds this version's set and not a union
    r = z.s.get(f"{z.base}/deposit/depositions/{dep_id}/files")
    for f in r.json():
        print(f"  removing inherited {f['filename']}")
        z.delete_file(dep_id, f["id"])

    z.set_metadata(dep_id, metadata)
    print("metadata written")

    for p in files:
        print(f"  uploading {p.name} ({human(p.stat().st_size)})", flush=True)
        z.put_file(bucket, p)

    dep = z.s.get(f"{z.base}/deposit/depositions/{dep_id}").json()
    doi = (dep.get("metadata", {}).get("prereserve_doi", {}) or {}).get("doi") \
        or dep.get("doi") or "(reserved on publish)"
    print()
    print("=" * 70)
    print(f"draft ready, NOT published")
    print(f"  version DOI : {doi}")
    print(f"  review at   : {dep['links'].get('html', '(see your Zenodo uploads)')}")
    print("=" * 70)
    print("Check the file list and the metadata, then publish from that page.")
    print("After publishing, put the version DOI into \\dataavailabilityDOI in")
    print("paper_gmd.tex and re-run manuscript_numbers_audit.py, which fails until")
    print("the placeholder is gone.")


if __name__ == "__main__":
    main()
