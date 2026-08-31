"""Resolve every DOI in the bibliography against its registrant record.

An integrity review found two fabricated entries in this file: one whose DOI
resolved to an unrelated paper on groundwater in Tanzania, and one whose DOI did
not resolve at all and whose correct record had been deleted to settle a BibTeX
key collision. Both were cited in the manuscript, one of them four times, and
both had survived because nobody had ever asked a resolver.

This asks. For each entry with a DOI it fetches the Crossref record and compares
the registrant's title and first author against what the file claims. It reports
three kinds of failure, and they mean different things:

  UNRESOLVED  the DOI does not exist. The entry is unusable as it stands.
  MISMATCH    the DOI exists and describes a different paper. This is the
              dangerous one, because the entry looks fine and the citation
              silently points a reader somewhere else.
  DRIFT       same paper, but the title, journal or year in the file disagrees
              with the registrant. Usually a transcription slip.

Titles are compared on a normalised word set rather than by string equality,
because BibTeX braces, LaTeX escapes and subtitle punctuation differ routinely
between a file and a registrant record without either being wrong.

Requires network access. Entries without a DOI are listed and not checked; that
is a limit of the method, not a pass.

Usage:
  python models/scripts/analysis/verify_bibliography.py
  python models/scripts/analysis/verify_bibliography.py --bib path/to/refs.bib
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BIB = ROOT / ".docs" / "papers" / "5" / "refs.bib"
API = "https://api.crossref.org/works/"
UA = "AnchorGate-bib-check/1.0 (mailto:manuelricardo.perez@uptc.edu.co)"
# Below this Jaccard overlap of title words we call it a different paper.
MISMATCH_BELOW = 0.34
DRIFT_BELOW = 0.75


def entries(text):
    """(key, fields) for each @type{key, ...} block."""
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        start = m.end()
        depth, i = 1, m.start(0) + text[m.start(0):].index("{")
        i += 1
        while i < len(text) and depth:
            depth += (text[i] == "{") - (text[i] == "}")
            i += 1
        body = text[start:i - 1]
        f = {}
        for fm in re.finditer(r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*(?=\w+\s*=|$)",
                              body, re.S):
            f[fm.group(1).lower()] = " ".join(fm.group(2).split())
        out.append((m.group(2), f))
    return out


def words(s):
    # GMD asks data citations to carry a "[data set]" suffix that no registrant
    # record contains, so it is stripped before comparison rather than counted
    # as a discrepancy.
    s = re.sub(r"\[(data set|code|dataset)\]", " ", s or "", flags=re.I)
    # Braces are removed WITHOUT inserting a space: BibTeX case protection is
    # written {H}ybrid, and splitting there turns one word into two and makes a
    # correct entry look like a different paper.
    s = re.sub(r"[{}$]", "", s or "")
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    stop = {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with", "at"}
    return {w for w in s.split() if w and w not in stop}


def overlap(a, b):
    wa, wb = words(a), words(b)
    return len(wa & wb) / max(1, len(wa | wb))


def crossref(doi):
    req = urllib.request.Request(API + urllib.parse.quote(doi),
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"]


def datacite(doi):
    """Data-set DOIs are registered with DataCite, not Crossref.

    Crossref returns 404 for a perfectly good DataCite DOI, so a checker that
    only asks Crossref reports every data citation as unresolved and trains its
    reader to ignore it. Both registries are asked before anything is called bad.
    """
    req = urllib.request.Request(
        "https://api.datacite.org/dois/" + urllib.parse.quote(doi),
        headers={"User-Agent": UA, "Accept": "application/vnd.api+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        a = json.load(r)["data"]["attributes"]
        titles = [t.get("title", "") for t in a.get("titles", [])]
        return {"title": titles or [""],
                "issued": {"date-parts": [[a.get("publicationYear")]]}}


def resolve(doi):
    """Crossref first, DataCite second; raise only if neither has it."""
    try:
        return crossref(doi)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    return datacite(doi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", type=Path, default=DEFAULT_BIB)
    args = ap.parse_args()
    if not args.bib.exists():
        print(f"no bibliography at {args.bib}")
        return 1

    all_entries = entries(args.bib.read_text(encoding="utf-8"))
    withdoi = [(k, f) for k, f in all_entries if f.get("doi")]
    nodoi = [k for k, f in all_entries if not f.get("doi")]
    print(f"{len(all_entries)} entries, {len(withdoi)} with a DOI, "
          f"{len(nodoi)} without\n")

    bad, drift, unres = [], [], []
    for i, (key, f) in enumerate(withdoi, 1):
        doi = f["doi"].strip().rstrip(".")
        try:
            rec = resolve(doi)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                unres.append((key, doi))
                print(f"  UNRESOLVED  {key:<26} {doi}")
            else:
                print(f"  (http {e.code})  {key:<26} {doi}")
            continue
        except Exception as e:                                # noqa: BLE001
            print(f"  (error)      {key:<26} {doi}  {type(e).__name__}")
            continue
        theirs = (rec.get("title") or [""])[0]
        ov = overlap(f.get("title", ""), theirs)
        if ov < MISMATCH_BELOW:
            bad.append((key, doi, f.get("title", ""), theirs))
            print(f"  MISMATCH    {key:<26} {doi}")
            print(f"              file: {f.get('title', '')[:96]}")
            print(f"              doi : {theirs[:96]}")
        elif ov < DRIFT_BELOW:
            drift.append((key, doi, f.get("title", ""), theirs))
            print(f"  DRIFT       {key:<26} {doi}")
            print(f"              file: {f.get('title', '')[:96]}")
            print(f"              doi : {theirs[:96]}")
        else:
            yr = str((rec.get("issued", {}).get("date-parts") or [[None]])[0][0])
            # Crossref's `issued` is the earliest date the publisher deposited,
            # which for a journal with online-first publication is the year
            # before the print volume. A one-year gap is that, not an error; a
            # larger gap is a transcription slip worth seeing.
            if f.get("year") and yr and yr.isdigit() and f["year"].isdigit() \
                    and abs(int(f["year"]) - int(yr)) > 1:
                drift.append((key, doi, f"year {f['year']}", f"year {yr}"))
                print(f"  DRIFT       {key:<26} year {f['year']} vs {yr} at DOI")
        if i % 8 == 0:
            time.sleep(0.4)

    print()
    print("=" * 70)
    print(f"{len(bad)} mismatched, {len(unres)} unresolved, {len(drift)} drifted")
    if nodoi:
        print(f"\n{len(nodoi)} entries carry no DOI and were not checked:")
        print("  " + ", ".join(sorted(nodoi)))
    if bad or unres:
        print("\nA mismatched DOI points a reader at a different paper than the "
              "one\nthe sentence cites. Fix these before the bibliography ships.")
        return 1
    return 0


if __name__ == "__main__":
    import urllib.parse                                        # noqa: E402
    sys.exit(main())
