# Depositing the archive and getting the version DOI

GMD requires a persistent DOI at submission, not at acceptance, and does not accept
material offered on request. The record therefore has to exist before the manuscript
goes in.

**The GitHub release route does not work here.** `.gitignore` excludes
`models/output/`, `data/output/`, `*.npy` and `*.csv`, so a release-triggered Zenodo
deposit archives the code and none of the inputs it reads. That is what the v1.0.x
records contain, and it is the fatal finding the last review round raised. The
deposit has to be built and uploaded directly.

## 1. Build the record

```
python models/scripts/analysis/deposit_manifest.py --build d:/tmp/anchorbench_deposit
```

This resolves what the archive must contain from the paths the analysis scripts
actually open, rather than from a list kept by hand, and writes five files:

| file | size |
|------|------|
| `anchorbench-v1.3.0-code.zip` | 124 MB |
| `anchorbench-v1.3.0-predictions-and-metrics.zip` | 260 MB |
| `complete_dataset_...clean.nc` | 504 MB |
| `anchorbench-v1.3.0-checkpoints.zip` | 49 MB |
| `README.md`, `MANIFEST.sha256` | small |

937 MB in total, against a Zenodo limit of 50 GB per record.

## 2. Upload it

Get a personal access token from
<https://zenodo.org/account/settings/applications/> with the `deposit:write` and
`deposit:actions` scopes, then:

```
set ZENODO_TOKEN=...
python zenodo/upload.py --dir d:/tmp/anchorbench_deposit
```

The script opens a **new version** of the existing concept record
(10.5281/zenodo.21576208), so the concept DOI keeps resolving to the latest version
and the citation in the earlier papers stays valid. It clears the files inherited
from the previous version, writes the metadata from `.zenodo.json`, uploads the five
files, prints the reserved version DOI, and **stops without publishing**.

Add `--sandbox` (with `ZENODO_SANDBOX_TOKEN`) to rehearse on Zenodo's test instance
first. Nothing from the sandbox carries over.

## 3. Review and publish

Open the draft, check the file list and the metadata, and publish from that page.
Publishing is irreversible: the files become public and the DOI becomes citable.

If you would rather not use a token, the same thing can be done by hand: open the
existing record, choose "New version", remove the inherited files, upload the five
files from the build directory, and copy the metadata from `.zenodo.json`. The
script exists because that metadata is long and retyping it is where records go
wrong.

## 4. Put the DOI in the manuscript

Edit `\dataavailabilityDOI` near the top of `.docs/papers/5/paper_gmd.tex`:

```latex
\newcommand{\dataavailabilityDOI}{\url{https://doi.org/10.5281/zenodo.NNNNNNNN}}
```

Use the **version** DOI that step 2 printed, not the concept DOI. The Code and data
availability section already explains the difference and cites both.

Then:

```
python models/scripts/analysis/manuscript_numbers_audit.py
```

It fails while the placeholder is present and passes once the DOI is in, which is
the check that this step was not forgotten.

## 5. Tag the code

The Zenodo record is the archive of record, but the repository should carry a tag at
the same commit so the two can be lined up later:

```
git tag -a v1.3.0 -m "AnchorBench v1.0, GMD submission archive"
```

Pushing the tag is yours to do.
