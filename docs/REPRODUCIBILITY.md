# Reproducibility

There are three separate checks: reading the saved results, testing the implementation, and reproducing a live experiment. Only the first two have been verified locally during repository preparation.

## Saved results

```bash
python scripts/verify_results.py
```

This standard-library-only command is read-only. It uses the existing analysis loader to check the four methods and their paired task IDs, checks for environment-error rows, and compares summary counts/reward/success/steps/calls against the raw records. The expected dataset is 100 unique valid episodes per method and 100 paired task IDs.

```bash
python analysis/analyze_formal.py
```

This regenerates tables, confidence intervals, paired comparisons, interaction estimates and SVG figures under `results/formal/analysis/`. It uses a fixed seed (`20260801`), 20,000 bootstrap samples, and 100,000 randomization samples. The file is self-contained and uses the Python standard library. Resampling is more expensive than basic verification.

## Local implementation

Install the package from the repository root, then run:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m webshop_baseline --help
```

Use a virtual environment for installation. The repository preparation check passed all 18 existing unit tests; no live API request was used. The included CI workflow targets Windows and Ubuntu with Python 3.10 and 3.12. Its presence alone is not evidence that a hosted run has passed.

## Recorded live-run settings

These values come from the committed configuration and experiment metadata, not a fresh verification of upstream service availability.

| Setting | Recorded value |
|:--|:--|
| Model identifier | `deepseek-v4-flash` |
| Temperature | `0` |
| Action / auxiliary token limits | `100` / `220` |
| Task split | Development: `fixed_0`–`fixed_9`; formal: `fixed_10`–`fixed_109` |
| Action budget | 15 calls to the action policy |
| Reflection limit / midpoint | 2 calls / step 8 |
| WebShop commit | `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd` |
| Dataset mirror | `YWZBrandon/webshop-data` |
| Dataset revision | `ce990fff5aee388db2706f07820c578ab68e0453` |
| Environment patch label | `full-streaming-ijson-v1` |

## Missing environment asset

The historical full-data streaming patch is referenced in the metadata but is **not present** in this repository. The WebShop server, dataset and search index are also not bundled. The repository therefore supports inspection and statistical reanalysis of the saved results, but does not currently provide a complete from-scratch deployment of the original live environment.

Use the upstream [WebShop repository](https://github.com/princeton-nlp/WebShop) for its deployment instructions. Recovering the exact patch and validating a fresh full-data deployment remain outstanding. Do not silently substitute another environment and label its results an exact reproduction.

## Running without changing saved formal results

Use `configs/baseline_frozen.yaml` for a one-task development check. For new formal runs, copy a formal YAML file within `configs/` and change both `experiment.results_file` and `experiment.summary_file` to a new directory, such as `results/local/`. Keep the config in `configs/`: path resolution assumes that layout.

The runner appends records and resumes existing tasks by default. `--no-resume` appends new attempts rather than replacing prior rows; the formal-data audit rejects duplicate valid task records. Never use it on the committed formal records.

The PowerShell interleaved runner uses the four committed formal configs and reads the API key from the Windows user environment. For a new run, first give its configs separate output paths. For a single command, setting `DEEPSEEK_API_KEY` in the current process is sufficient. No API key belongs in a YAML file or commit.

## Scope of the evidence

The analysis uses one recorded attempt per method/task and pairs results by task. Confidence intervals quantify variation across these tasks; they do not estimate all variation from model sampling, service state or repeated trials. No new full experiment was run during repository preparation.
