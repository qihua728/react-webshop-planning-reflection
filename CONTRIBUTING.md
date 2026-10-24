# Contributing

Keep changes small enough to review against a specific behavior or experiment question.

## Before opening a pull request

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/verify_results.py
```

Tests and saved-result verification do not require a model key or live WebShop server. Describe any live checks separately, including the model, data version, task split and API usage.

## Experiment changes

- Keep saved formal records unchanged. Put new runs in separate output directories.
- Use the same task IDs and shared settings for every method in a comparison.
- Count planning and reflection toward model usage and latency.
- Report failed runs and missing tasks; do not select only favorable attempts.
- Include config/prompt changes and the comparison protocol with a result update.

## Sensitive files

Do not commit credentials, `.env` files, service logs, personal paths or identifying course details. Use `.env.example` only for placeholders. Check commit author email as well as file content before publishing; ignore rules do not remove information already in history.

An open-source license for the original project files has not yet been selected. Preserve existing upstream notices and do not assume the repository grants rights over benchmark data or third-party assets.
