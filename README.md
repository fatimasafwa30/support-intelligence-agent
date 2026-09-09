# Hiver Support Agent

This repository is the foundation for an AI customer-support agent built for the Hiver SDE Intern take-home assignment. Future milestones will work with the Customer Support on Twitter dataset; this milestone creates only the project structure.

## Project layout

- `data/raw/` — source dataset files when they are provided locally; not included in version control.
- `data/processed/` — cleaned or transformed data generated in later work.
- `data/golden/` — curated reference examples for future validation.
- `notebooks/` — exploratory notebooks.
- `src/data/` — future data-loading and preparation code.
- `src/intents/` — future intent-classification code.
- `src/retrieval/` — future knowledge-retrieval code.
- `src/agent/` — future support-agent orchestration code.
- `src/evaluation/` — future evaluation code.
- `tests/` — automated tests.
- `configs/` — project configuration files.
- `reports/` — generated reports and artifacts; not included in version control.
- `main.py` — executable application entry point placeholder.
- `requirements.txt` — dependency manifest; currently intentionally empty of packages.

Run the placeholder with:

```bash
python main.py
```
