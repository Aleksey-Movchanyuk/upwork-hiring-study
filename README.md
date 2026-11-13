# Which Upwork jobs will hire? Simple rules vs machine learning

Code, an anonymised feature table and the technical appendix behind a study of 11,507 Upwork job postings: can a short checklist predict whether a posting ends in a recorded hire as well as a machine-learning model can?

## Main findings (test period: the latest 15% of postings, 1,727 of them)

- A seven-question yes/no checklist reaches ROC AUC 0.722; gradient boosting on the same 52 facts reaches 0.741 (paired difference +0.019, 95% range 0.001 to 0.037).
- 64 further models and combinations (linear, trees, boosting, neural networks, blends, stacking) were tried. None of the ten picked in advance on a separate tuning slice beat the checklist clearly.
- What matters most: kind of job (length, budget, fixed price or hourly), client history, and description length. AI-style writing adds nothing once length is accounted for.

## Reproduce it in a few minutes

```bash
cd public
pip install -r requirements.txt
python reproduce.py            # main table, paired tests, calibration, SHAP, error analysis
python benchmark_models.py     # the 64-model benchmark (10 to 20 minutes)
```

`public/upwork_features_public.csv` has one row per posting: a time rank, the outcome (`hired`) and the 52 facts. It contains no job text, titles, links, dates or client names, and money columns are rounded to three significant figures, so numbers can differ from the published ones in the third decimal. See `public/README.md` for the data dictionary and `public/tracked_alerts.csv` for the keyword alerts that selected the postings.

## What is in this repository

| Path | What |
|---|---|
| `public/` | the anonymised CSV, `reproduce.py`, `benchmark_models.py`, results |
| `scripts/` | the full pipeline (extraction, statistics, models, charts). It needs a private database of captured postings, so it is here for transparency, not to run |
| `figures/` | the charts |
| `TECHNICAL_APPENDIX.md`, `TECHNICAL_APPENDIX_RU.md` | formulas, settings, tests and checks (English and Russian) |

## How the data was collected, and its limits

The postings come from five keyword alerts of one freelancer (automation and AI, data engineering, web scraping, BI, crypto), limited to clients with a verified payment method, over about three months in 2026. "Hired" means that at least one hire showed in the posting's public activity panel at any of about three automated checks (median 22 days of observation). A "no" can mean an open, abandoned or off-platform posting. Details and caveats are in the technical appendix (section G1) and `public/README.md`.

Job text is not shared: it belongs to other people.

## Licence

No licence has been chosen yet, so the usual copyright rules apply until one is added.
