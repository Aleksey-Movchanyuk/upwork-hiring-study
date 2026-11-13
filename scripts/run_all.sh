#!/usr/bin/env bash
# Rebuild every figure and table for the article from the database.
# Usage: bash article/scripts/run_all.sh   (from the project root)
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
$PY 01_extract.py      # DB -> de-identified features.parquet
$PY 02_eda.py          # exploratory charts
$PY 03_inference.py    # tests, regression, bootstrap
$PY 04_ml.py           # time-split models, calibration, cost analysis
$PY 08_term_mining.py  # data-driven word/phrase scan (FDR-corrected, adjusted)
$PY 09_feature_groups.py  # which kind of information predicts a hire (client / job / text / timing)
$PY 10_importance.py   # ranking of facts by how much they matter
$PY 11_rules_vs_models.py  # simple rules vs logistic, boosting, neural network
$PY 12_ngrams.py       # do word sequences (1-3-grams) help?
$PY 13_unigram_chart.py  # chart of the strongest single words (training vs test)
$PY 14_freelancer_view.py  # market money by budget, 7-point checklist, niche shares
$PY 16_publication_checks.py  # honest checklist, paired tests, competition, odds ratios (needs 11 and 12)
$PY 17_length_feature.py      # description length: alone, with/without in model and checklist, robustness (needs 11 and 16)
$PY 18_feature_matrix.py      # matrix of small charts per feature group (needs 10)
$PY 19_export_public_dataset.py  # anonymised CSV for GitHub (public/upwork_features_public.csv)
$PY 15_method_comparison.py  # one dashboard comparing all methods (needs 11, 12 and 16)
echo "Optional: python 07_ai_classifier.py applies the AI-marker dictionary (ai_markers.py) and checks it against your labels."
echo "Optional manual step: python 05_label_sample.py, label the CSV, then python 06_validate_ai_proxy.py"
