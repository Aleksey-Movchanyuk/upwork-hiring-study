# Public data and one-script reproduction

Files
- `upwork_features_public.csv`: 11,507 postings, one row each, sorted by capture time (`time_rank` 1 = oldest). Columns: `time_rank`, `hired` (1 = at least one hire recorded), and the 52 facts used by the models.
- `tracked_alerts.csv` (and `.txt`): the five keyword alerts that selected the postings (55 keywords, verified-payment filter).
- `reproduce.py`: one script (cells marked `# %%`) that rebuilds the time split, the rules, the models and every table and chart listed below.
- `requirements.txt`: `pip install -r requirements.txt`, then `python reproduce.py` (1 to 5 minutes).
- `results/`: what the script writes (main table, calibration, SHAP, error analysis).

What is NOT in the CSV: job titles and descriptions, links, ids, dates, client names. Money columns (client total spent, budgets, client past hires) are rounded to 3 significant figures. The TF-IDF word model needs the text, so it is not reproduced. Numbers can differ from the article in the third decimal (gradient boosting 0.744 here, 0.741 in the article).

Facts (52 columns, 20 groups)
| Group | Columns |
|---|---|
| Job type | job_type_label |
| Budget | budget_fixed, budget_hourly_min, budget_hourly_max, has_budget_fixed |
| Duration, engagement, tier | duration_label, engagement, tier_text |
| Client spend and hires | client_total_spent (empty = no history), client_total_hires |
| Client badge, payment check | client_top_client, client_payment_verification_status |
| Client country | client_country, country_group |
| Other settings | location_mandatory, enterprise_job, premium, tags_count, skills_count |
| Length | description_char_count, description_word_count, title_char_count |
| Copies of the text | desc_prior_copies, title_prior_copies (identical text captured earlier; later copies are not counted) |
| Word-list flags (title + description) | has_crypto_term, has_wallet_term, has_offplatform_contact, agency_tone, new_build, fix_title, long_term_wording, screening_prompt, quick_task_wording |
| Links and emails | desc_has_url, desc_has_email |
| Writing style | sentence_len_mean, sentence_len_cv, line_count, exclamation_count, bullet_count |
| AI-style marks | em_dash_count, em_dash_ratio, curly_quote_ratio, ai_style_signals, markdown_marks, emoji_count, ai_phrase_per_100w, ai_marker_score, non_ascii_ratio |
| Posting time | published_hour, published_dow, age_at_capture_h |

Booleans are 0/1. Empty categories are the string `unknown`. Output (`hired`) means a hire was recorded at any automated check of the posting's activity panel within about 50 days; see the article for what that does and does not mean.
