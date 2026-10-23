# DeliveryIQ — Architecture & Module Contract

DeliveryIQ is a local, single-process Streamlit application over a pre-computed analytical
dataset and a pre-trained scikit-learn model. Nothing is trained or re-read from raw CSV at
app launch.

```
 data/raw/*.csv ──► src/preprocessing.py ──► data/processed/orders_*.parquet ─┐
   (Olist, 7 files)   (+ feature_engineering.py)                              │
                                                                              ▼
 data/processed/orders_delivered.parquet ──► src/train_model.py ──► models/delay_classifier.joblib
                                                                    models/model_metrics.json
                                                                              │
 Streamlit (app.py + pages/*.py) ◄── src/ui.py, src/charts.py ◄── src/analytics.py,
                                     src/statistical_analysis.py, src/insights.py, src/prediction.py
```

Dependency rule: `config` ← `feature_engineering` ← `preprocessing`; `config` ← `analytics`,
`statistical_analysis`, `insights`, `train_model`, `prediction`, `charts`; additionally `train_model` ←
`prediction` (training reuses `derive_model_features`, `coerce_feature_frame`, `HARD_LIMITS` so inference and training
cannot drift); only `ui.py`, `app.py` and `pages/*` import `streamlit`. Everything under `src/` except `ui.py` is pure Python/pandas and
unit-testable without Streamlit.

Always run with the project venv: `.venv/bin/python` (Python 3.12). Shared constants live in
`src/config.py` — read it first.

---

## 1. Analytical grain and definitions

* **One row per order** in `orders_all.parquet` (99,441 rows, `order_id` unique).
* **Analysis set** `orders_delivered.parquet` = rows with `analysis_eligible == True`:
  `order_status == "delivered"` AND purchase, estimated-delivery and actual customer-delivery
  timestamps all present AND `delivered_customer_date >= purchase_timestamp`.
  Cancelled / unavailable / shipped / in-progress orders and delivered-but-no-delivery-date
  orders are **never** labelled on-time; they get `is_late = <NA>` and an `exclusion_reason`.
* **Lateness is calendar-date based:**
  `delivery_delay_days = (delivered_customer_date.normalize() − estimated_delivery_date.normalize()).days`;
  `is_late = 1 if delivery_delay_days > 0 else 0`. Arriving any time on the promised date is on time.
* `delivery_duration_days` = (delivered_customer_date − purchase_timestamp) in fractional days.
* `processing_days` = (delivered_carrier_date − purchase_timestamp) in fractional days; **NaN when the
  carrier date is missing or earlier than purchase** (flagged by `carrier_date_invalid`). Retrospective only.
* `estimated_delivery_days` = (estimated_delivery_date.normalize() − purchase_timestamp.normalize()).days —
  the delivery promise shown at checkout, therefore **known at purchase time** and allowed as a predictor.
* **Multi-item / multi-seller orders:** items are aggregated to the order. The **representative seller** is the
  seller with the largest summed item `price` in that order (ties → lexicographically smallest `seller_id`).
  `seller_state`, `seller_city`, `state_pair`, `same_state` use the representative seller. The representative
  *product category* is the category of the highest-priced item (ties → smallest `order_item_id`).
  `n_sellers` / `n_categories` keep the multi-seller / multi-category information.
* Product categories use the English translation file; missing → `"unknown"` (`category_missing = 1`).
* Reviews (optional file) are reduced to one row per order (latest `review_creation_date`) →
  `review_score`, **descriptive only, never a model input**.
* Customer / seller identifiers are not carried into the processed tables except `order_id`
  (no personal identifiers in the dashboard).

## 2. Processed schema (`orders_all.parquet`; `orders_delivered.parquet` is the eligible subset)

| group | columns |
|---|---|
| key | `order_id` (str, unique) |
| status | `order_status` (category), `analysis_eligible` (bool), `exclusion_reason` (str; "" if eligible) |
| timestamps (datetime64) | `order_purchase_timestamp`, `order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date`, `order_estimated_delivery_date` |
| targets / outcomes | `is_late` (Int8, NA if ineligible), `delivery_delay_days` (Int32, NA if ineligible), `delivery_duration_days` (float), `processing_days` (float), `approval_days` (float), `carrier_to_customer_days` (float) |
| quality flags (int8 0/1) | `approval_missing`, `carrier_date_invalid`, `category_missing`, `weight_missing` |
| time | `purchase_date` (datetime64 normalized), `purchase_year`, `purchase_month`, `purchase_quarter`, `purchase_dow` (0=Mon), `purchase_year_month` (str `YYYY-MM`), `estimated_delivery_days` (Int16; checkout promise, purchase-time feature listed in `config.PURCHASE_TIME_NUMERIC`) |
| geography | `customer_state`, `customer_city`, `seller_state`, `seller_city`, `state_pair` (`"SP→RJ"` seller→customer), `same_state` (int8 0/1) |
| order aggregates | `n_items`, `n_sellers`, `n_products`, `n_categories`, `total_price`, `total_freight`, `freight_ratio` (= total_freight / total_price, NaN-safe), `total_weight_g`, `total_volume_cm3`, `max_product_length_cm`, `max_product_height_cm`, `max_product_width_cm`, `product_category` (representative, English, snake_case as in translation file) |
| descriptive extras | `review_score` (float, NA if no review) |

`orders_delivered.parquet` has unused categorical levels removed (no zero-count `order_status` / state / city / category levels); report status counts list observed values only. Dtypes in this table are asserted by `tests/test_preprocessing.py`.

Also written: `orders_delivered.csv` (same rows as the delivered parquet, for CSV consumers) and
`data_quality_report.json` (row counts per raw file, duplicates found/removed, missing-value counts, invalid
date counts, exclusion-reason counts, multi-seller order count, final eligible count, late rate, date range).

### Implementation notes (Phase 1, as built)

* `exclusion_reason` values: `status_<status>` (any non-delivered status), `missing_purchase_or_estimate`,
  `delivered_missing_delivery_date`, `delivered_before_purchase`, `delivered_no_items` (defensive; 0 rows in the real
  data, it guarantees item-less orders can never become eligible). Eligible rows have `""`.
* Outcomes (`is_late`, `delivery_delay_days`, `delivery_duration_days`, `carrier_to_customer_days`) are computed for
  eligible rows only (NA/NaN otherwise). `processing_days` / `approval_days` are retrospective diagnostics for any row.
* `carrier_date_invalid = 1` only when the carrier timestamp is present and earlier than purchase (`processing_days` then
  NaN; row kept). A negative carrier->customer or approval gap is set to NaN (customer-before-carrier: 23 delivered rows).
* Orders without item rows (775: unavailable/canceled/created/invoiced/shipped) stay in `orders_all` with `n_items=0`,
  `n_sellers=0`, `total_price=total_freight=0`, NaN weight/dimensions/`freight_ratio`/`seller_*`/`state_pair`,
  `same_state=0`, `product_category="unknown"`, `weight_missing=category_missing=1`. They are never eligible.
* Order-level `total_weight_g`, `total_volume_cm3` and `max_product_*_cm` are NaN (and `weight_missing=1`) if *any* item of
  the order lacks the attribute. `category_missing=1` when the *representative* category is missing.
  `n_categories` counts distinct *known* categories (so an all-unknown order has 0).
* Two source categories missing from the translation file (`pc_gamer`, `portateis_cozinha_e_preparadores_de_alimentos`)
  are mapped by a documented supplement in `preprocessing.SUPPLEMENTAL_TRANSLATIONS`.
* `load_raw` reads every column as string (`utf-8-sig`); parsing/coercion happens explicitly and is counted in the report.
* `feature_engineering.PROCESSED_COLUMNS` is the authoritative column list; `validate_processed` flags missing *and*
  unexpected columns (e.g. any customer/seller identifier).

* A product weight or dimension that is zero or negative is physically implausible (4 products with 0 g in the real
  data, 6 eligible orders) and is **treated as missing** (NaN, `weight_missing = 1`). Counts are in
  `data_quality_report.json` under `product_quality`.
* `config.LEAKAGE_COLUMNS` also lists the flags derived from post-purchase timestamps (`carrier_date_invalid`,
  `approval_missing`). `config.NON_FEATURE_COLUMNS` (keys, raw purchase/estimate timestamps, purchase date/year columns)
  and `config.PURCHASE_TIME_UNUSED` (known at purchase but not in the allow-list) complete the partition: every processed
  column is a purchase-time feature, a leakage/outcome column, a non-feature or purchase-time-unused (tested).

Missing numeric product attributes (weight/dimensions) are **kept as NaN in the processed data** with
`weight_missing` flag; model-time imputation (median, fitted on train only) is done inside the sklearn pipeline.

## 3. Module APIs (public functions; keep these names and signatures)

### `src/feature_engineering.py` (pure functions on DataFrames)
`add_time_features(df)`, `add_delivery_targets(df)`, `aggregate_items(items, products, sellers, translation)`
→ order-level frame incl. representative seller / category, `add_geo_features(df)`, plus
`validate_processed(df) -> list[str]` (returns schema/invariant problems; empty list if clean).

### `src/preprocessing.py`
`load_raw(raw_dir=RAW_DIR) -> dict[str, DataFrame]` (raises `MissingDatasetError` with download/placement
instructions when files are absent), `build_orders_table(raw: dict) -> tuple[DataFrame, dict]`
(all-orders frame, quality report), `run_pipeline(raw_dir=RAW_DIR, out_dir=PROCESSED_DIR) -> dict`,
`load_processed(path=ORDERS_DELIVERED_PATH) -> DataFrame` (no streamlit; raises `FileNotFoundError` with the
command to run). CLI: `python -m src.preprocessing`.

### `src/analytics.py` — all functions are safe on empty frames (return zeros/NaN/empty frames, never raise ZeroDivisionError)
Operates on the **delivered/eligible** frame.
* `apply_filters(df, date_range=None, states=None, categories=None, statuses=None, search=None) -> DataFrame`
  (`date_range` = (start, end) on `purchase_date`; `states` = customer_state list; `categories` list).
* `kpi_summary(df) -> dict` keys: `n_orders, on_time_rate, late_rate, n_late, mean_duration, median_duration,
  mean_late_days (avg delivery_delay_days among late), total_freight, n_states` (rates are fractions 0–1).
* `wilson_ci(k, n, alpha=0.05) -> (lo, hi)`.
* `monthly_performance(df)` → cols `purchase_year_month, n_orders, n_late, late_rate, on_time_rate, mean_duration`.
* `duration_summary(df)` → dict (mean, median, std, p90, p95, min, max, skew).
* `state_performance(df, min_orders=0)`, `category_performance(df, min_orders=0)` → cols `<key>, n_orders,
  on_time_rate, late_rate, late_ci_low, late_ci_high, mean_duration, median_duration, mean_late_days,
  mean_freight, mean_weight_g, reliable` (reliable = n_orders >= min_orders).
* `state_pair_performance(df, min_orders=0)` (key `state_pair`, + `seller_state`, `customer_state`).
* `high_delay_segments(perf_df, min_orders, overall_late_rate, ...)` → flags segments whose late rate is above
  overall AND whose Wilson lower bound exceeds the overall rate AND n ≥ min_orders.
* `binned_stat(df, col, bins, ...)` → per-bin n, late_rate, mean_duration (weight bins, freight bins).
* `processing_summary(df)`, `late_by_processing_bucket(df)` (retrospective).
* `delay_trend(df, freq="M")`.

**Track A as built (analytics / statistics / insights):**
* `apply_filters(statuses=...)` filters on the *derived delivery-status label* `"late"` / `"on_time"` (from `is_late`;
  case-insensitive, `"on time"` accepted), **not** on `order_status` (all rows are already delivered). An unknown label
  emits a `UserWarning` and matches no rows.
* Table functions (`state_performance(by=)`, `category_performance`, `state_pair_performance`, `monthly_performance`,
  `delay_trend`, `binned_stat`, `same_state_comparison`) take `min_orders` (default 0, so `reliable` is True for every row
  unless the caller sets a threshold) and, where Wilson intervals are shown, `alpha` (default 0.05). `monthly_performance`
  and `delay_trend` also return `late_ci_low`, `late_ci_high`, `reliable`.
* Added helpers: `weight_bins`, `freight_bins`, `same_state_comparison`, `state_matrix`, `category_cost_comparison`,
  `scatter_sample`, `delivery_status_label`, `wilson_ci_arrays`; `high_delay_segments(..., alpha, direction="high"|"low")`
  adds `excess_vs_overall` and `rate_ratio`.
* `statistical_analysis` adds `two_proportion_ztest`, `holm_adjust`, `cramers_v_label`, `spearman_strength`;
  `run_all_tests` returns a `summary` frame with Holm-adjusted p-values (`p_holm`, `significant_holm`).
* `generate_insights(df, min_orders, alpha)`: segment screens use *unadjusted* Wilson intervals at `1 - alpha` and every
  such insight states how many segments were screened (`n_screened` in `metrics`); the month screen is Bonferroni-adjusted.
  `category_delay.metrics["flagged"]` lists the shown categories whose interval excludes the overall rate; recommendations
  name only those.
* Terminology: `reliable` (analytics tables) only means `n_orders >= min_orders`; "reliably higher/lower" in insight text
  and the `regional_report` `flag` ("Above/Below overall (interval excludes it)", "Interval includes overall") mean the
  unadjusted Wilson interval excludes the overall rate. `summary_report(..., alpha=None)` takes its confidence level from
  the insights' `alpha`; `regional_report(df, min_orders, alpha)`. `spearman_test` reports no CI (NaN, with a `note`) when
  either variable is binary; `run_all_tests` stores `p_holm` / `n_tests` on each result so `plain_language` quotes the
  Holm-adjusted p-value next to the per-test (unadjusted) one (`n_tests` counts only tests that produced a p-value).
  Kruskal-Wallis effect size is epsilon-squared = H / (n - 1); p-values that underflow to 0 are reported as
  `statistical_analysis.P_FLOOR`; the `purchase_month` chi-square carries a `caveat` (months pooled across years).

### `src/statistical_analysis.py`
`chi_square_test(df, col, target="is_late", min_expected=5, top_n=None) -> dict` (`chi2, dof, p_value, n,
cramers_v, effect_label, table, merged_rare_levels`), `spearman_test(df, x, y, alpha) -> dict` (`rho, p_value, n,
ci_low, ci_high` via Fisher-z, `strength_label`), `proportion_diff_ci(k1,n1,k2,n2)`, `kruskal_or_mannwhitney` (optional),
`run_all_tests(df) -> dict` and `plain_language(result) -> str`. Interpretation text always states that significance ≠
practical importance ≠ causality.

### `src/insights.py`
`Insight` dataclass (`kind, title, text, metrics: dict, n: int, severity: "info"|"watch"|"alert"`),
`generate_insights(df, min_orders=DEFAULT_MIN_ORDERS) -> list[Insight]`,
`recommendations(insights) -> list[str]`, `summary_report(df, filters_description, insights, model_metrics=None) -> str`
(Markdown text, downloadable), `regional_report(df, min_orders) -> DataFrame`. All text built from computed numbers.

### `src/train_model.py` (CLI `python -m src.train_model`)
Loads `orders_delivered.parquet`; features ⊆ `PURCHASE_TIME_FEATURES` (asserts no `LEAKAGE_COLUMNS`);
chronological train/validation/test split by `order_purchase_timestamp` (windows documented in metrics JSON);
sklearn `Pipeline(ColumnTransformer(...), model)`; Dummy vs LogisticRegression vs RandomForest (+ optional
HistGradientBoosting); tuning on train/validation only; calibrated probabilities where it helps; threshold/risk
bands chosen on validation; permutation importance on validation/test. Saves:
* `models/delay_classifier.joblib` — a dict bundle: `{"pipeline", "feature_columns", "numeric_features",
  "categorical_features", "target", "model_name", "risk_thresholds": {"moderate": float, "high": float},
  "form_options": {"customer_state": [...], "seller_state": [...], "product_category": [...],
  "state_pair_median_estimated_days": {...}, "defaults": {...}, "ranges": {col: [min, max]}},
  "trained_at", "sklearn_version", "train_window", "metrics_summary"}`
* `models/model_metrics.json` — per-model metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier),
  confusion matrices, ROC/PR/calibration curve points, permutation importances, split windows + row counts +
  class balance, selected model & rationale, risk-band definition, leakage-audit result.

### `src/prediction.py` (no streamlit)
`load_model(path=MODEL_PATH) -> dict` (raises `ModelNotFoundError` with the training command),
`validate_input(inputs: dict, bundle) -> tuple[dict, list[str]]` (cleaned values + human-readable errors; never
silently drops/ignores a missing feature), `predict_risk(inputs, bundle=None) -> dict` (`probability, risk_label,
risk_color, thresholds, model_name, features_used`), `predict_batch(df, bundle=None) -> DataFrame`,
`risk_label(p, thresholds)`. Raises `ValueError` on invalid input.

**Implementation notes (Phase 2, as built — `train_model` / `prediction`)**

* *Bundle keys* (`models/delay_classifier.joblib`) beyond the contract: `base_pipeline` (uncalibrated), `calibrated`
  (bool), `decision_threshold` (F1-optimal cut on validation), `probability_floor` (exact-zero isotonic scores are raised to
  the Laplace-smoothed validation late rate of that level, 0.0 if none), `score_resolution` (distinct score levels / max
  on validation: the calibrated output is a step function).
* *`form_options`* adds `state_pair_min_orders`, `defaults_note`, `hard_limits`, `date_span` (the purchase-date span the model was FITTED on = `train_window`, drives the out-of-span warning), `dataset_date_span` (whole dataset, informational);
  `ranges` are the **training-window** min/max and only drive non-blocking warnings.
* *Input contract:* raw purchase-time primitives (`prediction.ACCEPTED_FIELDS`): three required categoricals, `n_items`,
  `total_price`, `total_freight`, and a time given as `purchase_date` (preferred) or `purchase_month` (+ optional
  `purchase_dow`). Optional numerics left blank are filled from training defaults (state-pair median for
  `estimated_delivery_days`) and reported in `assumptions_used`. `purchase_date` accepts datetime-like objects and ISO
  `YYYY-MM-DD[ HH:MM[:SS]]` strings only (numbers and dd/mm/yyyy are errors).
* *Errors vs warnings:* errors (block scoring) = missing/unknown categoricals, non-numeric / non-integer values, values
  outside `prediction.HARD_LIMITS` (impossible values, not data-derived), unreadable or absurd dates, unknown field names.
  Warnings (never block) = value outside the training range or the dataset date span: the model extrapolates (e.g.
  3-7-day promises occur in the test window but not in training).
* *Public pieces:* `derive_model_features(raw)` (single source of truth for `freight_ratio`, `same_state`,
  month/dow/quarter, parity-tested against the processed table), `coerce_feature_frame` (first pipeline step),
  `check_input(inputs) -> (cleaned, errors, warnings)` (`validate_input` = its first two items), `risk_labels` (vectorised).
* *`predict_risk`* returns, besides the contract keys: `assumptions_used`, `derived_features`, `decision_threshold`,
  `calibrated`, `warnings`, `probability_note` (step-function caveat + the test-window finding), `test_window_caveat`
  (structured test-window figures: mean predicted vs observed, Brier / log-loss vs the constant baseline, test late rate per risk
  band, `test_performance_degraded`; also stored in the bundle). *`predict_batch`* adds columns `late_probability`, `risk_label`,
  `assumptions_used`, `input_warning`, `input_error`.
* *Honest reporting in `model_metrics.json`:* validation metrics are labelled tuned / in-sample (`validation_note`); the
  selected model carries `test_window_assessment` (lift with bootstrap CI, Brier / log-loss vs the constant baseline, mean
  predicted vs observed) and `test_window_warnings`, which feed `rationale` and `metric_guidance`; `risk_bands` carry Wilson
  intervals, actual shares and `*_diagnostics` (monotone? overlapping adjacent intervals?); `split.covariate_support` gives the
  per-window share of orders outside the training feature range; `permutation_importance` reports validation (`items`) and
  test (`test_items`, diagnostic only) rankings. Top-10 % precision / recall use expected values under random tie-breaking
  (a constant-score baseline equals the prevalence).

### `src/charts.py` (Plotly figure builders; each accepts DataFrames from analytics, returns `go.Figure`;
must render a clean "no data for current filters" figure on empty input)
Includes: status donut, monthly performance (dual metric), duration histogram (+mean/median lines), state late-rate
bar, category late-rate / duration bars, weight-vs-duration, freight-vs-duration, processing histogram, seller→customer
heatmap, confusion matrix, model comparison, feature importance, ROC/PR/calibration, Brazil choropleth (if geojson present).
Applies one shared Plotly template built from `config.PALETTE`.

### `src/ui.py` (the only src module importing streamlit)
CSS injection, `kpi_card`/`kpi_row`, `section_header`, `insight_box`, `empty_state`, sidebar brand + dataset info,
`@st.cache_data` loaders (`get_orders()`, `get_quality_report()`), `@st.cache_resource` model loader,
`render_global_filters(df) -> DataFrame` (date range, customer state, product category; stored in `st.session_state`;
shows n orders remaining), `friendly_error(...)`.

### App entry
`app.py` builds the shell with `st.set_page_config`, injects CSS, defines `st.navigation([...])` over the five
`pages/*.py` files with titles/icons (Overview is default), renders sidebar brand, global filters and dataset info,
then `pg.run()`. Pages: `1_Overview.py`, `2_Delay_Analysis.py`, `3_Regional_Intelligence.py`,
`4_Delay_Prediction.py`, `5_Reports_Insights.py`. Pages call a shared helper in `ui.py` to obtain the
filtered frame so filters apply consistently.

*Known Streamlit 1.66 quirks of deep links (not bugs in the page code).* (1) Because a `pages/` directory exists, Streamlit
runs the very first session after server start through its legacy multipage path, which registers the pages under their
file names (`Delay_Analysis`) before `app.py` calls `st.navigation`; a deep link such as `/delay-analysis` as the first
connection therefore shows a dismissible "Page not found" dialog over the correctly rendered page. Later sessions are
unaffected (`st.navigation` switches the legacy path off). Avoiding it would need the directory to be renamed. (2) The
default page is always served at `/`; its `url_path="overview"` is ignored, so `/overview` is "Page not found" by design.
The 404s for `/<page>/_stcore/health` in the browser console are the frontend probing base paths and are harmless.

## 4. Quality rules (apply to every module)

* No placeholder or random numbers anywhere; every displayed statistic is computed from the data/model.
* Explicit handling for empty / tiny samples; always show `n` beside rates; flag samples below `min_orders`.
* Never claim causality; label retrospective (non-purchase-time) analyses as such.
* Type hints, short docstrings, vectorised pandas; comments only where logic is non-obvious.
* Tests live in `tests/` and run with `.venv/bin/python -m pytest`.
