"""Narrative (Markdown) cells for notebooks/01_data_exploration.ipynb.

Every number quoted here was read from the executed notebook outputs; the code cells print the same values.
"""

MD: dict[str, str] = {}

MD["title"] = """
# DeliveryIQ — Notebook 01: Data Exploration

**Project:** DeliveryIQ — Intelligent Shipment Analytics & Delay Prediction (Fundamentals of Data Science)
**Notebook:** `notebooks/01_data_exploration.ipynb`  |  **Data:** Olist Brazilian E-Commerce Public Dataset

## Objectives
1. Understand the raw Olist tables: schemas, keys, how they join, and the cardinality of orders, items and sellers.
2. Audit data quality (missing values, duplicates, timestamp problems, logical anomalies, order status) and justify every cleaning decision.
3. Show, on real example orders, how raw columns become the engineered columns used by the dashboard and the model
   (`delivery_duration_days`, `delivery_delay_days`, `is_late`, `processing_days`, `estimated_delivery_days`, ...).
4. Explore delivery performance (duration, delay, lateness over time, by state, category, weight, freight, seller processing time).
5. Separate **purchase-time** features from **after-the-fact** columns, and explain why mixing them is data leakage.

## Dataset provenance
*Brazilian E-Commerce Public Dataset by Olist* (Kaggle: <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>).
It contains roughly 100,000 orders placed on the Olist marketplace between 2016 and 2018, with items, sellers, customers,
products, reviews and a Portuguese-to-English category translation. The raw CSVs live in `data/raw/`; the cleaned analysis tables
are produced by `src/preprocessing.py` into `data/processed/`. Customer and seller identifiers in the source are anonymised by Olist.
The dataset has **no** courier, GPS, weather, or carrier-route information; nothing like that is used or invented here.

## How to run
From the project root (or from `notebooks/`), with the project virtual environment:

```bash
.venv/bin/python -m src.preprocessing        # (re)builds data/processed/*.parquet and the quality report
.venv/bin/jupyter lab notebooks/01_data_exploration.ipynb
```
The notebook finds the project root automatically. Figures are also saved as PNG files in `reports/figures/`.
It runs top-to-bottom in well under two minutes.

## Contents
1. Dataset understanding &nbsp;·&nbsp; 2. Data quality & cleaning decisions &nbsp;·&nbsp; 3. Feature-engineering walk-through &nbsp;·&nbsp;
4. Exploratory analysis &nbsp;·&nbsp; 5. Predictive vs descriptive feature sets (leakage) &nbsp;·&nbsp; 6. Key takeaways, limitations, next steps

**Conventions used throughout.** Rates always come with their sample size `n`. "Associated with" never means "caused by".
Analyses that use information only available after the purchase are labelled **RETROSPECTIVE**.
A segment is called *reliable* only if it has at least `config.DEFAULT_MIN_ORDERS` (= 100) orders.
"""

MD["s2_intro"] = """
## 1. Dataset understanding

This notebook is *exploration*, so the raw files are read here directly with `pandas` (default type inference) to show their
true schema. The production pipeline (`src.preprocessing.load_raw`) reads everything as text and parses explicitly; we call that
real pipeline later in section 2 and reconcile it with what we find here.
"""

MD["er"] = """
### Entity-relationship structure

```
 customers (customer_id PK, customer_unique_id, zip, city, state)
        │ 1
        │            one customer_id appears in exactly one order
        │ N (1 in practice)
 orders (order_id PK, customer_id FK, order_status, 5 timestamps)
        │ 1                                            │ 1
        │ N                                            │ N
 order_items (order_id FK + order_item_id PK,          order_reviews (review_id, order_id FK, score, dates, comments)
              product_id FK, seller_id FK, price, freight_value, shipping_limit_date)
        │ N                      │ N
        │ 1                      │ 1
 products (product_id PK, category [PT], weight, L/H/W)     sellers (seller_id PK, zip, city, state)
        │ N
        │ 1
 category_translation (product_category_name [PT] -> product_category_name_english)
```

Join keys: `orders.customer_id → customers.customer_id`, `order_items.order_id → orders.order_id`, `order_items.product_id → products.product_id`,
`order_items.seller_id → sellers.seller_id`, `order_reviews.order_id → orders.order_id`, `products.product_category_name → category_translation.product_category_name`.
The cells below verify these relationships on the data instead of assuming them.
"""

MD["s2_card"] = """
### Key cardinalities: orders → items → sellers

An **order** is the analytical grain (one delivery promise per order), but the raw `order_items` table has one row *per item*,
so items must be aggregated to the order. A single order can also contain items from several sellers.
"""

MD["s2_interp"] = """
**What we learn**

* All foreign keys are complete: there are **0 orphan rows** in every join, so inner/left joins never silently drop data. The only gap is
  13 product rows whose category is missing from the translation file (2 distinct categories, handled in section 2).
* `orders` is 1:N with `order_items` (112,650 item rows for 99,441 orders). **89.4%** of orders (88,863) have exactly one item, 7.6% have two;
  9,803 orders have two or more items. The maximum is 21 items in one order; mean = 1.142 items per order among orders that have items.
* **775 orders have no item rows at all** (0.8%). They are almost all `unavailable` or `canceled` (see the status table below), and can never carry a delivery outcome.
* **1,278 orders involve 2 or more distinct sellers** (the maximum is 5). Such an order has one customer-facing delivery promise
  but several shipments, so a single "the seller's state" is ill-defined. The project handles it by choosing a deterministic
  *representative seller* (section 3) and keeping `n_sellers` as a feature.
* `customer_id` is unique per order (max one order per `customer_id`); the 99,441 `customer_id` values map to 96,096 `customer_unique_id`
  values, so some people ordered more than once. `review_id` is **not** unique (98,410 distinct in 99,224 rows) and 547 orders have more than one review,
  which is why the pipeline keeps the latest review per order.
"""

MD["s3_intro"] = """
## 2. Data quality & cleaning decisions

For every issue below we (a) measure it, (b) state the decision, and (c) point at the code in `src/preprocessing.py` /
`src/feature_engineering.py` that implements it.

### 2.1 Missing values
"""

MD["s3_anom_interp"] = """
**Reading the checks**

* **Missing values:** the two review *comment* columns are mostly empty (88.34% and 58.70%) — people rarely write text, expected and irrelevant to delivery.
  What matters for this project: `order_delivered_customer_date` is missing for 2,965 orders (2.98%), `order_delivered_carrier_date` for 1,783 (1.79%), and
  `order_approved_at` for 160 (0.16%). 610 products (1.85%) lack category/name/description/photo count; 2 products lack weight and dimensions.
* **Duplicates:** there are **no exact duplicate rows** and **no duplicate primary keys** in any table. The only repeated keys are in reviews
  (a review ID shared by several orders; 551 repeated `order_id` rows), handled by keeping the latest review per order. The pipeline still checks and de-duplicates defensively.
* **Timestamps:** every non-missing timestamp parses with the format `%Y-%m-%d %H:%M:%S` (0 unparseable). The estimated delivery date always has time 00:00:00 — it is a *calendar date*, not a moment. This matters for the lateness definition (section 3).
* **Logical anomalies** (delivered orders in brackets): no order is approved before purchase, delivered before purchase, or has an estimate before purchase.
  But **166 carrier-pickup timestamps are earlier than the purchase** (165 delivered); the median is only 0.6 h earlier (90th percentile 1.5 h), so mostly clock/entry skew,
  yet a few are days or months early (max about 4,109 h). **1,359** carrier timestamps precede the approval timestamp (plausible when sellers ship before payment is approved), and **23** customer-delivery
  timestamps precede the carrier pickup.
* **Decisions:** the 166 carrier-before-purchase rows are *kept* (the order itself is fine) but flagged `carrier_date_invalid = 1` and their `processing_days` is set to NaN rather than "fixed"
  (`add_delivery_targets`). The 23 impossible carrier→customer gaps are set to NaN for `carrier_to_customer_days`. We do not use the carrier-before-approval pattern because it is not an error.
  None of these diagnostics is known at purchase time anyway, so they never feed a predictor.

### 2.2 Order status
"""

MD["s3_exclusion"] = """
**Why cancelled / undelivered orders are excluded**

The analysis set must contain orders for which we *observe* whether the promise was kept.

* **96,478 orders (97.02%) are `delivered`**; of those, **8 have no customer-delivery timestamp** and cannot be classified, leaving **96,470 eligible orders**.
* The other **2,963 orders** (`shipped` 1,107; `canceled` 625; `unavailable` 609; `invoiced` 314; `processing` 301; `created` 5; `approved` 2) have no delivery outcome:
  none of the shipped / unavailable / invoiced / processing orders has a customer-delivery date, and only 6 of the 625 cancelled orders has one.
* Treating these as "on time" would inflate on-time performance with orders that never arrived; treating them as "late" would be equally unfounded for orders still in transit. They get `is_late = <NA>`
  and an `exclusion_reason` (`status_<status>`), and are never counted as on time. 603 `unavailable` orders also have no item rows.
* **Consequence (selection bias):** all statistics in this project describe *delivered* orders only. Any problem that *prevents* delivery (cancellations, stock-outs) is invisible in the late rate.

### 2.3 Product-category missingness and zero weights
"""

MD["s3_cleaning"] = """
**Findings.** 610 products (1.85%) have no category, which affects 1,603 item rows (1.42%). In the analysis set, 1,358 orders end up with an *unknown* representative category (1.41%).
Two categories that do occur in `products` (`pc_gamer`, `portateis_cozinha_e_preparadores_de_alimentos`) are absent from the official translation file (13 product rows). Four products have a weight of exactly **0 g**
(all `cama_mesa_banho`, i.e. bed/bath/table, which cannot weigh 0 g) and two have missing weight and dimensions.

### 2.4 Cleaning decisions and where they are implemented

| Issue (measured above) | Decision | Implemented in |
|---|---|---|
| Cancelled / unavailable / shipped / in-progress orders; delivered with no delivery date | Kept in `orders_all` but `analysis_eligible = False`, `is_late = <NA>`, `exclusion_reason` set; never labelled on time | `feature_engineering.add_delivery_targets` |
| Estimated date has no time of day | Lateness is judged on **calendar dates** | `add_delivery_targets` (`delivery_delay_days`, `is_late`) |
| Carrier timestamp before purchase (166) | Row kept, `carrier_date_invalid = 1`, `processing_days = NaN` | `add_delivery_targets` |
| Customer delivery before carrier pickup (23) / negative approval gap | Gap set to NaN, never silently swapped | `add_delivery_targets` |
| Items with several rows / several sellers | Aggregated to one row per order with a deterministic representative seller & category | `feature_engineering.aggregate_items` |
| Missing category (610 products) | Category becomes `"unknown"`, `category_missing = 1` | `aggregate_items` |
| Category missing from translation (2) | Documented supplement mapping | `preprocessing.SUPPLEMENTAL_TRANSLATIONS` |
| Weight / dimension of 0 or missing (6 products) | Treated as missing (NaN) + `weight_missing = 1`; imputation (median, fit on training data only) is left to the model pipeline | `aggregate_items` |
| Multiple reviews for one order | Keep the latest per order; `review_score` is descriptive only | `preprocessing._reduce_reviews` |
| Duplicates | Exact duplicates dropped, repeated keys keep first (none found in orders / items) | `preprocessing._inspect_and_dedupe` |
| Customer / seller identifiers | Not carried into processed tables (privacy) | `feature_engineering.PROCESSED_COLUMNS`, `validate_processed` |

### 2.5 Reconciling this notebook with the real pipeline

Below we recompute the headline counts **independently** (plain pandas, own timestamp parsing, own lateness rule) and compare with
(1) the quality report written by the pipeline, and (2) a fresh call of `src.preprocessing.build_orders_table`. A disagreement would stop the notebook with an assertion error.
"""

MD["s3_end"] = """
**Result:** all eight headline metrics agree across the three sources — 99,441 raw orders, 96,478 delivered, **96,470 eligible**, **6,534 late orders**, a calendar-date late rate of **6.77%**
(timestamp-based reference 8.11%, discussed next), 1,278 multi-seller orders and 166 carrier-before-purchase rows. The parquet file on disk also matches a fresh pipeline run, and the pipeline's own
`validate_processed` reports no problems. From here on we use the processed analysis set (`data/processed/orders_delivered.parquet`).
"""

MD["s4_intro"] = """
## 3. Feature-engineering walk-through on real orders

The analysis set has one row per eligible order. Three real orders are shown first; each engineered value is then **re-derived by hand** from raw timestamps and asserted equal to the pipeline output.

* `delivery_duration_days` = delivery timestamp − purchase timestamp, in fractional days.
* `estimated_delivery_days` = estimated date − purchase date (both at midnight, whole days) — the promise visible at checkout, so *known at purchase time*.
* `delivery_delay_days` = delivery **calendar date** − estimated **calendar date** (negative = early).
* `is_late` = 1 if `delivery_delay_days > 0`, else 0.
* `processing_days` = carrier-pickup timestamp − purchase timestamp (**RETROSPECTIVE**; NaN if the pickup is missing or before purchase).

The examples: the eligible order whose duration is closest to the on-time median, the late order whose delay is closest to the late-order median, and a *boundary case* delivered on the estimated date itself.
"""

MD["s4_boundary"] = """
**Reading the examples.** The typical on-time order was delivered 8 days before the estimated date (delay = −8); the typical late order arrived 7 days after the promise
(delay = +7, a 33.4-day journey against a 26-day promise).

**The boundary case (`0005a1a1...`)** is delivered on 2018-03-29 at 18:17:31, and the estimated date is 2018-03-29. The estimate is stored as `00:00:00`, but a customer promised delivery "by 29 March" is not late
at 18:17 on 29 March. A strict timestamp comparison would call it late (the last extra column in the table above shows 1), the calendar rule correctly gives `delay = 0`, `is_late = 0`.

### Calendar-date vs timestamp lateness
"""

MD["s4_multi"] = """
**Why the two definitions differ.** The delivery promise carries only a *date*. Comparing a full delivery timestamp with midnight at the start of the promised day counts every delivery that happens on the promised day (after 00:00) as late. This affects
**1,292 orders (1.34% of all analysis orders)**, exactly the orders with `delivery_delay_days == 0`. The timestamp rule therefore reports 7,826 late orders (**8.11%**) against **6,534 (6.77%)** with the calendar rule used throughout this project, and the
difference of 1,292 orders is entirely made up of on-the-promised-day deliveries. The calendar rule matches what the promise means to the customer.

### Multi-seller example: how an order becomes one row

Raw `order_items` rows must be turned into a single order row. For orders with several sellers the project picks a **representative seller** (largest summed item price; ties → smallest `seller_id`) and a
**representative category** (category of the highest-priced item; ties → smallest `order_item_id`). `n_sellers`, `n_categories` and `n_items` retain the multi-seller information. The first example
(sorted by `order_id`) with exactly two sellers in different states and two different categories:
"""

MD["s5_intro"] = """
**Reading the multi-seller order (`002f98c0...`).** It has two items from two sellers (MG and SP). The SP seller's item is the more expensive (BRL 44.90 vs 8.99), so the representative seller state is SP
and the representative category is `toys`; the customer is in RS, so `same_state = 0` and `state_pair = SP→RS`. Total price and freight are the item sums (BRL 53.89 and 39.73, `freight_ratio` = 0.737: freight is 74% of the goods value),
weight is 650 g, volume 6,482 cm³, the checkout promise was 32 days, and the actual duration 3.4 days (`is_late = 0`). All values were re-derived from the raw item rows in the cell above. Only 1.32% of analysis orders are multi-seller, so
this simplification affects few orders.

**Aggregates over the whole analysis set (table above):** median order value BRL 86.50 (mean 137.04, maximum 13,440), median freight BRL 17.17, median weight 750 g, median promise 24 days (mean 24.4), 35.97% of orders are
same-state, 9.99% are multi-item. Price, freight and weight are strongly right-skewed (skewness well above 1, see the summary table in section 4.9), which matters for modelling.

## 4. Exploratory analysis

All charts below are also saved as PNGs in `reports/figures/`. Unless stated otherwise the data is the analysis set (n = 96,470 delivered orders). Rates are shown with 95% Wilson confidence intervals where useful.

### 4.1 Target distribution and class imbalance
"""

MD["s5_target_interp"] = """
**Interpretation.** 6,534 orders (**6.77%**) were late and 89,936 (93.23%) on time or early: about **13.8 on-time orders per late order**. The classes are strongly imbalanced, so a trivial model that always answers
"on time" would be 93.23% accurate while detecting no late order. For the prediction work later, accuracy is the wrong headline metric; precision, recall, PR-AUC and calibrated probabilities are more informative.

### 4.2 Delivery duration and delay relative to the promise
"""

MD["s5_dist_interp"] = """
**Interpretation.**

* **Duration** is right-skewed: mean **12.6 days** vs median **10.2 days**, 90th percentile 23.1, 95th 29.3, 99th 46.1, skewness 3.83, maximum 209.6 days. The mean is pulled up by a long tail, so the median is the better "typical" value.
* **Delay vs promise:** the median order arrives **12 days before** the estimated date (mean −11.9 days). **91.9%** of orders are early, 1.3% arrive on the promised date and 6.8% late. The average checkout promise (24.4 days) is about twice the average
  actual duration (12.6 days). In other words, the promise is generous on average and "late" is a *tail event relative to a padded promise*, which is why lateness is much harder to predict than raw duration.
* **Late orders:** among the 6,534 late orders the median lateness is 7 days, mean 10.6, 90th percentile 22 days, maximum 188 days.

### 4.3 Monthly volume and late rate
"""

MD["s5_monthly_interp"] = """
**Interpretation.**

* **Sparse early data:** only 267 analysis orders were purchased in 2016 (2016-09: n = 1; 2016-10: n = 265; 2016-12: n = 1; there are no eligible orders in 2016-11). The months with n < 100 are drawn as hollow markers or are off the axis
  (e.g. the 2016-09 single order is "100% late", which is meaningless). The first month with n >= 100 is 2016-10 (n = 265; 0.75%, a very wide interval); month-to-month comparisons are meaningful from 2017-01 (n = 750).
* **Growth:** volume grows from 750 orders in January 2017 to 4,478 in October 2017, then jumps to **7,288 in November 2017** and stays at about 5,500–7,100 orders per month through 2018-08.
* **Late-rate regimes:** from May to October 2017 the monthly late rate sits around 2.8–4.4%. It then spikes to **12.40% in November 2017 (n = 7,288)**, falls to 5.70–7.46% in Dec–Jan, and spikes again to **14.13% in February 2018 (n = 6,555)** and
  **18.96% in March 2018 (n = 7,003)**, the highest month, before dropping to 4.50% in April 2018. Among months with n >= 100 the lowest rate is 2016-10 (0.75%, n = 265, very wide interval); since 2017-01 the lowest is June 2018 (1.16%, n = 6,096).
* **Zoom:** the Nov-2017 spike is concentrated in a demand peak: the busiest purchase day, **2017-11-24, had 1,147 orders** (median of the other days in that window: 190) and a 17.3% late rate. Purchases up to Nov 19 had a late rate of 6.0% (n = 3,033) vs 17.2% for Nov 21–30 (n = 4,036).
  (That date coincides with Black Friday 2017 — calendar knowledge, not something this dataset records.) The Feb–Mar 2018 episode is **not** a volume spike: weekly volume is roughly flat (about 1,350–1,850 orders per week from January to April), but the weekly late rate climbs from 8.6% (week of Feb 5) to
  **26.3% (week of Feb 26, n = 1,849)** and returns to 11.8% by the week of Mar 19. In that worst week the mean checkout promise was 22.9 days vs 24.4 overall.
* **What we can and cannot say:** high lateness *coincides with* a demand peak (Nov 2017) and with a period of unchanged volume (Feb–Mar 2018). The data contain no carrier, strike, weather or capacity variables, so **we cannot attribute these episodes to a cause**.
* **Selection check:** the share of purchased orders that end up in the analysis set is 97.1–98.9% for each month from February to August 2018, so the low late rates of mid-2018 are not obviously produced by many undelivered orders being dropped (the 20 orders purchased in 2018-09/10 are not eligible).
* **Modelling implication:** the late rate is strongly non-stationary over time. A random train/test split would mix periods; the model should be validated chronologically, and its performance on a future shock period should be expected to be lower.

### 4.4 Late rate by customer state
"""

MD["s5_state_interp"] = """
**Interpretation.**

* **Large regional differences.** Among the 24 states with n >= 100 the late rate ranges from **2.76% (AM, n = 145)** to **21.41% (AL, n = 397; 95% CI 17.7–25.7%)**, a 7.8-fold spread. The highest rates are in the North-East: AL 21.4%, MA 17.4% (n = 717), SE 15.2% (n = 335),
  PI 13.9% (n = 476), CE 13.8% (n = 1,279). RJ, the second-largest state (n = 12,350), has a late rate of 12.1%.
* **The big southern/south-eastern states are low:** SP 4.49% (n = 40,494; 42% of all orders), MG 4.57% (n = 11,354), PR 4.04% (n = 4,923), RS 6.08% (n = 5,344).
* **Duration is not lateness.** AL's mean duration is 24.5 days vs 8.8 days in SP, but AM (26.4 days) and RO (19.4 days) have *low* late rates, 2.76% and 2.88%. The mean checkout promise in those states is long (45.9 days AM, 39.4 days RO, vs 33.2 days AL and 19.8 days SP), i.e. the promise appears to
  absorb the longer shipping time there. These two states have small samples (n = 145 and n = 243) and very wide intervals (AM: 1.1–6.9%).
* **Small samples:** RR (n = 41), AC (n = 80) and AP (n = 67) are below the 100-order threshold; their estimates are shown in light blue and must not be ranked.
* **Statistical test:** a chi-square test of `customer_state` vs `is_late` gives chi2 = 1,785.8 (dof 26, p < 1e-300) with Cramér's V = 0.136 — the association is statistically unambiguous with a small-to-moderate effect size. Statistical significance with n = 96k is almost guaranteed;
  the effect size is the useful number. State is an association with, not a cause of, lateness: it stands in for distance, carrier networks and infrastructure, none of which are in the data.

### 4.5 Late rate by product category
"""

MD["s5_cat_interp"] = """
**Interpretation.** 74 categories occur in the analysis set; 52 have at least 100 orders and are ranked (22 are too small).

* The highest late rates are in **audio (11.92%, n = 344)**, home_confort (9.46%, n = 370), fashion_underwear_beach (9.40%, n = 117), books_technical (8.27%, n = 254) and baby (8.11%, n = 2,761). The lowest are construction_tools_safety (2.60%, n = 154),
  food_drink (3.65%, n = 219) and agro_industry_and_commerce (3.96%, n = 177).
* **Most intervals overlap the overall 6.77% line, but not all:** for 40 of 52 categories the 95% interval overlaps the overall rate; 12 differ detectably (interval entirely above or below it), e.g. audio, home_confort, baby (95% CI 7.2–9.2%), health_beauty, bed_bath_table and housewares (about 2–3 exclusions would be expected by chance alone). The chi-square test across categories agrees (chi2 = 142.4, dof 51, p ≈ 1.4e-10), but all differences are small: Cramér's V is only **0.039** —
  category explains far less than customer state (V = 0.136).
* The ten largest categories (bed_bath_table n = 9,184, 7.48%; health_beauty n = 8,613, 7.54%; sports_leisure, computers_accessories, furniture_decor, housewares) (plus watches_gifts, telephony, auto, toys) all sit between 5.42% (housewares) and 7.54% (health_beauty). The 1,358 orders with an *unknown* category are close to the average (7.14%).
* Caveat: we look at 52 categories at once, so a few apparent extremes are expected by chance; the category shown is the *representative* (highest-priced item's) category.

### 4.6 Weight, freight and delivery duration
"""

MD["s5_wf_interp"] = """
**Interpretation (associations only).**

* **Weight:** mean duration rises gently from 11.7 days (lightest decile, up to 175 g) to 14.3 days (heaviest decile, above 6,575 g); the late rate wanders between 5.82% and 7.96% without a clean trend, with about 8,600–10,500 orders in each bin.
  Spearman rho is **0.087** with duration and **0.016** with `is_late`: weak.
* **Freight:** a much stronger relationship with duration: mean duration is 5.9 days in the cheapest freight decile (up to BRL 9.34) and 15.9 days in the most expensive (above BRL 39.39), Spearman rho = **0.382**. The late rate rises from 4.24% to 8.49%, but the rank correlation with `is_late`
  is only **0.052**.
* Freight value in this data reflects, among other things, weight *and shipping distance*, so it very likely acts as a proxy for geography (consistent with the state results). We cannot separate these effects here, and the pattern says nothing about what would happen if freight were changed.
* Note that `estimated_delivery_days` has rho = 0.523 with duration but −0.023 with `is_late` (section 4.9): the promise already scales with expected duration, which is why longer deliveries are not much *later* deliveries.

### 4.7 Seller processing time (RETROSPECTIVE)
"""

MD["s5_proc_interp"] = """
**Interpretation (RETROSPECTIVE: carrier pickup is only known after the order is placed).**

* `processing_days` (purchase → carrier pickup) is available for 96,304 of 96,470 orders (166 are NaN: carrier date missing or before purchase). Mean 3.2 days, median 2.2 days, with a long tail.
* The late rate rises steeply with processing time: **3.27% (< 1 day, n = 18,980)** → 6.04% (2–3 days, n = 16,299) → 12.54% (7–10 days, n = 4,193) → **28.31% (>= 10 days, n = 3,769)**. Spearman rho with `is_late` = 0.131.
* This is partly mechanical: processing time is a *component* of the total duration, so slow processing contributes directly to a late arrival. On average **70.5%** of the total duration happens *after* carrier pickup, so transit still dominates.
* Because this information does not exist when the customer clicks "buy", it is an operations/seller-performance diagnostic and **cannot** be a predictor of purchase-time risk (section 5).

### 4.8 Same-state vs cross-state shipments
"""

MD["s5_ss_interp"] = """
**Interpretation.**

* **35.97%** of analysis orders are same-state (representative seller and customer in the same state). Their late rate is **4.50% (n = 34,696; 95% CI 4.29–4.73%)** vs **8.05% (n = 61,774; 7.84–8.26%)** for cross-state orders.
* Mean duration is 7.9 vs 15.1 days (median 6.5 vs 12.8), and the checkout promise is 18.2 vs 27.8 days: the promise grows with distance, but not enough to offset the higher late rate.
* chi-square: chi2 = 440.9, p = 6.8e-98, Cramér's V = **0.068** (small effect). The gap is real in this sample, but "cross-state" is another proxy for distance/logistics, not a cause that we can isolate.
* **State pairs:** SP→SP (n = 30,727) has a late rate of 4.65%; the next biggest flow SP→RJ (n = 8,155) has **14.13%**, SP→MG (n = 7,437) 5.28%, SP→RS (n = 3,594) 6.62%. Among pairs with n >= 100, the highest rates are SP→AL 23.92% (n = 255),
  MA→SP 20.33% (n = 123), SP→MA 18.90% (n = 492). Most sellers are in SP, so "same-state" is dominated by SP→SP.

### 4.9 Summary statistics and rank correlations with purchase-time features
"""

MD["s5_end"] = """
**Interpretation.**

* The summary table confirms strong right skew in the money and size variables (e.g. skewness of 9.9 for `total_price`, 12.3 for `total_freight`, 6.6 for `total_weight_g`): a median-based imputation and robust or tree-based models are better choices than linear models on raw values.
  `review_score` (mean 4.16 among 95,824 orders with a review; 646 missing) is skewed the other way (−1.48).
* **Rank correlations:** the strongest purchase-time numeric association with *duration* is the checkout promise `estimated_delivery_days` (rho = 0.523) followed by `total_freight` (0.382). For *lateness*, **no numeric purchase-time feature has |rho| above 0.052** (`total_freight` 0.052,
  `purchase_month` −0.040, `n_sellers` −0.027). Individually the numeric features are weak signals of lateness; a model must exploit geography, timing and interactions, and we should expect only modest predictive power.
* These correlations are descriptive; with n = 96k even tiny correlations are "significant", so we judge by size rather than p-values.

## 5. Predictive vs descriptive feature sets: the data-leakage warning

The dashboard answers two different questions, and they need different columns:

* **Descriptive / retrospective analytics** ("what happened and where?") may use *every* column, including delivery timestamps, processing time and review score.
* **Predictive risk** ("how likely is this order to be late, at the moment of purchase?") may use **only information that exists when the order is placed**.
"""

MD["s6_intro"] = """
`src/config.py` encodes this split as three disjoint lists that the training script and the tests enforce:

* `PURCHASE_TIME_FEATURES` — the allow-list of model inputs: order size/price/freight/weight/volume aggregates, `estimated_delivery_days` (the checkout promise), purchase month / weekday / quarter, customer state, seller state, representative product category and `same_state`.
* `LEAKAGE_COLUMNS` — deny-list: final status, post-purchase timestamps, durations and delays, processing time, review score, eligibility and exclusion bookkeeping, and flags derived from post-purchase timestamps. These either **are** the outcome or only exist after it.
* `NON_FEATURE_COLUMNS` / `PURCHASE_TIME_UNUSED` — keys, raw timestamps used for the chronological split, time-drift proxies, and purchase-time columns deliberately left out (high-cardinality cities, data-quality flags, secondary counts).

| Group | Example columns | Known at purchase? | Used by the model? |
|---|---|---|---|
| Purchase-time features | `estimated_delivery_days`, `total_freight`, `total_weight_g`, `customer_state`, `seller_state`, `product_category`, `same_state`, `purchase_month` | Yes | Allowed |
| Leakage / outcome | `delivery_delay_days`, `delivery_duration_days`, `processing_days`, `carrier_to_customer_days`, `review_score`, `order_status`, `is_late` | **No** | **Never** |
| Non-feature / unused | `order_id`, raw timestamps, `purchase_year_month`, `customer_city`, `state_pair` | Mixed | No |

The code below assigns **every processed column** to exactly one group (the notebook fails if a column is unclassified), and then makes the danger visible: a *single* column used as a score for `is_late`.
"""

MD["s6_interp"] = """
**Interpretation (this is not model performance, only the univariate ROC-AUC of each column used as a raw score; 0.5 = no information, values far below 0.5 mean the reverse ordering is informative).**

* `delivery_delay_days` has **AUC = 1.000** — by construction, because `is_late` is just "delay > 0". `delivery_duration_days` (0.947), `carrier_to_customer_days` (0.896), `review_score` (0.181, i.e. 0.819 when reversed: late orders get lower scores)
  and `processing_days` (0.651) are all strongly informative **after the fact**.
* The purchase-time columns are far weaker: the best are `same_state` (0.435, i.e. 0.565 reversed), `total_freight` (0.560), `total_price` (0.528), `estimated_delivery_days` (0.474).
* A model trained with any post-purchase column would look excellent in a notebook and be useless in the live "Delay Prediction" page, where those values do not yet exist. Conversely, the weak single-feature signals above set the expectation that a *honest* model will
  be only moderately predictive; a near-perfect validation score would be a red flag for leakage, not a success.
* `estimated_delivery_days` is allowed because the estimated date is shown to the customer at checkout. Its raw timestamp is **not** a feature (it is a split key), and `delivery_delay_days` (delivery − estimate) is not allowed because it contains the delivery date.
"""

MD["s7"] = """
## 6. Key takeaways, limitations and next steps

### Key takeaways (all numbers from the executed cells above)
1. **Analysis set:** 96,470 delivered orders (97.0% of 99,441), purchases from 2016-09-15 to 2018-08-29; **6,534 late (6.77%)** by the calendar-date rule (8.11% if a timestamp comparison were used — the difference is exactly the 1,292 orders delivered on the promised date).
2. **Delivery is usually early:** mean duration 12.6 days (median 10.2, p95 29.3) against a mean promise of 24.4 days; 91.9% of orders arrive before the estimated date. Lateness is a tail event with a heavy class imbalance (13.8 : 1).
3. **Lateness is non-stationary:** monthly late rate from ~3% (May–Oct 2017) to 12.4% (Nov 2017), 14.1% (Feb 2018) and **19.0% (Mar 2018)**, then back to 4.5% in Apr 2018. Nov 24, 2017 alone had 1,147 orders. 2016 has only 267 eligible orders.
4. **Geography is associated with lateness:** late rate from 2.76% (AM, n = 145) to 21.41% (AL, n = 397); SP (n = 40,494) 4.49% vs RJ (n = 12,350) 12.1%. Cross-state orders are late 8.05% of the time vs 4.50% for same-state orders (Cramér's V = 0.068; state: 0.136).
5. **Product attributes matter little on their own:** weight and category have weak associations (weight vs `is_late` rho = 0.016; category Cramér's V = 0.039); freight is strongly associated with duration (rho = 0.382) but weakly with lateness (0.052), likely because it reflects distance.
6. **Seller processing time is a strong retrospective marker** (late rate 3.27% when < 1 day vs 28.31% when >= 10 days) but is unknown at purchase time and is excluded from prediction.
7. **Leakage is real and measurable:** single post-purchase columns reach AUC 0.65–1.00; the best purchase-time single columns reach only 0.56–0.57 (or their reverse). Model results must be judged against that backdrop.

### Limitations
* **Selection bias:** only delivered orders with a delivery date are analysed; cancelled, unavailable and in-transit orders (2,971 orders, 2.99% of all) are excluded, so the late rate describes delivered orders only.
* **Historical and uneven period:** 2016–2018 data from one marketplace; very sparse before 2017-01 and affected by episodes (Nov 2017, Feb–Mar 2018) whose causes are not in the data. Patterns may not transfer to today.
* **No courier / GPS / weather / carrier-route / capacity data.** State, freight and `same_state` are only coarse proxies for distance and logistics. We do not infer or simulate any of these.
* **Associations, not causes:** none of the differences shown (by state, category, weight, freight, seller processing time) supports a causal claim.
* **Definition-dependent target:** the 6.77% rate depends on the calendar-date rule and on the checkout promise Olist set; a different promise policy would change what "late" means.
* **Order-level simplifications:** multi-seller orders (1.32%) use a representative seller; 1.41% of analysis orders have an unknown category and 22 have no weight.

### Next steps (Notebook 02 and the app)
* Train and compare models using **only** `config.PURCHASE_TIME_FEATURES`, with a **chronological** train/validation/test split, class-imbalance-aware metrics (PR-AUC, recall at a chosen precision, calibration), and a documented leakage audit.
* Add formal hypothesis tests with effect sizes (state, same-state, category, month) in the statistical-analysis module and surface the minimum-sample caveats in the dashboard.
* Build the Streamlit pages (Overview, Delay Analysis, Regional Intelligence, Delay Prediction, Reports) on the same processed data and the same definitions used here.
"""
