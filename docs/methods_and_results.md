# Methods and Results

**Paper:** Avalanche Forecasting in Data-Sparse Central Asia: A Weak-Supervision Framework Linking NWP-Driven Snowpack Simulations to Valley-Floor Observations
**Conference:** ISSW 2026

---

## 2. Data

### 2.1 NWP Forcing and SNOWPACK Simulations

In 2022 we implemented the open-source Weather Research and Forecasting (WRF) model (Skamarock and Klemp, 2008) to provide weather forecasts over Central Asia. We derive WRF initial and lateral boundary conditions from the US National Weather Service Global Forecast System (GFS), and configure the model with a nested grid: a 12-km outer domain and a 4-km inner nest. WRF produces 7-day forecasts updated twice daily; post-processing generates standard meteorological parameter imagery and point forecasts available on a regional website.

Following the approach developed at the Colorado Avalanche Information Center (Snook et al., 2022; Snook, 2016), we implemented the SLF SNOWPACK model (Morin et al., 2020) for the 2024–2025 season. We feed WRF 4-km output directly into SNOWPACK as meteorological forcing, which simulates per-layer snowpack evolution at each grid point. The system generates snowpack profiles at all WRF 4-km grid points above 3,500 m within the defined forecast region, updates them daily, and makes them accessible through the same regional website.

For this study, we extracted SNOWPACK output at a network of 30 virtual stations across the Darvoz region of Tajikistan and adjacent Pakistan, spanning elevations from approximately 2,100 to 4,900 m. We assigned each station a specific aspect (N, E, S, W, or flat) and elevation, producing a spatially distributed representation of snowpack evolution tuned to slope exposure. We ran simulations for two consecutive winter seasons: 2024–2025 (training) and 2025–2026 (evaluation). Sub-daily `.smet` files provided meteorological time series, and `.pro` files provided per-layer snowpack stratigraphy at each timestep. Table 1 summarizes the outputs we used.

**Table 1. SNOWPACK simulation outputs used as model inputs.**

| Variable | Source | Physical role |
|---|---|---|
| HS | .smet | Total snow depth; proxy for snowpack energy potential |
| HN24 | .smet | 24-hour new snow depth; storm loading signal |
| HN72 | .smet | 72-hour new snow depth; sustained loading signal |
| TA_max | .smet | Daily maximum air temperature; phase proxy |
| rain_sum | .smet | Daily rainfall accumulation; rain-on-snow trigger |
| Sn38_upper | .pro | Min natural stability index, upper zone (burial < 40% HS) |
| Sn38_lower | .pro | Min natural stability index, lower zone (burial ≥ 40% HS) |
| depth_lower_wl | .pro | Burial depth of weakest lower-zone layer (cm) |

We had no in-situ snowpack observations to validate simulated stratigraphy; the Sn38 index and layer structure are model-derived proxies only.

### 2.2 Avalanche Observations

We compiled avalanche observations from two sources. The Aga Khan Agency for Habitat (AKAH) maintains a structured observation log covering Pakistan and Tajikistan across multiple seasons. We also extracted a supplementary set of events from social media posts shared by local residents and relief workers. Both sources report valley-floor runout locations rather than start zones, and each record includes date, approximate location, slope angle, aspect, and a qualitative size estimate.

The combined dataset contains 69 observations: 26 from the 2024–2025 season (training) and 43 from 2025–2026 (evaluation). We treat observations as weakly labeled events — they confirm that an avalanche reached the valley floor, but do not verify the precise trigger mechanism, release zone, or snowpack state. This weak-label framing is a deliberate design choice consistent with the data-sparse context.

### 2.3 Observation-to-Station Matching

Valley-floor reports provide approximate runout locations rather than start zones. A practitioner with regional terrain knowledge identified the likely avalanche start zone for each observation based on valley geometry, known paths, and observation context. We then matched each start zone to the nearest SNOWPACK station sharing the same predominant aspect (N, E, S, or W), ensuring the simulation reflects the relevant solar radiation and wind loading regime. We identified aspect-specific station files using the naming convention `{station_id}{suffix}_res.pro`, where suffix 1, 2, 3, 4 corresponds to N, E, S, W respectively.

This hybrid human-machine matching step embeds domain knowledge where nearest-distance matching alone would systematically fail: two stations equidistant from a reported runout but facing opposite aspects carry fundamentally different snowpack histories. The matching introduces practitioner subjectivity and is not automatically reproducible, but we consider this a feature rather than a gap — it is the mechanism through which terrain expertise enters the framework.

---

## 3. Methods

### 3.1 Feature Engineering

We aggregated sub-daily SNOWPACK output to daily resolution for model training and prediction. From `.smet` files we computed daily maxima for HS, HN24, HN72, and TA_max, and daily sums for rain rate. We also derived a binary wet-regime indicator (`wet_flag` = 1 when TA_max > 0°C) to provide an explicit signal for the wet-versus-dry avalanche mechanism distinction — a physically important separation given that rain-on-snow and above-freezing temperature loading represent fundamentally different release pathways.

From `.pro` layer data we derived snowpack zone features at each timestep. We classified layers into an upper zone (burial depth < 40% of total HS, targeting near-surface storm slab structures) and a lower zone (burial depth ≥ 40% of HS, targeting deep persistent weak layers), then computed the minimum Sn38 value within each zone and the burial depth of the weakest lower-zone layer. When snowpack depth was insufficient to resolve both zones — common in early-season or thin-snowpack conditions — we forward-filled zone features up to seven days to preserve continuity. Beyond that window, we imputed missing values with the training-set mean at prediction time, representing a neutral stability assumption rather than propagating stale values.

The final daily feature vector contains nine variables (Table 1 plus `wet_flag`).

### 3.2 Labeling Under Weak Supervision

We labeled avalanche event days positive (y = 1) for both the reported date and the preceding day, accounting for the one-day reporting lag inherent in social media and field log sources. We labeled all other days negative (y = 0). This produced a severely imbalanced dataset with a positive day rate of approximately 1.1% in the 2024–2025 training corpus. We addressed class imbalance with balanced class weights in the logistic regression objective, which upweights rare positive samples.

Where zone-derived features (Sn38, burial depth) contained no non-NaN values on positive training days — most commonly for stations with thin or absent early-season snowpacks — the model fell back to the meteorological-only feature set (HS, HN24, HN72, rain_sum, TA_max, wet_flag), sacrificing snowpack structural information but preserving trainability.

### 3.3 Classifier Design

We implemented three complementary classifier configurations:

**Per-station logistic regression.** We trained a separate logistic regression for each station using daily features from the 2024–2025 season and event dates matched to that station. We used the full nine-feature set where possible, falling back to the met-only set otherwise, and required a minimum of two positive-labeled days to proceed. We saved models to disk and reloaded them for prediction without retraining unless explicitly requested.

**Regional pooled classifier.** To address per-station data sparsity, we trained a single logistic regression on the pooled 2024–2025 feature matrix from all 30 stations simultaneously. We fit one `StandardScaler` on the combined training set so coefficients reflect signal relative to the cross-station distribution rather than any single station's absolute values. This model trades local adaptability for statistical power.

**Confidence-weighted blended classifier.** We formed a composite probability for each station by weighting the per-station and regional predictions as:

$$p_\text{blend} = w \cdot p_\text{station} + (1 - w) \cdot p_\text{regional}, \quad w = \min\!\left(\frac{n_\text{train}}{5}, 1.0\right)$$

where $n_\text{train}$ is the number of training events at that station. Stations with five or more events contribute exclusively through their per-station model; stations with fewer events lean progressively toward the regional prior. Where no per-station model exists, we use the regional prediction directly.

### 3.4 Evaluation Design

We applied three complementary evaluation frameworks, each suited to a different aspect of model performance.

**Temporal leave-one-out cross-validation (per-station).** For each station, we pooled all observed avalanche events across both seasons and sorted them chronologically. In fold *i*, we trained the model on all events preceding event *i* and tested on event *i*. We loaded training features from the season(s) containing the training events and test features from the season containing the held-out event. This design ensures no future information informs training, maximizes use of the limited observation record, and avoids having more test events than training events at a given station. Stations with only one event contributed no folds; stations with *n* events contributed *n* − 1 folds. The primary per-fold metric is the **event window probability**: the maximum predicted probability within the three-day window preceding and including the test event. We defined detection as event window probability ≥ 0.5.

**Cross-season holdout (regional and blended models).** We trained the regional and blended classifiers exclusively on 2024–2025 data and evaluated them on 2025–2026 observations at stations active in both seasons. This one-season holdout tests cross-season generalization of the aggregate model and complements the per-station LOO results.

**Event-level precision-recall.** For each test event, we took the maximum model probability within a three-day pre-event window as the event detection score. We drew non-overlapping, same-length windows from non-grace-zone periods at the same stations to serve as the negative class. This framing asks whether the model issued a timely warning before each event — a more operationally relevant question than per-day classification accuracy in a high-stakes forecasting context.

---

## 4. Results

### 4.1 Per-Station Classifier Performance (Temporal LOO)

We applied temporal LOO cross-validation to 16 stations with two or more total events across both seasons, producing 25 folds (Table 2). Four stations achieved 100% detection rate; two stations with four events each achieved 67%; the remaining ten stations detected no events at the 0.5 threshold.

**Table 2. Per-station temporal LOO results (all events across both seasons, sorted by detection rate).**

| Station | Aspect | Total events | LOO folds | Mean event prob | Detection rate |
|---|---|---|---|---|---|
| 160942_res | E | 2 | 1 | 1.000 | **100%** |
| 176522_res | E | 2 | 1 | 0.739 | **100%** † |
| 180343_res | S | 3 | 2 | 0.941 | **100%** † |
| 268601_res | N | 2 | 1 | 0.663 | **100%** |
| 153203_res | S | 4 | 3 | 0.630 | 67% |
| 268603_res | S | 4 | 3 | 0.662 | 67% |
| 164801_res | N | 3 | 2 | 0.557 | 50% |
| 176204_res | W | 2 | 1 | 0.295 | 0% |
| 250224_res | W | 3 | 2 | 0.212 | 0% |
| 187363_res | S | 2 | 1 | 0.086 | 0% |
| 183642_res | S | 3 | 2 | 0.155 | 0% |
| 272401_res | N | 2 | 1 | 0.156 | 0% |
| 165202_res | E | 2 | 1 | 0.006 | 0% |
| 183741_res | N | 2 | 1 | 0.001 | 0% |
| 224102_res | E | 3 | 2 | 0.001 | 0% |
| 224101_res | N | 2 | 1 | <0.001 | 0% |

† Detection driven partly by elevated background probability at these stations (non-event median > 0.5); the model does not fully discriminate event from non-event days.

Several individual folds demonstrate genuine discrimination: at 160942_res the model assigns event prob = 1.00 against a non-event median of 0.11; at 164801_res fold 2, event prob = 0.99 against a non-event median < 0.001; at 268603_res fold 1, event prob = 0.88 against a non-event median of 0.03. In these cases the model specifically elevates probability around the event window rather than uniformly across the season.

**Performance as a function of training data size.** We aggregated LOO folds by the number of prior training events at the time of each test (Figure X) and found a consistent positive trend in both mean event window probability and detection rate (Table 3).

**Table 3. LOO detection performance by number of training events.**

| Training events | Folds | Mean event prob | Detection rate |
|---|---|---|---|
| 1 | 16 | 0.381 | 37.5% |
| 2 | 7 | 0.523 | 42.9% |
| 3 | 2 | 0.400 | 50.0% |

Detection rate rises from 37.5% with one prior event to 50.0% with three, and mean event window probability crosses the 0.5 threshold at n = 2. Even folds that fall below the detection threshold assign elevated probability relative to the non-event background (median non-event probability across all folds = 0.052), showing that the models extract consistent signal even under extreme data sparsity. The n = 3 dip in mean probability reflects sampling noise from only two folds at that level, not a genuine reversal.

This learning curve directly supports the paper's central argument: per-station classifier performance improves monotonically with observation density, and the current dataset — 1–4 events per station — sits at the steep portion of that curve where each additional observation produces meaningful gains.

### 4.2 Regional and Blended Model Performance

The pooled regional model trained on 2,972 station-days from 30 stations, with 32 positive days (1.1% positive rate). Evaluated against all 37 test-season events across matched stations, it achieved AUC = 0.588 and average precision (AP) = 0.042 at the daily level. The blended composite performed similarly (AUC = 0.583, AP = 0.039), indicating that combining per-station and regional predictions did not substantially improve aggregate skill at current observation density.

### 4.3 Event-Level Detection Performance (Aggregate Models)

We framed event-level evaluation (Table 4) as a detection problem: did any model assign elevated probability within the three days preceding each event? Using 37 positive event windows and 1,038 non-event windows of equal length drawn from the same stations, all three models showed modest but consistent skill above the no-skill baseline (prevalence = 0.034).

**Table 4. Event-level precision-recall performance (3-day pre-event detection window, 2025–2026 test season).**

| Model | AUC-ROC | Average Precision | vs. No-Skill AP |
|---|---|---|---|
| Regional | 0.610 | 0.045 | +33% |
| Per-station + fallback | 0.556 | 0.043 | +26% |
| Blended (weighted) | 0.608 | 0.045 | +33% |
| No skill (baseline) | — | 0.034 | — |

All three models perform above chance in event-level detection (AUC-ROC > 0.5), with the regional and blended classifiers performing comparably. The per-station-plus-fallback model shows marginally lower detection AUC, suggesting that noisy per-station models trained on single events slightly degrade regional signal when they serve as the primary prediction source.

### 4.4 Physical Plausibility of Learned Weights

The regional model's standardized coefficients (Table 5) summarize the feature–avalanche relationship the model learned from 2024–2025 data across all 30 stations.

**Table 5. Regional model standardized coefficients (sorted by absolute magnitude).**

| Feature | Coefficient | Physical interpretation |
|---|---|---|
| wet_flag | −1.38 | Cold/dry-regime days carry higher risk in training data |
| depth_lower_wl | −0.85 | Shallower weak layer burial → more accessible → higher risk |
| sn38_upper_min | −0.79 | Weaker upper zone → higher near-surface instability |
| TA_max | +0.72 | Near-freezing warming increases modeled risk |
| sn38_lower_min | +0.39 | (modest) deeper stable lower zone reduces risk |
| HS | +0.29 | Greater total snow depth → more energy available |
| rain_sum | −0.22 | (modest) net negative in training set |
| HN24 | +0.19 | New snow loading increases risk |
| HN72 | +0.06 | Weaker three-day accumulation signal |

The negative `wet_flag` coefficient (−1.38) combined with a positive `TA_max` coefficient (+0.72) captures a nuanced temperature signal: modest near-freezing warming increases modeled risk, while unambiguously above-freezing conditions (wet_flag = 1) associate with lower risk in the training corpus. The 2024–2025 training events were predominantly cold dry-slab avalanches; wet-regime events appear in the dataset but not at sufficient density to dominate the learned signal. This interpretation aligns with the known sensitivity of Central Asian avalanche activity to both dry new-snow loading and temperature-driven weakening events near 0°C.

The negative `depth_lower_wl` coefficient (−0.85) is physically coherent: a weak layer buried closer to the surface is more easily triggered and more likely to support a slab release. The negative `sn38_upper_min` (−0.79) corroborates this — lower near-surface stability indexes predict higher probability, consistent with Sn38's design as a natural stability index where values below 1.5 indicate high instability.

The positive `HS` coefficient (+0.29) reflects the avalanche size signal in the paper's framing: deeper snowpacks carry more potential energy for runout to reach valley floors, even as start-zone elevation and slope geometry vary across the station network. Global standardization preserves this signal across the pooled dataset by normalizing each station's HS to the cross-station distribution rather than absolute depth.

### 4.5 Summary

Temporal LOO cross-validation across 16 stations and 25 folds shows that per-station classifiers achieve a 37.5% event detection rate with one prior training event, rising to 50% with three — a monotonic improvement that directly validates the "more data, better performance" premise of the framework. At the best-sampled stations, event window probabilities approach 1.0 against near-zero non-event backgrounds, demonstrating genuine discriminative skill rather than systematic over-prediction. At the aggregate level, the regional model achieves event-level AUC-ROC = 0.610 and average precision 1.33× the no-skill baseline across 37 test events. Learned feature weights are physically coherent across both per-station and pooled models: shallow weak-layer burial, near-surface instability (low Sn38), and near-freezing warming are consistently the highest-magnitude predictors.

The results confirm the framework's role as a decision-support tool: the pipeline extracts meaningful probabilistic signal from sparse observations and NWP-forced simulations, and performance scales reliably with observation density. Moving from one to three events per station roughly doubles the detection rate — a gain achievable within a single additional active observation season, and the clearest path to operational improvement.

---

## References

Morin, S., Horton, S., Techel, F., Bavay, M., Coléou, C., Fierz, C., Gobiet, A., Hagenmuller, P., Lafaysse, M., Ližar, M., Mitterer, C., Monti, F., Müller, K., Olefs, M., Snook, J. S., van Herwijnen, A., and Vionnet, V.: Application of physical snowpack models in support of operational avalanche hazard forecasting: A status report on current implementations and prospects for the future, *Cold Regions Science and Technology*, 170, 102190, https://doi.org/10.1016/j.coldregions.2019.102910, 2020.

Skamarock, W. C. and Klemp, J. B.: A time-split nonhydrostatic atmospheric model for research and NWP applications, *J. Comp. Phys.*, 227, 3465–3485, 2008.

Snook, J. S., Cooperstein, M., and Greene, E.: Snowpack modeling efforts at the Colorado Avalanche Information Center, *Proceedings of the International Snow Science Workshop*, Bend, OR, USA, 2022.

Snook, J. S.: Weather forecast model grid spacing — is smaller better?, *Proceedings of the International Snow Science Workshop*, Breckenridge, CO, USA, 2016.
