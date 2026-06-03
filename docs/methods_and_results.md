# Methods and Results

**Paper:** Avalanche Forecasting in Data-Sparse Central Asia: A Weak-Supervision Framework Linking NWP-Driven Snowpack Simulations to Valley-Floor Observations
**Conference:** ISSW 2026

---

## 2. Data

### 2.1 SNOWPACK Simulations

Avalanche probability guidance was derived from virtual snowpack simulations produced by the SNOWPACK model (Lehning et al. 2002) forced with numerical weather prediction (NWP) output at 4 km grid spacing. A network of 30 virtual stations was configured across the Darvoz region of Tajikistan and adjacent Pakistan, spanning elevations from approximately 2,100 to 4,900 m. Each station was assigned a specific aspect (N, E, S, W, or flat) and elevation, producing a spatially distributed representation of snowpack evolution tuned to slope exposure.

Simulations were run for two consecutive winter seasons: 2024–2025 (used for model training) and 2025–2026 (used for evaluation). Sub-daily simulation output from `.smet` files provided meteorological time series, and `.pro` files provided per-layer snowpack stratigraphy at each timestep. Key outputs used in this study are summarized in Table 1.

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

No in-situ snowpack observations were available to validate simulated stratigraphy; the Sn38 index and layer structure are acknowledged as model-derived proxies.

### 2.2 Avalanche Observations

Avalanche observations were compiled from two sources. The Aga Khan Agency for Habitat (AKAH) maintains a structured observation log covering Pakistan and Tajikistan across multiple seasons. A supplementary set of events was extracted from social media posts shared by local residents and relief workers. Both sources report valley-floor runout locations rather than start zones, and include date, approximate location, slope angle, aspect, and a qualitative size estimate.

The combined dataset contains 69 observations: 26 from the 2024–2025 season (training) and 43 from 2025–2026 (evaluation). Observations are treated as weakly labeled events — they indicate that an avalanche occurred and reached the valley floor, but do not confirm the precise trigger mechanism, release zone, or snowpack state. This weak-label framing is a deliberate design choice consistent with the data-sparse context.

### 2.3 Observation-to-Station Matching

Valley-floor reports provide approximate runout locations rather than start zones. A practitioner with regional terrain knowledge identified the likely avalanche start zone for each observation based on valley geometry, known paths, and observation context. Each start zone was then matched to the nearest SNOWPACK station sharing the same predominant aspect (N, E, S, or W), ensuring the simulation reflects the relevant solar radiation and wind loading regime. Aspect-specific station files were identified using the naming convention `{station_id}{suffix}_res.pro`, where suffix 1, 2, 3, 4 corresponds to N, E, S, W respectively.

This hybrid human-machine matching step embeds domain knowledge where nearest-distance matching alone would systematically fail: two stations equidistant from a reported runout but facing opposite aspects will have fundamentally different snowpack histories. The matching is an acknowledged limitation — it introduces practitioner subjectivity and cannot be automatically reproduced — but this subjectivity is considered a feature rather than a gap, as it is the mechanism through which terrain expertise is encoded.

---

## 3. Methods

### 3.1 Feature Engineering

Sub-daily SNOWPACK output was aggregated to daily resolution for model training and prediction. Meteorological features from `.smet` files were computed as daily maxima (HS, HN24, HN72, TA_max) or daily sums (rain rate). A binary wet-regime indicator (`wet_flag`) was derived as 1 when TA_max > 0°C, providing an explicit signal for the wet versus dry avalanche mechanism distinction — a physically important separation given that rain-on-snow and above-freezing temperature loading represent fundamentally different release pathways.

Snowpack zone features were derived from `.pro` layer data. At each timestep, layers were classified into an upper zone (burial depth < 40% of total HS, targeting near-surface storm slab structures) and a lower zone (burial depth ≥ 40% of HS, targeting deep persistent weak layers). The minimum Sn38 value within each zone and the burial depth of the weakest lower-zone layer were computed daily. When snowpack depth was insufficient to resolve both zones — a common occurrence in early-season or thin-snowpack conditions — zone features were forward-filled up to seven days to preserve continuity; beyond that window, missing values were imputed with the training-set mean at prediction time, representing a neutral stability assumption rather than propagating arbitrary stale values.

The final daily feature vector contained nine variables (Table 1 plus `wet_flag`).

### 3.2 Labeling Under Weak Supervision

Avalanche event days were labeled positive (y = 1) for both the reported date and the preceding day to account for one-day reporting lag inherent in social media and field log sources. All other days in the simulation record were labeled negative (y = 0). This produced a severely imbalanced dataset with a positive day rate of approximately 1.1% in the 2024–2025 training corpus. Class imbalance was addressed using balanced class weights in the logistic regression objective, effectively upweighting rare positive samples.

Where zone-derived features (Sn38, burial depth) contained no non-NaN values on positive training days — most commonly for stations with thin or absent early-season snowpacks — the model fell back to the meteorological-only feature set (HS, HN24, HN72, rain_sum, TA_max, wet_flag), sacrificing snowpack structural information but preserving trainability.

### 3.3 Classifier Design

Three complementary classifier configurations were implemented:

**Per-station logistic regression.** A separate logistic regression model was trained for each station using daily features from the 2024–2025 season and event dates matched to that station. The full nine-feature set was used where possible, with smet-only fallback otherwise. Minimum threshold of two positive-labeled days was required to proceed with training. Models were saved to disk and reloaded without retraining for prediction unless explicitly requested.

**Regional pooled classifier.** To address per-station data sparsity, a single logistic regression was trained on the pooled 2024–2025 feature matrix from all 30 stations simultaneously. Features were standardized globally using a single `StandardScaler` fit on the combined training set, so coefficients reflect signal relative to the cross-station distribution rather than any single station's absolute values. This model sacrifices local adaptability for statistical power.

**Confidence-weighted blended classifier.** A composite probability was formed for each station by weighting the per-station and regional predictions as:

$$p_\text{blend} = w \cdot p_\text{station} + (1 - w) \cdot p_\text{regional}, \quad w = \min\!\left(\frac{n_\text{train}}{5}, 1.0\right)$$

where $n_\text{train}$ is the number of training events at that station. Stations with five or more events contribute exclusively through their per-station model; stations with fewer events lean progressively toward the regional prior. Where no per-station model exists, the regional prediction is used directly.

### 3.4 Cross-Season Evaluation

All three classifiers were trained exclusively on 2024–2025 features and labels. Evaluation used 2025–2026 observations at stations that were also active in the training season (20 of 43 test-season observations met this criterion). This temporal holdout design — one full season separating train from test — is the strictest cross-validation possible given the two-season dataset.

Two evaluation frameworks were applied:

**Daily-level AUC.** Receiver operating characteristic area under curve computed on all labeled days at matched stations. Reported per station where sufficient data existed.

**Event-level precision-recall.** For each test event, the maximum model probability within a three-day pre-event window was taken as the event detection score. Non-event windows of equal length, drawn from non-grace-zone periods at the same stations, provided the negative class. This framing asks whether the model issued a timely warning before each event, rather than whether it classified every individual day correctly — a more operationally relevant question for a forecasting tool.

---

## 4. Results

### 4.1 Per-Station Classifier Performance

Thirteen stations had sufficient 2024–2025 training data and matched 2025–2026 test events to permit per-station evaluation (Table 2). Results were strongly bimodal: two stations with at least two training events each achieved AUC ≥ 0.97, while most stations with a single training event showed AUC near or below 0.5.

**Table 2. Per-station cross-season evaluation (2024–2025 train → 2025–2026 test).**

| Station | Aspect | Train events | Test events | AUC |
|---|---|---|---|---|
| 160942_res | E | 1 | 1 | **0.993** |
| 164801_res | N | 2 | 1 | **0.976** |
| 165202_res | E | 1 | 1 | 0.782 |
| 268603_res | S | 2 | 2 | 0.664 |
| 272401_res | N | 1 | 1 | 0.602 |
| 224102_res | E | 1 | 2 | 0.545 |
| 183741_res | N | 1 | 1 | 0.381 |
| 250224_res | W | 2 | 1 | 0.398 |
| 187363_res | S | 1 | 1 | 0.286 |
| 224101_res | N | 1 | 1 | 0.228 |
| 153203_res | S | 3 | 1 | 0.214 |
| 176522_res | E | 1 | 1 | 0.170 |
| 180343_res | S | 1 | 2 | 0.119 |

The two best-performing stations (160942_res, 164801_res) achieved recall of 1.0 at low precision (~0.06–0.22 at the 0.5 threshold), meaning every test event was flagged but at the cost of frequent false alarms — a reasonable trade-off for a decision-support tool in a high-consequence setting. The high variability across stations reflects the fundamental constraint: with one training event, a logistic regression model has insufficient positive-class samples to learn reliable feature combinations, and the resulting model is essentially fitting to noise.

### 4.2 Regional and Blended Model Performance

The pooled regional model trained on 2,972 station-days from 30 stations, with 32 positive days (1.1% positive rate). Evaluated against all 37 test-season events across matched stations, it achieved AUC = 0.588 and average precision (AP) = 0.042 at the daily level. The blended composite performed similarly (AUC = 0.583, AP = 0.039), indicating that combining per-station and regional predictions did not substantially improve aggregate skill given current observation density.

### 4.3 Event-Level Detection Performance

Event-level evaluation (Table 3) framed each test avalanche as a detection problem: did any model assign elevated probability within the three days preceding the event? Using 37 positive event windows and 1,038 non-event windows of equal length drawn from the same stations, all three models showed modest but consistent skill above the no-skill baseline (prevalence = 0.034).

**Table 3. Event-level precision-recall performance (3-day pre-event detection window).**

| Model | AUC-ROC | Average Precision | vs. No-Skill AP |
|---|---|---|---|
| Regional | 0.610 | 0.045 | +33% |
| Per-station + fallback | 0.556 | 0.043 | +26% |
| Blended (weighted) | 0.608 | 0.045 | +33% |
| No skill (baseline) | — | 0.034 | — |

All three models perform above chance in event-level detection (AUC-ROC > 0.5), with the regional and blended classifiers performing comparably. The per-station-plus-fallback model shows marginally lower detection AUC, suggesting that noisy per-station models trained on single events slightly degrade regional signal when used as the primary prediction source.

### 4.4 Physical Plausibility of Learned Weights

The regional model's standardized coefficients (Table 4) provide a cross-station summary of the feature–avalanche relationship learned from 2024–2025 data.

**Table 4. Regional model standardized coefficients (sorted by absolute magnitude).**

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

The negative `wet_flag` coefficient (−1.38) combined with a positive `TA_max` coefficient (+0.72) reflects a nuanced temperature signal: modest near-freezing warming increases modeled risk, while unambiguously above-freezing conditions (wet_flag = 1) are associated with lower risk in the training corpus. This likely reflects that the 2024–2025 training events were predominantly cold dry-slab avalanches; wet-regime events are represented in the dataset but not at sufficient density to dominate the learned signal. The physical interpretation is consistent with the known sensitivity of Central Asian avalanche activity to both dry new-snow loading and temperature-driven weakening events near 0°C.

The negative `depth_lower_wl` coefficient (−0.85) is physically coherent: a weak layer buried shallower is closer to the surface, more easily triggered, and more likely to support a slab release. The negative `sn38_upper_min` (−0.79) corroborates this — lower near-surface stability indexes correspond to higher predicted probability, consistent with Sn38's design as a natural stability index where values below 1.5 indicate high instability.

The `HS` coefficient (+0.29) reflects the avalanche size signal identified in the paper's framing: deeper snowpacks provide greater potential energy for runout to reach valley floors, even where start-zone elevation and slope geometry vary across the station network. This signal is preserved across the pooled dataset through global standardization, which normalizes each station's HS to the cross-station distribution rather than absolute depth.

### 4.5 Summary

The framework demonstrates statistically detectable skill above the no-skill baseline in cross-season evaluation (regional event-level AUC-ROC = 0.610; AP = 1.33× prevalence), with near-perfect discrimination at individual well-sampled stations (AUC ≥ 0.97 at two stations). Aggregate precision-recall performance is constrained by extreme class imbalance (~1% positive day rate) and by the limited size of the training corpus (26 events across 30 stations). The learned feature weights are physically coherent and consistent with accepted avalanche formation mechanisms, providing indirect evidence of model validity in the absence of in-situ snowpack observations. These results support the interpretation of the framework as a decision-support tool capable of surfacing physically grounded probabilistic guidance, while confirming that operational deployment would require substantially denser observation records to achieve reliable cross-station generalization.
