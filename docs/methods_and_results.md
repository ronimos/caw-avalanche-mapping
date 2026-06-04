# Methods and Results

**Paper:** Avalanche Forecasting in Data-Sparse Central Asia: A Weak-Supervision Framework Linking NWP-Driven Snowpack Simulations to Valley-Floor Observations
**Conference:** ISSW 2026

---

## 2. Data

### 2.1 NWP Forcing and SNOWPACK Simulations

In 2022 we implemented the open-source Weather Research and Forecasting (WRF) model (Skamarock and Klemp, 2008) to provide weather forecasts over Central Asia. We derive WRF initial and lateral boundary conditions from the US National Weather Service Global Forecast System (GFS), and configure the model with a nested grid: a 12-km outer domain and a 4-km inner nest. WRF produces 7-day forecasts updated twice daily; post-processing generates standard meteorological parameter imagery and point forecasts available on a regional website.

Following the approach developed at the Colorado Avalanche Information Center (Snook et al., 2022; Snook, 2016), we implemented the SLF SNOWPACK model (Lehning et al., 2002; Morin et al., 2020) for the 2024–2025 season. We feed WRF 4-km output directly into SNOWPACK as meteorological forcing, which simulates per-layer snowpack evolution at each grid point. The system generates snowpack profiles at all WRF 4-km grid points above 3,500 m within the defined forecast region, updates them daily, and makes them accessible through the same regional website.

For this study, we extracted SNOWPACK output at a network of 30 virtual stations across the Darvoz region of Tajikistan and adjacent Pakistan, spanning elevations from approximately 2,100 to 4,900 m. We assigned each station a specific aspect (N, E, S, W, or flat) and elevation, producing a spatially distributed representation of snowpack evolution tuned to slope exposure. We ran simulations for two consecutive winter seasons: 2024–2025 (training) and 2025–2026 (evaluation). Sub-daily `.smet` files provided meteorological time series, and `.pro` files provided per-layer snowpack stratigraphy at each timestep. Table 1 summarizes the outputs we used.

**Table 1. Candidate SNOWPACK simulation outputs.** *(The model uses a reduced subset — see §3.1 and Table 2.)*

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

The combined dataset contains 69 observations: 26 from the 2024–2025 season (training) and 43 from 2025–2026 (evaluation). We treat observations as weakly labeled events in two senses. First, a positive label confirms that an avalanche reached the valley floor but does not verify the precise trigger mechanism, release zone, or snowpack state. Second, and more fundamentally, the absence of an observation does not confirm the absence of an avalanche: our records are an opportunistic, almost certainly incomplete sample of activity, so an unknown number of true events go unreported and are silently labeled negative. This is a positive-unlabeled (PU) setting (Elkan and Noto, 2008) rather than a clean positive/negative one — the "negative" class is contaminated with unlabeled positives, which biases trained probabilities downward and makes our reported skill a conservative lower bound on what the underlying signal could support. We deliberately do not attempt a formal PU correction: reliable estimation of the label frequency is infeasible at this sample size (~35 positive days), the strongly non-random reporting process (large avalanches near villages and roads are far more likely to be recorded) violates the selected-completely-at-random (SCAR) assumption that standard PU corrections require (Elkan and Noto, 2008), and our headline metrics (AUC-ROC, average precision) are rank-based and therefore invariant to the monotonic probability rescaling such a correction would apply. We instead carry the PU contamination as an acknowledged limitation. This weak-label framing is a deliberate design choice consistent with the data-sparse context.

### 2.3 Observation-to-Station Matching

Valley-floor reports provide approximate runout locations rather than start zones. A practitioner with regional terrain knowledge identified the likely avalanche start zone for each observation based on valley geometry, known paths, and observation context. We then matched each start zone to the nearest SNOWPACK station sharing the same predominant aspect (N, E, S, or W), ensuring the simulation reflects the relevant solar radiation and wind loading regime. We identified aspect-specific station files using the naming convention `{station_id}{suffix}_res.pro`, where suffix 1, 2, 3, 4 corresponds to N, E, S, W respectively.

This hybrid human-machine matching step embeds domain knowledge where nearest-distance matching alone would systematically fail: two stations equidistant from a reported runout but facing opposite aspects carry fundamentally different snowpack histories. The matching introduces practitioner subjectivity and is not automatically reproducible, but we consider this a feature rather than a gap — it is the mechanism through which terrain expertise enters the framework.

---

## 3. Methods

### 3.1 Feature Engineering

We aggregated sub-daily SNOWPACK output to daily resolution for model training and prediction. From `.smet` files we computed daily maxima for HS, HN24, HN72, and TA_max, and daily sums for rain rate. From `.pro` layer data we classified layers at each timestep into an upper zone (burial depth < 40% of total HS, targeting near-surface storm slabs) and a lower zone (burial depth ≥ 40% of HS, targeting deep persistent weak layers), and computed the minimum Sn38 within each zone plus the burial depth of the weakest lower-zone layer. We also computed a whole-profile minimum stability index, `sn38_min`. When snowpack depth was insufficient to resolve both zones — common in early-season or thin-snowpack conditions — we forward-filled zone features up to seven days, then imputed any remaining missing values with the training-set mean at prediction time, representing a neutral stability assumption rather than propagating stale values.

**Feature selection.** With only ~35 positive (event) days in the training corpus, a nine-feature model is severely over-parameterized — roughly 3.5 events per predictor, against the standard 10–20 rule of thumb for logistic regression — which produces unstable, overfit coefficients. We therefore reduced the feature set to five physically distinct, minimally collinear variables (Table 2): total snow depth (HS, a size / valley-reach proxy), 24-hour new snow (HN24, the primary loading trigger), maximum air temperature (TA_max, the wet-versus-dry mechanism signal), the whole-profile minimum stability index (sn38_min), and the burial depth of the weakest lower-zone layer (depth_lower_wl). We dropped HN72 (collinear with HN24), the binary `wet_flag` (a threshold of TA_max), and rain_sum (sparse, its signal overlapping TA_max), and collapsed the upper/lower Sn38 minima into the single `sn38_min`. The full nine-feature set is retained only for ablation.

**Table 2. Reduced five-feature set used for all models.**

| Feature | Physical role |
|---|---|
| HS | Total snow depth — size / valley-reach potential |
| HN24 | 24-hour new snow — primary loading trigger |
| TA_max | Daily max air temperature — wet vs dry mechanism |
| sn38_min | Whole-profile minimum stability index — weakest-layer strength |
| depth_lower_wl | Burial depth of weakest lower-zone layer |

### 3.2 Labeling Under Weak Supervision

We labeled avalanche event days positive (y = 1) for both the reported date and the preceding day, accounting for the one-day reporting lag inherent in social media and field log sources. We labeled all other days negative (y = 0). This produced a severely imbalanced dataset with a positive day rate of approximately 1.1% in the 2024–2025 training corpus. The frequentist logistic models address this imbalance with balanced class weights, which upweight rare positive samples; the hierarchical Bayesian model instead keeps the natural base rate and centers its intercept prior on the empirical log-odds, yielding calibrated (low-magnitude) probabilities evaluated with threshold-free metrics.

Where zone-derived features contained no non-NaN values on positive training days — most commonly for stations with thin or absent early-season snowpacks — the model fell back to a meteorological-only feature set (HS, HN24, TA_max), sacrificing snowpack structural information but preserving trainability.

### 3.3 Classifier Design

We implemented four classifier configurations. The first three are L2-regularized frequentist logistic regressions (scikit-learn; Pedregosa et al., 2011); the fourth is a hierarchical Bayesian model that unifies them. All use the reduced five-feature set and standardized inputs.

**Per-station logistic regression.** We trained a separate model for each station using its 2024–2025 features and matched event dates, requiring a minimum of two positive-labeled days. Because 1–2 positive samples cannot support a tuned penalty, we applied strong fixed regularization (C = 0.1). Models are saved and reloaded without retraining unless requested.

**Regional pooled classifier.** To address per-station sparsity, we trained a single logistic regression on the pooled 2024–2025 feature matrix from all 30 stations, fitting one `StandardScaler` on the combined set so coefficients reflect cross-station signal. We selected the regularization strength C by stratified five-fold cross-validation scoring average precision (selected C = 0.3). This model trades local adaptability for statistical power.

**Confidence-weighted blended classifier.** We formed a composite per-station probability by weighting the per-station and regional predictions as $p_\text{blend} = w\,p_\text{station} + (1-w)\,p_\text{regional}$ with $w = \min(n_\text{train}/5,\,1)$. This is a hand-built approximation of partial pooling; the hierarchical model below performs the same shrinkage in a principled, data-driven way.

**Hierarchical Bayesian model (partial pooling).** We fit a single multilevel logistic regression (partial pooling; Gelman and Hill, 2007) in which each station's intercept and slopes are drawn from shared regional distributions:

$$\text{logit}(p_{s,t}) = \alpha_s + \boldsymbol{\beta}_s \cdot \mathbf{x}_{s,t}, \qquad \alpha_s \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha), \quad \beta_{s,k} \sim \mathcal{N}(\mu_{\beta,k}, \sigma_{\beta,k})$$

with weakly-informative hyperpriors and a non-centered parameterization for stable sampling (Betancourt and Girolami, 2015). The between-station spreads ($\sigma_\alpha, \sigma_{\beta,k}$) are learned from the data, so each station's coefficients shrink toward the regional mean by an amount the data dictates: stations with many events retain their own signal, stations with one event shrink strongly toward the population. This replaces the three models above with one coherent framework, and the posterior provides a per-station prediction *and* its uncertainty (the spread across posterior draws) — the uncertainty directly supplying a confidence measure for operational display. We trained on the 13 stations with at least one event (event-free stations contribute only unconstrained noise to the hierarchy and are served at prediction time by the population-level coefficients), centering the intercept prior on the empirical log-odds to keep probabilities calibrated. We sampled with the No-U-Turn Sampler (NUTS; Hoffman and Gelman, 2014) as implemented in PyMC (Abril-Pla et al., 2023) — two chains, 1000 tuning + 1000 draws each; zero divergences.

### 3.4 Evaluation Design

We applied three complementary evaluation frameworks, each suited to a different aspect of model performance.

**Temporal leave-one-out cross-validation (per-station).** For each station, we pooled all observed avalanche events across both seasons and sorted them chronologically. In fold *i*, we trained the model on all events preceding event *i* and tested on event *i*. We loaded training features from the season(s) containing the training events and test features from the season containing the held-out event. This design ensures no future information informs training, maximizes use of the limited observation record, and avoids having more test events than training events at a given station. Stations with only one event contributed no folds; stations with *n* events contributed *n* − 1 folds. The primary per-fold metric is the **event window probability**: the maximum predicted probability within the three-day window preceding and including the test event. We defined detection as event window probability ≥ 0.5.

**Cross-season holdout (regional, blended, and hierarchical models).** We trained the aggregate models exclusively on 2024–2025 data and evaluated them on all 2025–2026 events. This one-season holdout tests cross-season generalization and complements the per-station LOO results. Because the hierarchical model produces calibrated low-magnitude probabilities, we compare all aggregate models with the threshold-free metrics AUC-ROC and average precision (AP); for the heavy class imbalance here, precision-recall summaries such as AP are more informative than ROC alone (Saito and Rehmsmeier, 2015).

**Event-level precision-recall.** For each test event, we took the maximum model probability within a three-day pre-event window as the event detection score. We drew non-overlapping, same-length windows from non-grace-zone periods at the same stations to serve as the negative class. This framing asks whether the model issued a timely warning before each event — a more operationally relevant question than per-day classification accuracy in a high-stakes forecasting context.

### 3.5 Operational Threshold Tuning

For each station we derived a recall-maximizing operational threshold by taking the minimum event window probability observed across all LOO folds. This threshold guarantees detection of every event the model has seen at that station, at the cost of some false alarms on non-event days. We then counted the number of individual non-event days in the 2025–2026 season where the model exceeded that station-specific threshold, excluding a ±4-day grace zone around each observed event. We expressed this as a false alarm rate: false alarm days divided by total non-event days. We defined a station as operationally viable when its false alarm rate fell at or below 10%, reflecting a practical tolerance of roughly one unnecessary alert per month in a 5-month winter season.

---

## 4. Results

### 4.1 Per-Station Classifier Performance (Temporal LOO)

We applied temporal LOO cross-validation to the 16 stations with two or more total events across both seasons, producing 25 folds (Table 3). With the reduced five-feature set and strong regularization, eight stations achieved a 100% detection rate, three achieved 50%, and five (all single-fold) detected no event at the 0.5 threshold.

**Table 3. Per-station temporal LOO results (reduced model; all events across both seasons, sorted by detection rate).**

| Station | Aspect | Total events | LOO folds | Mean event prob | Detection rate |
|---|---|---|---|---|---|
| 160942_res | E | 2 | 1 | 1.000 | **100%** |
| 272401_res | N | 2 | 1 | 0.921 | **100%** |
| 180343_res | S | 3 | 2 | 0.831 | **100%** |
| 268603_res | S | 4 | 3 | 0.816 | **100%** |
| 153203_res | S | 4 | 3 | 0.784 | **100%** |
| 164801_res | N | 3 | 2 | 0.771 | **100%** |
| 176204_res | W | 2 | 1 | 0.760 | **100%** |
| 268601_res | N | 2 | 1 | 0.700 | **100%** |
| 250224_res | W | 3 | 2 | 0.537 | 50% |
| 224102_res | E | 3 | 2 | 0.348 | 50% |
| 183642_res | S | 3 | 2 | 0.329 | 50% |
| 187363_res | S | 2 | 1 | 0.242 | 0% |
| 183741_res | N | 2 | 1 | 0.218 | 0% |
| 176522_res | E | 2 | 1 | 0.178 | 0% |
| 165202_res | E | 2 | 1 | 0.051 | 0% |
| 224101_res | N | 2 | 1 | 0.010 | 0% |

The strongest stations show genuine discrimination, not blanket over-prediction: 160942_res assigns probability 1.00 to its event window, and 272401_res 0.92, against a median non-event probability of 0.15 across all folds. The five zero-detection stations are all single-fold cases where the model trains on exactly one prior event — too little to anchor a reliable decision boundary.

**Performance as a function of training data size.** Aggregating LOO folds by the number of prior training events at the time of each test (Figure X) shows a strong, consistent positive trend in both mean event window probability and detection rate (Table 4).

**Table 4. LOO detection performance by number of training events.**

| Training events | Folds | Mean event prob | Detection rate |
|---|---|---|---|
| 1 | 16 | 0.477 | 56.2% |
| 2 | 7 | 0.779 | 85.7% |
| 3 | 2 | 0.719 | 100.0% |

Detection rate climbs from 56% with one prior event to 86% with two and 100% with three, while mean event window probability rises from 0.48 to 0.78. This learning curve is the paper's central empirical result: per-station performance improves steeply with observation density, and the current dataset — 1–4 events per station — sits squarely on the steep portion where each additional observation yields a large gain. The n = 3 level rests on only two folds and should be read as indicative; the n = 1 (16 folds) and n = 2 (7 folds) points are more robust.

### 4.2 Aggregate Model Performance and Event-Level Detection

We evaluated the four models on all 37 test-season events using the event-level framing: did the model assign elevated probability within the three days preceding each event (37 positive windows vs. 1,038 non-event windows; no-skill prevalence = 0.034)? Table 5 reports threshold-free metrics.

**Table 5. Event-level performance (3-day pre-event window, 2024–2025 train → 2025–2026 test).**

| Model | AUC-ROC | Average Precision | vs. No-Skill AP |
|---|---|---|---|
| **Hierarchical (Bayesian)** | **0.694** | **0.250** | **7.4×** |
| Blended (weighted) | 0.687 | 0.218 | 6.4× |
| Regional (pooled) | 0.687 | 0.216 | 6.4× |
| Per-station + fallback | 0.665 | 0.165 | 4.9× |
| No skill (baseline) | 0.500 | 0.034 | 1.0× |

The hierarchical model is the strongest performer, achieving an average precision 7.4× the no-skill baseline and dominating the high-precision region of the precision-recall curve (Figure Y) — the operationally important regime where a forecaster wants alarms to be trustworthy. Its advantage over the regional and blended models is modest in aggregate but consistent, and it comes with two structural benefits the others lack: it is a single coherent model rather than three, and it produces per-station uncertainty (Section 4.4). The per-station-plus-fallback configuration is weakest, confirming that single-event per-station models are too noisy to serve as the primary predictor and are better used only as one input to the pooled or hierarchical framework.

These numbers represent a large improvement over an initial nine-feature, weakly-regularized configuration (regional AP = 0.045, AUC-ROC = 0.610). Reducing the feature set to five physically-distinct variables and applying proper regularization — addressing the ~3.5-events-per-predictor over-parameterization — raised regional AP roughly fivefold, confirming that the original models were badly overfit rather than signal-limited.

### 4.3 Physical Plausibility of Learned Weights

The regional model's standardized coefficients (Table 6) summarize the feature–avalanche relationship learned across all stations. With the reduced feature set and CV-tuned regularization (C = 0.3), the coefficients are stable and physically interpretable — unlike the unstable weights of the original over-parameterized model.

**Table 6. Regional model standardized coefficients (reduced feature set, C = 0.3).**

| Feature | Coefficient | Physical interpretation |
|---|---|---|
| HS | +1.33 | Greater total snow depth → larger, valley-reaching avalanches |
| depth_lower_wl | −0.92 | Shallower weak-layer burial → more easily triggered → higher risk |
| TA_max | +0.27 | Warming increases modeled risk |
| HN24 | +0.26 | New-snow loading increases risk |
| sn38_min | +0.14 | Weak signal at the pooled level |

Total snow depth (HS) emerges as the dominant predictor (+1.33), consistent with the paper's framing that valley-floor avalanches require a deep enough snowpack to run the full distance from start zone to valley floor — the size signal Ron emphasized in matching observations to stations. The negative `depth_lower_wl` coefficient (−0.92) is physically coherent: a weak layer buried closer to the surface is more easily triggered and more likely to support a propagating slab. New-snow loading (HN24, +0.26) and warming (TA_max, +0.27) contribute as expected from loading and temperature-weakening mechanisms. The whole-profile stability index (sn38_min) carries little weight at the pooled level, suggesting that the simulated Sn38 index adds limited cross-station signal beyond the depth and loading variables — though it remains informative at individual well-sampled stations, where per-station coefficients vary widely.

### 4.4 Per-Station Uncertainty from the Hierarchical Model

Beyond its competitive accuracy, the hierarchical model's chief practical advantage is that it reports *how confident it is* at each station. Because every station's coefficients are drawn from the posterior, the spread of predicted probabilities across posterior draws is a direct, model-derived uncertainty estimate. We summarize each station's forecast-window prediction as a posterior mean probability and its standard deviation (Table 7 reports representative extremes).

**Table 7. Hierarchical model forecast-window prediction and uncertainty (representative stations).**

| Station | In training | Mean prob | Posterior std | Reading |
|---|---|---|---|---|
| 142301_res | no | 0.83 | 0.31 | High risk, but very low confidence (no local data) |
| 142303_res | no | 0.80 | 0.32 | High risk, very low confidence |
| 224103_res | no | 0.006 | 0.003 | Confidently low risk |
| 224102_res | yes | 0.011 | 0.009 | Low risk, well constrained |
| 187363_res | yes | 0.017 | 0.019 | Low risk, well constrained |

The contrast is the key result. For stations with no local training data (e.g. 142301_res), the model can still produce a prediction from the regional prior — but it correctly reports large uncertainty (std ≈ 0.31), flagging that a high mean probability there should not be trusted as much as the same value at a well-sampled station. This is exactly the signal an operational map needs to distinguish "high risk, trust it" from "high risk, but the model is guessing." It replaces the heuristic operational-threshold confidence tier (Section 4.5) with a principled, continuous measure that falls out of the model itself.

### 4.5 Operational Threshold Analysis

For the frequentist per-station models, we applied the recall-maximizing threshold to the 2025–2026 season and counted individual non-event days where the model exceeded that threshold (Figure Z; Table 8). The 149-day season provides 131–140 non-event days per station after removing ±4-day grace zones around observed events.

**Table 8. Operational threshold analysis — false alarm rate at recall-maximizing threshold.**
*(Reduced model. Season: 2025–2026. Non-event days exclude ±4-day grace zones around observed events.)*

| Station | Folds | Threshold | FA days | Non-event days | FA rate |
|---|---|---|---|---|---|
| 160942_res (E) | 1 | 1.000 | 0 | 140 | **0.0%** |
| 153203_res (S) | 3 | 0.654 | 2 | 140 | **1.4%** |
| 180343_res (S) | 2 | 0.802 | 3 | 138 | **2.2%** |
| 164801_res (N) | 2 | 0.550 | 14 | 140 | **10.0%** |
| 272401_res (N) | 1 | 0.921 | 16 | 140 | 11.4% |
| 176522_res (E) | 1 | 0.178 | 25 | 140 | 17.9% |
| 250224_res (W) | 2 | 0.436 | 31 | 140 | 22.1% |
| 187363_res (S) | 1 | 0.242 | 42 | 140 | 30.0% |
| 183741_res (N) | 1 | 0.218 | 50 | 140 | 35.7% |
| 165202_res (E) | 1 | 0.051 | 55 | 140 | 39.3% |
| 224102_res (E) | 2 | 0.004 | 62 | 131 | 47.3% |
| 224101_res (N) | 1 | 0.010 | 72 | 140 | 51.4% |
| 268603_res (S) | 3 | 0.689 | 90 | 139 | 64.7% |

**Operationally ready (false alarm rate ≤ 10%).** Four stations meet the viability criterion — double the count under the original over-parameterized model. 160942_res achieves zero false alarm days (threshold 1.0, fires only at near-certainty); 153203_res (1.4%) and 180343_res (2.2%) flag only a handful of days all season; 164801_res sits at the 10% boundary (14 of 140 days). Notably, 180343_res — which the original model flagged on 70% of non-event days as an "always-on" artifact — now produces a genuinely discriminating 2.2% false alarm rate, a direct consequence of the feature reduction and regularization.

**Marginal (10–25%).** Three stations (272401, 176522, 250224) flag between 11% and 22% of non-event days — usable with expert review of each alert rather than automated dispatch.

**Not operationally ready (> 25%).** Six stations remain above 25%. The clearest case is 268603_res: it achieves 100% LOO detection yet flags 65% of non-event days, because its threshold (0.689) sits below a high seasonal background — it detects events by predicting high risk much of the time rather than by sharply discriminating event days. This is the residual "always-high" pattern, now confined to a single station rather than several.

The operational analysis quantifies the data investment each station needs. At 164801_res (boundary-viable on two events), one further event would likely lift the threshold and reduce false alarms below 10%. At stations with thresholds below 0.05 (224101, 224102, 165202), the model has too little signal to anchor a useful threshold, and several additional events spanning diverse conditions are needed before deployment.

### 4.6 Summary

Reducing the feature set to five physically-distinct variables and applying proper regularization transformed the framework's measured skill: regional event-level average precision rose roughly fivefold (0.045 → 0.216) and the per-station learning curve steepened, with LOO detection rate climbing 56% → 86% → 100% across one, two, and three training events. The hierarchical Bayesian model is the strongest single configuration (event-level AP = 0.250, 7.4× the no-skill baseline), unifies the per-station and regional approaches in one principled framework, and — uniquely — reports per-station prediction uncertainty that an operational display can use directly. Learned feature weights are stable and physically coherent, led by total snow depth (the valley-reach signal) and shallow weak-layer burial.

Four stations now meet a ≤10% false-alarm operational criterion, up from two, and the "always-on" overfitting artifact that previously affected several stations is largely eliminated. The results confirm the framework as a decision-support tool whose performance scales steeply and reliably with observation density: moving from one to three events per station roughly doubles per-station detection, a gain achievable within a single additional observation season and the clearest path to operational improvement.

---

## References

Abril-Pla, O., Andreani, V., Carroll, C., Dong, L., Fonnesbeck, C. J., Kochurov, M., Kumar, R., Lao, J., Luhmann, C. C., Martin, O. A., Osthege, M., Vieira, R., Wiecki, T., and Zinkov, R.: PyMC: a modern, and comprehensive probabilistic programming framework in Python, *PeerJ Computer Science*, 9, e1516, https://doi.org/10.7717/peerj-cs.1516, 2023.

Betancourt, M. and Girolami, M.: Hamiltonian Monte Carlo for hierarchical models, in: *Current Trends in Bayesian Methodology with Applications*, Chapman and Hall/CRC, 79–101, 2015.

Elkan, C. and Noto, K.: Learning classifiers from only positive and unlabeled data, in: *Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 213–220, https://doi.org/10.1145/1401890.1401920, 2008.

Gelman, A. and Hill, J.: *Data Analysis Using Regression and Multilevel/Hierarchical Models*, Cambridge University Press, 2007.

Hoffman, M. D. and Gelman, A.: The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo, *Journal of Machine Learning Research*, 15, 1593–1623, 2014.

Lehning, M., Bartelt, P., Brown, B., Fierz, C., and Satyawali, P.: A physical SNOWPACK model for the Swiss avalanche warning services. Part II: snow microstructure, *Cold Regions Science and Technology*, 35(3), 147–167, 2002.

Morin, S., Horton, S., Techel, F., Bavay, M., Coléou, C., Fierz, C., Gobiet, A., Hagenmuller, P., Lafaysse, M., Ližar, M., Mitterer, C., Monti, F., Müller, K., Olefs, M., Snook, J. S., van Herwijnen, A., and Vionnet, V.: Application of physical snowpack models in support of operational avalanche hazard forecasting: A status report on current implementations and prospects for the future, *Cold Regions Science and Technology*, 170, 102190, https://doi.org/10.1016/j.coldregions.2019.102910, 2020.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, É.: Scikit-learn: machine learning in Python, *Journal of Machine Learning Research*, 12, 2825–2830, 2011.

Saito, T. and Rehmsmeier, M.: The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets, *PLOS ONE*, 10(3), e0118432, https://doi.org/10.1371/journal.pone.0118432, 2015.

Skamarock, W. C. and Klemp, J. B.: A time-split nonhydrostatic atmospheric model for research and NWP applications, *Journal of Computational Physics*, 227, 3465–3485, 2008.

Snook, J. S., Cooperstein, M., and Greene, E.: Snowpack modeling efforts at the Colorado Avalanche Information Center, *Proceedings of the International Snow Science Workshop*, Bend, OR, USA, 2022.

Snook, J. S.: Weather forecast model grid spacing — is smaller better?, *Proceedings of the International Snow Science Workshop*, Breckenridge, CO, USA, 2016.
