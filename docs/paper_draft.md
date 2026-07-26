# Forecasting Avalanches in Data-Sparse Central Asia using Snowpack and Weather Modeling with Avalanche Observations

Itai Sheleg¹, Doug Chabot², John Snook³, Jaime Peters¹, Walter Steinkogler⁴, Ron Simenhois³

¹ Lake County High School, Leadville, Colorado  
² Latok, LLC, Bozeman, MT  
³ Colorado Avalanche Information Center, CO, USA  
⁴ Wyssen Avalanche Control AG, Switzerland

---

## Abstract

Avalanches in Central Asia, particularly in Afghanistan, Tajikistan, and Pakistan, are a persistent hazard causing hundreds of fatalities in severe winters and routinely destroying homes, infrastructure, and livestock essential for survival. To address this risk, the Aga Khan Agency for Habitat (AKAH) established a remote avalanche forecasting program in 2015. A primary constraint on forecasting capability is the limited availability of snowpack, weather, and avalanche observations, driven by geographic remoteness and government restrictions on data sharing.

We present a proof-of-concept framework that fuses heterogeneous, publicly available data sources to support avalanche forecasting in data-sparse regions. We combine NWP-forced SNOWPACK simulations with avalanche observations extracted from AKAH structured logs and social media posts, treating informal reports as weakly labeled observations. From 30 virtual SNOWPACK stations across the Darvoz region of Tajikistan and 69 valley-floor avalanche observations across two winter seasons, we derived five daily features — total snow depth, 24-hour new snow, maximum air temperature, whole-profile minimum stability, and weak-layer burial depth — and fit a hierarchical Bayesian logistic regression with station-varying intercepts and shared slopes. Training on 2024–2025 and evaluating on 2025–2026, the model achieves event-level average precision 7× the no-skill baseline (AUC-ROC 0.715) and reports per-station forecast uncertainty directly from the posterior. Learned feature weights are physically coherent, with total snow depth and shallow weak-layer burial as the dominant predictors. The framework is best interpreted as a decision-support tool: it provides probabilistic guidance where conventional data streams are absent, while expert terrain knowledge and operational judgment remain central to forecasting.

---

## 1. Introduction

Avalanches are a severe and recurring hazard across the Hindu Kush, Pamir, Karakoram, and western Himalaya. A single 2012 avalanche cycle in Afghanistan, Pakistan, and Tajikistan killed more than 100 people (Chabot and Kaba, 2016), motivating the Aga Khan Development Network and Aga Khan Agency for Habitat (AKAH) to build a community-based avalanche warning system around village-run weather monitoring posts, coordinated evacuation planning, and trained search-and-rescue teams. When a comparably severe cycle struck the same region in February 2023, 38 people died (Chabot, 2024), a fraction of the 2012 toll attributed to the network built in the intervening years. Providing operational avalanche forecasts is a central component of these programs, yet the observational data that underpin modern forecasting are largely unavailable across the region.

Modern avalanche forecasting relies on dense observational networks, including automated weather stations, snowpack observations, and systematic avalanche records. Across much of Central Asia, this infrastructure is largely absent, and the challenge has three components. Field snowpack observations are essentially absent. Automated weather stations are rare at avalanche-relevant elevations, and where they exist, cost constraints and government restrictions on cross-border data sharing often prevent their use in regional forecasting (Chabot, 2024). Systematic records of avalanche occurrence needed to establish historical baselines and evaluate model performance are similarly lacking. Each limitation would significantly constrain an operational forecasting program; together, they render conventional forecasting approaches impractical.

We combine three globally available data sources: numerical weather prediction (NWP) model output, satellite imagery, and avalanche events identified from social media and local news reports to test whether these limitations can be substantially mitigated. Systematic records of avalanche events in this region exist almost entirely outside peer-reviewed literature, in technical reports, news, and social media (Acharya et al., 2023), making these sources not merely supplementary but the primary available record. Rather than replacing the forecaster, this framework provides probabilistic decision support by integrating information that would otherwise be inaccessible. Expert terrain knowledge and operational judgment remain central to the forecasting process. This paper focuses on the NWP-driven snowpack modeling and observation components of the framework; satellite-derived inputs are part of the broader system architecture but fall outside the present evaluation.

This paper presents the system architecture and a pilot deployment in Afghanistan, Pakistan, and Tajikistan, using 1155 virtual SNOWPACK stations and 69 avalanche observations compiled from AKAH's structured observation logs and social media reports. We evaluate whether the model's probabilistic hazard estimates are physically consistent with known regional avalanche cycles and discuss the steps required to transition the framework toward reliable operational forecasting in data-sparse regions worldwide.


---

## 2. Data

### 2.1 NWP Forcing and SNOWPACK Simulations

In 2022 we implemented the open-source Weather Research and Forecasting (WRF) model (Skamarock and Klemp, 2008) to provide weather forecasts over Central Asia. We derive WRF initial and forecast lateral boundary conditions from the US National Weather Service Global Forecast System (GFS), and configure the model with a nested grid: a 12-km outer domain and a 4-km inner nest. WRF produces 7-day forecasts updated twice daily; post-processing generates standard meteorological parameter imagery and point forecasts available on a regional website. WRF forecasts proved valuable during the first winter season.

Following the approach developed at the Colorado Avalanche Information Center (Snook et al., 2022; Snook, 2016), we implemented the SLF SNOWPACK model (Lehning et al., 2002; Morin et al., 2020) for the following season. We configure WRF to output meteorological forecasts as direct input into SNOWPACK, which simulates per-layer snowpack evolution at each grid point. The system generates snowpack profiles at all WRF 4-km grid points above 3,500 m within the defined forecast region, updates them daily, and makes them accessible through the regional website, with the ability to display full seasonal profiles on demand.

For this study, we extracted SNOWPACK output at a network of 30 virtual stations across the Darvoz region of Tajikistan and adjacent Pakistan, spanning elevations from approximately 2,100 to 4,900 m. We assigned each station a specific aspect (N, E, S, W, or flat) and elevation, producing a spatially distributed representation of snowpack evolution tuned to slope exposure. We ran simulations for two consecutive winter seasons: 2024–2025 (training) and 2025–2026 (evaluation). Sub-daily `.smet` files provided meteorological time series, and `.pro` files provided per-layer snowpack stratigraphy at each timestep. Table 1 lists the five variables the model uses.

*[Figure 1: Study area map with the January 23, 2026 event cycle selected. Triangles mark the valley-floor avalanche reports for that date; circles are SNOWPACK virtual stations with rings colored by modeled probability in the January 22–25 forecast window. The dark-red cluster (≥80%) in the Chitral region coincides with the reported events, while stations without matching observations are shown faded. Clicking a station opens its full seasonal stability chart. Generated: output/fig1_map.png.]*

**Table 1. The five model features, their SNOWPACK source, and physical role.**

| Feature | Source | Physical role |
|---|---|---|
| HS | .smet | Total snow depth — size / valley-reach potential |
| HN24 | .smet | 24-hour new snow — primary loading trigger |
| TA_max | .smet | Daily max air temperature — wet vs dry mechanism |
| sn38_min | .pro | Whole-profile minimum stability index — weakest-layer strength |
| depth_lower_wl | .pro | Burial depth of weakest lower-zone layer |

We had no in-situ snowpack observations to validate simulated stratigraphy; the Sn38 index and layer structure are model-derived proxies only.

### 2.2 Avalanche Observations

We compiled avalanche observations from two sources. The Aga Khan Agency for Habitat (AKAH) maintains a structured observation log covering Pakistan and Tajikistan across multiple seasons. We also extracted a supplementary set of events from social media posts shared by local residents and relief workers. Both sources report valley-floor runout locations rather than start zones, and each record includes date, approximate location, slope angle, aspect, and a qualitative size estimate.

The combined dataset contains 69 observations: 26 from the 2024–2025 season (training) and 43 from 2025–2026 (evaluation). We treat observations as weakly labeled events. A report confirms that an avalanche reached the valley floor, but it does not verify the trigger mechanism, release zone, or snowpack state. More fundamentally, the absence of a report does not confirm the absence of an avalanche: our records are an opportunistic, almost certainly incomplete sample of activity, so an unknown number of real events go unrecorded and are treated as quiet days. In machine-learning terms this is a positive-unlabeled setting (Elkan and Noto, 2008), and its practical consequence is favorable to our claims: any skill we measure is a conservative lower bound, because some of the "quiet" days the model gets penalized for flagging may in fact have had unreported avalanches. The supplementary technical report discusses this framing and why we apply no formal correction for it.

### 2.3 Observation-to-Station Matching

Valley-floor reports provide approximate runout locations rather than start zones. A practitioner with regional terrain knowledge identified the likely avalanche start zone for each observation based on valley geometry, known paths, and observation context. We then matched each start zone to the nearest SNOWPACK station sharing the same predominant aspect (N, E, S, or W), ensuring the simulation reflects the relevant solar radiation and wind loading regime.

This hybrid human-machine matching step embeds domain knowledge where nearest-distance matching alone would systematically fail: two stations equidistant from a reported runout but facing opposite aspects carry fundamentally different snowpack histories. The matching introduces practitioner subjectivity and is not automatically reproducible, but we consider this a feature rather than a gap — it is the mechanism through which terrain expertise enters the framework.

---

## 3. Methods

### 3.1 Feature Engineering

We aggregated sub-daily SNOWPACK output to daily resolution for model training and prediction. From `.smet` files we computed daily maxima for HS, HN24, and TA_max. From `.pro` layer data we classified layers at each timestep into an upper zone (burial depth < 40% of total HS, targeting near-surface storm slabs) and a lower zone (burial depth ≥ 40% of HS, targeting deep persistent weak layers), and computed the minimum Sn38 within each zone (`sn38_upper_min`, `sn38_lower_min`) plus the burial depth of the weakest lower-zone layer. To avoid the collinearity between the two zone minima, we collapsed them into a single whole-profile minimum, sn38_min = min(sn38_upper_min, sn38_lower_min) — the only Sn38-derived feature retained in the final model. When snowpack depth was insufficient to resolve both zones — common in early-season or thin-snowpack conditions — we forward-filled zone features up to seven days, then imputed any remaining missing values with the training-set mean at prediction time.

With only 35 positive-labeled days (event date plus the preceding day, per the labeling scheme in Section 3.2) across the 13 stations with usable 2024–2025 SNOWPACK data, a larger feature set would be severely over-parameterized. We therefore reduced to the five physically distinct, minimally collinear variables of Table 1: total snow depth (HS), 24-hour new snow (HN24), maximum air temperature (TA_max), the whole-profile minimum stability index (sn38_min), and the burial depth of the weakest lower-zone layer (depth_lower_wl). The feature selection process and the ablation against the fuller candidate set are documented in the supplementary technical report.

### 3.2 Labeling Under Weak Supervision

We labeled event days positive (y = 1) for both the reported date and the preceding day, accounting for the one-day reporting lag inherent in social media and field log sources. All other days were labeled negative (y = 0), producing a positive day rate of approximately 1.1% in the 2024–2025 training corpus. Where zone-derived features contained no valid values on positive training days — most commonly at stations with thin or absent early-season snowpacks — the model fell back to a meteorological-only feature set (HS, HN24, TA_max), sacrificing snowpack structural information but preserving trainability.

### 3.3 Hierarchical Bayesian Model

We fit a single hierarchical Bayesian model — a multilevel logistic regression with partial pooling (Gelman and Hill, 2007) — that treats the 13 stations with at least one recorded event as related members of one region rather than as isolated problems. Each station gets its own baseline avalanche rate, while the influence of the five features is shared across all stations. How much each station's baseline can differ from the regional average is itself learned from the data: a station with several observed events keeps its own signal, while a station with a single event leans heavily on the regional pattern. This is exactly the behavior a forecaster wants from a sparse network — local adaptation where the data support it, regional knowledge where they do not.

We excluded stations with no recorded events from training. Reporting in this region is opportunistic, so a station with no events more plausibly reflects no observer coverage than no avalanches; at prediction time these stations are served by the regional pattern. Because the model is Bayesian, every prediction comes with an uncertainty estimate at no extra cost — the basis for the confidence reporting in Section 4.4. We estimated the model with the PyMC probabilistic programming framework (Abril-Pla et al., 2023); the full model specification, prior choices, structure-selection evidence, and sampling diagnostics are documented in the supplementary technical report.

### 3.4 Evaluation

**Temporal leave-one-out cross-validation (per-station).** For each station, we sorted all observed events chronologically. In fold *i*, we trained on all events preceding event *i* and tested on event *i*, ensuring no future information informs training. The primary per-fold metric is the event window probability: the maximum predicted probability within the three-day window preceding and including the test event. We defined detection as event window probability ≥ 0.5.

**Cross-season holdout.** We trained the model exclusively on 2024–2025 data and evaluated on all 2025–2026 events. This one-season holdout tests cross-season generalization. We report AUC-ROC and average precision (AP); for severe class imbalance, precision-recall summaries are more informative than ROC alone (Saito and Rehmsmeier, 2015).

**Event-level framing.** For each test event we took the maximum model probability within a three-day pre-event window as the event detection score, and drew non-overlapping, same-length windows from non-grace-zone periods at the same stations as the negative class. This asks whether the model issued a timely warning before each event — the operationally relevant question in a high-stakes forecasting context.

---

## 4. Results

### 4.1 Per-Station Performance and Learning Curve

We evaluated each station by asking: given what the model knew from past events at that location, did it correctly flag the next event before it happened? Across the 16 stations with enough observations to test, eight correctly flagged every event, three caught roughly half, and five missed entirely — all five in the last group had only a single prior event to learn from, which is too little to build a reliable signal.

The clearest finding is how steeply performance improves as the model accumulates local observations (Table 2 and Figure 2). With just one prior event at a station, the model detected 56% of the next events. With two, detection climbed to 86%. With three, it reached 100%. Each additional observation delivers a large, consistent gain — the dataset sits on the steep part of the learning curve where more data matters most.

*[Figure 2: Learning curve — detection rate and mean warning probability as a function of the number of prior observed events at a station. Generated from output/loo_performance_by_events.png.]*

**Table 2. Detection rate improves steeply with each additional observed event.**

| Prior events at station | Detection rate | Mean warning probability |
|---|---|---|
| 1 | 56% | 0.48 |
| 2 | 86% | 0.78 |
| 3 | 100% | 0.72 |

### 4.2 Cross-Season Performance

Trained on 2024–2025 data and tested against all 37 observed events in 2025–2026, the model issued warnings at the right time far more often than chance: it ranked dangerous days above safe ones with an AUC-ROC of 0.715 (where 1.0 is perfect and 0.5 is no skill), and its warning precision was seven times better than random guessing. The hierarchical model also outperformed every simpler configuration we tested — per-station, regionally pooled, and blended classifiers. Its key advantage is that it shares information across all stations: a location with few observations benefits from patterns learned at better-observed stations nearby, while still adapting to each location's individual history. The full model comparison and statistical detail are available in the supplementary technical report.

### 4.3 What the Model Learned

The model's learned feature weights are physically sensible (Figure 3). Total snow depth is by far the strongest predictor of a valley-floor avalanche: a deep snowpack provides the energy needed to reach the valley. Weak-layer burial depth is the second-strongest signal, acting in the opposite direction — the deeper a weak layer is buried, the harder it is to trigger, so shallow weak layers raise the modeled risk. New snowfall and warm temperatures both increase risk, with warming carrying roughly equal weight to new snow — consistent with the importance of rain-on-snow events in this climate. The Sn38 structural stability index adds modest additional signal. These patterns hold across the region and match what experienced forecasters expect from terrain and climate in Central Asia.

*[Figure 3: Feature influence on predicted avalanche risk — standardized model coefficients as a horizontal bar chart. Red bars increase risk, the blue bar decreases risk, longer bars mean stronger influence. Generated: output/feature_importance.png.]*

### 4.4 Knowing What the Model Doesn't Know

A key advantage of the hierarchical approach is that it reports not just a probability, but how confident it is in that probability. At stations with several observed events the model is tightly constrained; at stations with no local history it borrows from the regional average and correctly flags that the prediction is uncertain. This matters operationally: a high-risk probability at a well-observed station is a strong signal, while the same number at an unobserved station should prompt extra caution and expert judgment rather than automated action.

### 4.5 Operational Readiness

Testing each station against a standard of no more than roughly one false alarm per month (≤10% of non-event days), four stations are operationally ready today. One station produced zero false alarms across the entire 2025–2026 season; two others flagged fewer than three days all winter. Three more stations are in a marginal range — useful with forecaster review of each alert. The remaining stations are not yet ready, primarily because they have too few observed events to anchor a reliable threshold. Each additional season of observations will move stations from the not-ready to the marginal or ready tier.

### 4.6 Case Study: Model Output for a Known Event

*[Figure 4: Per-station stability chart for station 160942_res (east-facing, best-performing station) showing the 2025–2026 season: avalanche probability with operational threshold bands (top), snow stratigraphy with weak layers colored by Sn38 (middle), and 24-hour new snow with air temperature (bottom). The model assigns probability ~1.00 in the highlighted three-day window around the observed January 23 event, with markedly lower probabilities on surrounding non-event days. Generated: output/fig4_stability_160942.png.]*

Figure 4 illustrates the model's output for the best-performing station during a known event window. The probability spike is narrow, peaking immediately before the reported avalanche and returning to background levels within days. This is the pattern a forecaster would act on: a sharp, time-specific signal rather than a persistent high-risk background. The stability chart is the operational display that field staff and forecasters interact with directly.

---

## 5. Conclusions

A framework combining NWP-forced SNOWPACK simulations with opportunistically collected avalanche observations produces physically consistent, statistically skillful avalanche probability guidance in a region with near-zero conventional data. Trained on one season and tested on the next, the hierarchical model ranked dangerous days well above safe ones and issued warnings seven times more precisely than chance, and four stations already meet an operational false-alarm standard.

Three findings shape the path forward. First, performance scales steeply and predictably with observation density: moving from one to three recorded events at a station roughly doubles detection, so each additional season of observations delivers a large, measurable gain — the clearest and cheapest route to operational improvement. Second, what the model learned matches avalanche physics — deep snowpacks, shallow weak layers, fresh loading, and warming all push risk in the expected direction — which supports the framework's coherence even without direct snowpack validation. Third, the model reports its own confidence at every station, letting forecasters distinguish predictions grounded in local history from those extrapolated off the regional pattern.

Expert terrain knowledge remains embedded at the observation-matching step, and the system is designed to inform expert judgment rather than replace it. We view this framework as a decision-support layer whose reach grows with every observation added, and as a template for avalanche forecasting in data-sparse mountain regions worldwide.

---

## References

Abril-Pla, O., et al.: PyMC: a modern, and comprehensive probabilistic programming framework in Python, *PeerJ Computer Science*, 9, e1516, 2023.

Acharya, A., et al.: *[2023 reference — details TBD; cited for avalanche records in Central Asia existing mainly outside peer-reviewed literature.]*

Chabot, D. and Kaba, D.: *[2016 reference — details TBD.]*

Chabot, D.: *[2024 reference — details TBD.]*

Elkan, C. and Noto, K.: Learning classifiers from only positive and unlabeled data, *Proceedings SIGKDD*, 213–220, 2008.

Gelman, A. and Hill, J.: *Data Analysis Using Regression and Multilevel/Hierarchical Models*, Cambridge University Press, 2007.

Lehning, M., et al.: A physical SNOWPACK model for the Swiss avalanche warning services. Part II, *Cold Regions Science and Technology*, 35(3), 147–167, 2002.

Morin, S., et al.: Application of physical snowpack models in support of operational avalanche hazard forecasting, *Cold Regions Science and Technology*, 170, 102190, 2020.

Saito, T. and Rehmsmeier, M.: The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets, *PLOS ONE*, 10(3), e0118432, 2015.

Skamarock, W. C. and Klemp, J. B.: A time-split nonhydrostatic atmospheric model, *Journal of Computational Physics*, 227, 3465–3485, 2008.

Snook, J. S., Cooperstein, M., and Greene, E.: Snowpack modeling efforts at the Colorado Avalanche Information Center, *ISSW*, Bend, OR, 2022.

Snook, J. S.: Weather forecast model grid spacing — is smaller better?, *ISSW*, Breckenridge, CO, 2016.
