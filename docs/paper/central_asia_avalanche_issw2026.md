# Forecasting Avalanches in Data-Sparse Central Asia Using Snowpack and Weather Modeling with Avalanche Observations

**Itai Sheleg**¹, **Doug Chabot**², **John Snook**³, **Jaime Peters**¹, **Walter Steinkogler**⁴, **Ron Simenhois**³

¹ Lake County High School, Leadville, CO, USA
² Latok, LLC, Bozeman, MT, USA
³ Colorado Avalanche Information Center, CO, USA
⁴ Wyssen Avalanche Control AG, Switzerland

Correspondence: Itai Sheleg, Lake County High School, Leadville, CO, USA — itai.sheleg@gmail.com

**Keywords:** Avalanche Forecasting, SNOWPACK, Bayesian Modeling, Central Asia

---

## Abstract

Avalanches in Central Asia, particularly in Afghanistan, Tajikistan, and Pakistan, are a persistent hazard, causing hundreds of fatalities in severe winters and routinely destroying homes, infrastructure, and livestock essential for survival. To address this risk, in 2015, Aga Khan Agency for Habitat established a remote avalanche forecasting program supported by external expertise.

Following its implementation, it became clear that a primary constraint on forecasting capability is the limited availability of snowpack, weather, and avalanche observations, driven by both geographic remoteness and government restrictions on data sharing.

We address this gap by developing a proof-of-concept framework that fuses heterogeneous, publicly available data sources to support avalanche forecasting in data-sparse regions. Specifically, we combine numerical weather prediction (NWP)-forced SNOWPACK simulations (4 km grid spacing) with opportunistically extracted avalanche observations from social media posts shared by local residents. This approach treats informal reports as weakly labeled observations, enabling the reconstruction of avalanche occurrence in the absence of systematic records.

We identified 30 valley-floor avalanche events from social media and reconstructed the associated meteorological and snowpack conditions using nearby SNOWPACK outputs. Analyzed variables include precipitation totals and phase, air temperature, snow depth, weak-layer depth, and the natural stability index (*Sn38*). To distinguish avalanche from non-avalanche days, we trained per-station logistic regression classifiers on daily simulation-derived features.

Across 23 modeled stations, 24-hour new snow and maximum air temperature (*TA*max) emerge as the most consistent predictors. Positive *TA*max coefficients indicate that rain-on-snow events are at least as important as new-snow loading in driving avalanche activity.

This work incorporates a novel data-fusion and weak-supervision framework for avalanche forecasting, integrating NWP-driven snowpack modeling with opportunistic, publicly sourced observations. However, this framework is constrained by a limited event dataset and a model chain that lacks rigorous, component-wise validation, introducing uncertainty in both inputs and predictions. As such, the framework is best interpreted as a decision-support tool rather than a fully autonomous forecasting system. Coupled with expert validation and interpretation, it provides a scalable pathway for developing operational guidance in regions where conventional data streams are sparse or unavailable.

---

## 1. Introduction

Avalanches are a severe and recurring hazard across the Hindu Kush, Pamir, Karakoram, and western Himalaya. A single 2012 avalanche cycle in Afghanistan, Pakistan, and Tajikistan killed more than 100 people (Chabot & Kaba, 2016), motivating the Aga Khan Development Network and Aga Khan Agency for Habitat (AKAH) to build a community-based avalanche warning system around village-run weather monitoring posts, coordinated evacuation planning, and trained search-and-rescue teams. When a comparably severe cycle struck the same region in February 2023, 38 people died (Chabot, 2024), a fraction of the 2012 toll attributed to the network built in the intervening years. Providing operational avalanche forecasts is a central component of these programs, yet the observational data that underpin modern forecasting are largely unavailable across the region.

Modern avalanche forecasting relies on dense observational networks, including automated weather stations, snowpack observations, and systematic avalanche records. Across much of Central Asia, this infrastructure is largely absent, and the challenge has three components. Field snowpack observations are essentially absent. Automated weather stations are rare at avalanche-relevant elevations, and where they exist, cost constraints and government restrictions on cross-border data sharing often prevent their use in regional forecasting (Chabot, 2024). Systematic records of avalanche occurrence needed to establish historical baselines and evaluate model performance are similarly lacking. Each limitation would significantly constrain an operational forecasting program; together, they render conventional forecasting approaches impractical.

We combine three globally available data sources: numerical weather prediction (NWP) model output, satellite imagery, and avalanche events identified from social media and local news reports to test whether these limitations can be substantially mitigated. Systematic records of avalanche events in this region exist almost entirely outside peer-reviewed literature, in technical reports, news, and social media (Acharya et al., 2023), making these sources not merely supplementary but the primary available record. Rather than replacing the forecaster, this framework provides probabilistic decision support by integrating information that would otherwise be inaccessible. Expert terrain knowledge and operational judgment remain central to the forecasting process. This paper focuses on the NWP-driven snowpack modeling and observation components of the framework; satellite-derived inputs are part of the broader system architecture but fall outside the present evaluation.

This paper presents the system architecture and a pilot deployment in Afghanistan, Pakistan, and Tajikistan, using 1155 virtual SNOWPACK stations and 69 avalanche observations compiled from AKAH's structured observation logs and social media reports. We evaluate whether the model's probabilistic hazard estimates are physically consistent with known regional avalanche cycles and discuss the steps required to transition the framework toward reliable operational forecasting in data-sparse regions worldwide.

## 2. Data

### 2.1 NWP Forcing and SNOWPACK Simulations

In 2022, we implemented the open-source Weather Research and Forecasting (WRF) model (Skamarock & Klemp, 2008) to provide winter weather forecasts over Central Asia. We derive WRF initial and forecast lateral boundary conditions from the US National Weather Service Global Forecast System (GFS), and configure the model with a nested grid: a 12-km outer domain and a 4-km inner nest. WRF produces 7-day forecasts updated twice daily; post-processing generates standard meteorological parameter imagery and point forecasts available on a regional website. WRF forecasts proved valuable during the first winter season. Following the approach developed at the Colorado Avalanche Information Center (Snook et al., 2022; Snook, 2016), we implemented the SLF SNOWPACK model (Lehning et al., 2002; Morin et al., 2020) for the following season. We configure WRF to output meteorological forecasts as direct input into SNOWPACK, which simulates per-layer snowpack evolution at each grid point. The system generates snowpack profiles at all WRF 4-km grid points above 3,500 m within the defined forecast region, updates them daily, and makes them accessible through the regional website, with the ability to display full seasonal profiles on demand.

For this study, we extracted SNOWPACK output at a network of 30 virtual stations across the Darvoz region of Tajikistan and northwest Pakistan, spanning elevations from approximately 2,100 to 4,900 m. We assigned each station a specific aspect (N, E, S, W, or flat) and elevation, producing a spatially distributed representation of snowpack evolution tuned to slope exposure. We ran simulations for two consecutive winter seasons: 2024–2025 (training) and 2025–2026 (evaluation). Sub-daily `.smet` files provided meteorological time series, and `.pro` files provided per-layer snowpack stratigraphy at each timestep. Table 1 lists the five variables the model uses. Figure 1 shows the resulting station network and a representative forecast window from the 2025–2026 season.

**Table 1.** The five model features, their SNOWPACK source, and physical role.

| Feature | Source | Physical role |
|---|---|---|
| HS | SNOWPACK | Total snow depth — size / valley-reach potential |
| HN24 | NWP | 24-hour new snow — primary loading trigger |
| TA_max | NWP | Daily max air temperature — wet vs dry mechanism |
| Sn38_min | SNOWPACK | Whole-profile minimum stability index — weakest-layer strength |
| depth_lower_wl | SNOWPACK | Weak-layer depth — proximity to ground / basal layer |

We had no in-situ snowpack observations to validate simulated stratigraphy; the *Sn38* index and layer structure are model-derived proxies only.

![Study area map with the January 23, 2026 event cycle selected. Triangles mark the valley-floor avalanche reports for that date; circles are SNOWPACK virtual stations with rings colored by modeled probability in the January 22–25 forecast window. The dark-red cluster (≥80%) coincides with the reported events, while stations without matching observations are shown faded.](./figures/fig1_map.png)

**Figure 1.** Study area map with the January 23, 2026 event cycle selected. Triangles mark the valley-floor avalanche reports for that date; circles are SNOWPACK virtual stations with rings colored by modeled probability in the January 22–25 forecast window. The dark-red cluster (≥80%) coincides with the reported events, while stations without matching observations are shown faded.

### 2.2 Avalanche Observations

Avalanche observations are culled by AKAH and then forwarded to the avalanche forecaster. These observations are almost exclusively harvested from locals' Facebook pages in Afghanistan and Pakistan, and Instagram in Tajikistan. Most contain valuable videos and pictures showing the extent of the avalanches. These posts report valley-floor runout locations rather than start zones, and each record includes the date, approximate location, and estimates of slope angle, aspect, and qualitative size. AKAH provided the initial notification of avalanche events, but all observational data used in this study were independently derived from publicly available social media posts.

The combined dataset contains 69 observations: 26 from the 2024–2025 season (training) and 43 from 2025–2026 (evaluation). We treat observations as weakly labeled events. A report confirms that an avalanche reached the valley floor, but it does not verify the trigger mechanism, release zone, or snowpack state. More fundamentally, the absence of a report does not confirm the absence of an avalanche: our records are an opportunistic, almost certainly incomplete sample of activity, so an unknown number of real events go unrecorded and are treated as quiet days. In machine-learning terms, this is a positive-unlabeled setting (Elkan & Noto, 2008), and its practical consequence is favorable to our claims: any skill we measure is a conservative lower bound, because some of the "quiet" days the model gets penalized for flagging may in fact have had unreported avalanches. The supplementary technical report ([Sheleg et al., 2026](https://github.com/ronimos/caw-avalanche-mapping/blob/main/docs/methods_and_results.md)) discusses this framing and why we apply no formal correction for it.

### 2.3 Observation-to-Station Matching

Valley-floor reports provide approximate runout locations rather than start zones. A practitioner with regional terrain knowledge identified the likely avalanche start zone for each observation based on valley geometry, known paths, and observation context. We then matched each start zone to the nearest SNOWPACK station sharing the same predominant aspect (N, E, S, or W), ensuring the simulation reflects the relevant solar radiation and wind loading regime. This hybrid human-machine matching step embeds domain knowledge where nearest-distance matching alone would systematically fail: two stations equidistant from a reported runout but facing opposite aspects can carry fundamentally different snowpack histories. The matching introduces practitioner subjectivity and is not automatically reproducible, but we consider this a feature rather than a gap; it is the mechanism through which terrain expertise enters the framework.

## 3. Methods

### 3.1 Feature Engineering

We aggregated sub-daily SNOWPACK output to daily resolution for model training and prediction. From the NWP, we computed daily maxima for HS, HN24, and TA_max. From the SNOWPACK simulation, we classified layers at each timestep into an upper zone (burial depth < 40% of total HS, targeting near-surface storm slabs) and a lower zone (burial depth ≥ 40% of HS, targeting deep persistent weak layers), and computed the minimum Sn38 within each zone plus the burial depth of the weakest lower-zone layer.

With only 35 positive-labeled days (event date plus the preceding day, per the labeling scheme) across the 13 stations with usable 2024–2025 SNOWPACK data, a larger feature set would be severely over-parameterized. We therefore reduced to the five physically distinct, minimally collinear variables of Table 1: total snow depth (HS), 24-hour new snow (HN24), maximum air temperature (TA_max), the whole-profile minimum stability index (Sn38_min), and the burial depth of the weakest lower-zone layer (depth_lower_wl).

### 3.2 Labeling Under Weak Supervision

We labeled event days positive (y = 1) for both the reported date and the preceding day, accounting for the one-day reporting lag inherent in social media and field log sources. All other days were labeled negative (y = 0), producing a positive day rate of approximately 0.7% in the 2024–2025 training corpus. Where zone-derived features contained no valid values on positive training days (most commonly at stations with thin or absent early-season snowpacks), the model fell back to a meteorological-only feature set (HS, HN24, TA_max), sacrificing snowpack structural information but preserving trainability.

### 3.3 Combining Station and Regional Data

We built one model that lets every station borrow strength from the rest of the region while still keeping its own local signal (Gelman & Hill, 2007). Each of the 13 stations with at least one recorded event keeps its own baseline avalanche rate, while the influence of the five snow and weather variables is shared across the whole network. How much a station relies on its own history versus the regional pattern is learned from the data: a station with several observed events keeps mostly its own signal, while a station with only one event leans heavily on the regional average. In effect, the model trusts local history where there is enough of it, and falls back on regional experience where there is not.

Stations with no recorded events were left out of training, since in this region a quiet record more plausibly means no one was watching than that no avalanches occurred; at prediction time, these stations are simply served by the regional pattern. This approach also gives every prediction a built-in confidence estimate at no extra cost. Finally, we fit the model (a hierarchical, or multilevel, logistic regression) with the PyMC software package (Abril-Pla et al., 2023).

### 3.4 Evaluation

We tested the model three ways:

1. **Rolling per-station test.** For each station, we lined up its observed events in time order and asked: using only what the model knew before each event, did it warn in time? Each test trains on everything that came before that event at that station, pulling from either season, and checks the next event. Because training only ever looks backward in time, this measures how much accumulated local history helps; it deliberately mixes seasons, so it is not a test of a brand-new year. We counted an event as detected if the model's highest predicted probability in the three days up to and including the event reached 50%.
2. **Cross-season holdout.** We trained only on the 2024–2025 season and evaluated entirely on 2025–2026, so nothing from the test season ever reached training. This checks whether the model holds up in a year it has never seen. We report two standard skill scores here: AUC-ROC, which measures how well the model ranks dangerous days above safe ones (1.0 is perfect, 0.5 is no better than chance), and average precision, which is more informative than AUC-ROC when true events are rare, as they are here: roughly 2% of pooled station-days carry a positive label in the 2025–2026 evaluation season, versus 0.7% in 2024–2025 training (§3.2) — a gap driven by season coverage rather than a change in reporting: avalanches were recorded at all 30 matched stations in 2025–2026 but only 15 of 30 in 2024–2025, so a larger share of the pooled 2025–2026 station-days sit near a reported event (Saito & Rehmsmeier, 2015).
3. **Event-level scoring.** For every test event, we took the model's highest probability in the three days before it as its warning score, and compared it against equivalent windows on quiet days at the same stations. This asks the practical question — did the model raise a timely warning — rather than whether it labeled every single day correctly.

## 4. Results

### 4.1 Per-Station Performance and Learning Curve

We evaluated each station by asking: given what the model knew from past events at that location, did it correctly flag the next event before it happened? Across the 16 stations with enough observations to test, eight correctly flagged every event, three caught roughly half, and five missed entirely. Note that all five in the last group had only a single prior event to learn from, which is too little to build a reliable signal.

The clearest finding is how performance improves significantly as the model accumulates local observations (Table 2 and Figure 2). With just one prior event at a station, the model detected 56% of the next events. With two, detection climbed to 86%. With three, it reached 100%. Each additional observation delivers a large, consistent gain; the dataset sits on the steep part of the learning curve where more data matters most.

![Scatter plot of per-station event-window probability against number of prior training events. Detected events are green circles and missed events are red X marks, so the two classes are distinguishable by both color and shape; a black dashed mean line rises from 0.48 at one prior event to 0.78 at two and 0.72 at three.](./figures/fig2_learning_curve.png)

**Figure 2.** Learning curve: detection rate and mean warning probability as a function of the number of prior observed events at a station.

**Table 2.** Detection rate improves steeply with each additional observed event.

| Prior events at station | Detection rate | Mean warning probability |
|---|---|---|
| 1 | 56% | 0.48 |
| 2 | 86% | 0.78 |
| 3 | 100% | 0.72 |

### 4.2 Cross-Season Performance

Trained on 2024–2025 data and tested against all 37 observed events in 2025–2026, the model issued warnings at the right time far more often than chance: it ranked dangerous days above safe ones with an AUC-ROC of 0.715 (where 1.0 is perfect and 0.5 is no skill), and its warning precision was seven times better than random guessing. The hierarchical model also outperformed every simpler configuration we tested: per-station, regionally pooled, and blended classifiers. Its key advantage is that it shares information across all stations: a location with few observations benefits from patterns learned at better-observed stations nearby, while still adapting to each location's individual history. The full model comparison and statistical detail are available in the supplementary technical report ([Sheleg et al., 2026](https://github.com/ronimos/caw-avalanche-mapping/blob/main/docs/methods_and_results.md)).

### 4.3 What the Model Learned

The model's learned feature weights are physically sensible (Figure 3). Total snow depth is by far the strongest predictor of a valley-floor avalanche: a deep snowpack provides the volume and energy needed to reach the valley. Weak-layer burial depth is the second-strongest signal, acting in the opposite direction: the deeper a weak layer is buried, the harder it is to trigger, so shallow weak layers raise the modeled risk. New snowfall and warm temperatures both increase risk, with warming carrying roughly equal weight to new snow. This is consistent with the importance of rain-on-snow events in this climate. The Sn38 structural stability index adds modest additional signal. These patterns hold across the region and align with what experienced forecasters expect from the terrain and climate in Central Asia.

![Horizontal bar chart of standardized logistic-regression coefficients for the five model features. HS has the longest red (risk-increasing) bar, followed by TA_max and HN24; sn38_min is a short red bar; depth_lower_wl is the only blue (risk-decreasing) bar.](./figures/fig3_feature_importance.png)

**Figure 3.** Feature influence on predicted avalanche risk, with standardized model coefficients as a horizontal bar chart. Red bars increase risk; blue bars decrease risk; longer bars mean stronger influence.

### 4.4 Knowing What the Model Doesn't Know

A key advantage of the hierarchical approach is that it reports not just a probability, but how confident it is in that probability. At stations with several observed events, the model is tightly constrained; at stations with no local history, it relies on the regional average and correctly flags uncertainty in the prediction. This matters operationally: a high-risk probability at a well-observed station is a strong signal, while the same number at an unobserved station should prompt extra caution and expert judgment rather than automated action.

### 4.5 Operational Readiness

Testing each station against a standard of no more than roughly one false alarm per month (≤10% of non-event days), four stations are operationally ready today. One station produced zero false alarms across the entire 2025–2026 season; two others flagged fewer than three days all winter. Three more stations are in a marginal range, which can be useful with forecaster review of each alert. The remaining stations are not yet ready, primarily because they have too few observed events to anchor a reliable threshold. Each additional season of observations will move stations from the not-ready to the marginal or ready tier.

### 4.6 Case Study: Model Output for a Known Event

![Avalanche probability chart for station 160942 (east-facing) over the 2025-26 season, with operational threshold bands. Probability peaks near 1.00 in the highlighted three-day window around the January 23 event, and stays well below the 33 percent threshold on most other days.](./figures/fig4_stability_160942.png)

**Figure 4.** Per-station avalanche probability for station 160942_res (east-facing, best-performing station) over the 2025–2026 season, with operational threshold bands. The model assigns probability ~1.00 in the highlighted three-day window around the observed January 23 event, with markedly lower probabilities on surrounding non-event days.

Figure 4 illustrates the model's output for the best-performing station during a known event window. The probability spike is narrow, peaking immediately before the reported avalanche and returning to background levels within days. This is the pattern a forecaster would act on: a sharp, time-specific signal rather than a persistent high-risk background. The stability chart is the operational display that field staff and forecasters interact with directly.

## 5. Discussion

### 5.1 Strong signal from sparse data

The framework's core strength is how much signal it extracts from very little data. Restricting the model to a handful of core physical variables keeps it anchored to real avalanche triggers rather than fitting noise, and the logistic-regression coefficients expose exactly how each variable shifts the predicted risk. A forecaster can therefore inspect and sanity-check the model's reasoning directly, seeing why the model raised an alarm rather than only that it did.

The model also improves as it accumulates local observations. The system already performs well on the small training pool available today, and every additional observation makes a measurable improvement.

### 5.2 A chain of unverified assumptions

Building a forecasting tool for a data-sparse region forces a chain of assumptions, and each link carries risk that compounds down the chain. Because the region has essentially no in-situ weather stations, we drive SNOWPACK with numerical weather prediction rather than measurements. Small NWP errors propagate into the simulated snowpack, where SNOWPACK accumulates them layer by layer into a stability estimate that can drift from reality. With no surface stations to check the forecasts against, we cannot catch that drift. The pipeline rests on an educated but unverified physical hypothesis at every stage.

The observations carry their own bias. The dataset records only avalanches that people actually saw, so an avalanche that runs in an unpopulated drainage leaves no trace and the model treats that day as quiet. This skews the record toward populated valleys and roads, systematically undersampling remote terrain.

### 5.3 Decision support, not a replacement

These gaps set a hard limit on how forecasters should use the tool: it informs human decisions rather than making them. We built it as a support layer for experts, not a substitute. It gives forecasters better-organized information and a physically grounded second opinion. Expert judgment is not an afterthought bolted onto the output; the framework embeds it structurally, at the start-zone identification step that every observation passes through. The model sharpens and scales that judgment; it does not replace it.

### 5.4 Future work

Three extensions would address the limitations above and broaden the framework's reach.

**Satellite avalanche detection.** The observation bias is the framework's most consequential weakness, and satellite remote sensing offers a direct remedy. Synthetic-aperture radar (SAR) detects avalanche debris through cloud and darkness, so it can catch avalanches in remote, unpopulated drainages that no observer would ever report. We plan to fold SAR-based detections into the training corpus. This would enlarge and de-bias the label set and, just as important, let us confront model outputs with independently measured activity, grounding the predictions in observed reality rather than a purely simulated hypothesis.

**Regional situational-awareness view.** Beyond feeding the model, a fused record of recent avalanche activity from satellite detections and human observations would give forecasters the regional big picture that point predictions alone cannot provide. We plan to produce a region-scale visualization of recent avalanching that overlays both sources onto the terrain, so a forecaster can see at a glance where the snowpack has been failing across the whole domain and read each station's probability in that wider context.

**Path-specific flow modeling.** The current framework matches each observation to a start zone but stops short of modeling what happens below it. Coupling the stability output to an avalanche flow and runout model would let the system reason about path-specific behavior, turning a per-station probability into a spatially explicit hazard estimate and building directly on the terrain-tracing already in the pipeline.

## 6. Conclusions

A framework combining NWP-forced SNOWPACK simulations with opportunistically collected avalanche observations produces physically consistent, statistically skillful avalanche probability guidance in a region with near-zero conventional data. Trained on one season and tested on the next, the hierarchical model ranked dangerous days well above safe ones and issued warnings seven times more precisely than chance, and four stations already meet an operational false-alarm standard.

Three findings shape the path forward: performance scales steeply with observation density, so each added season is the cheapest route to improvement; the learned feature weights match avalanche physics, supporting the framework's coherence even without direct snowpack validation; and the model reports its own confidence at every station, letting forecasters weigh predictions grounded in local history against those extrapolated off the regional pattern.

We view this framework as a decision-support layer whose reach grows with every observation added, and as a template for avalanche forecasting in data-sparse mountain regions worldwide.

## Acknowledgements

We want to thank the American Avalanche Association for supporting this work through the ISSW 2026 Young Professional Scholarship.

## References

Abril-Pla, O., Andreani, V., Carroll, C., Dong, L., Fonnesbeck, C. J., Kochurov, M., Kumar, R., Lao, J., Luhmann, C. C., Martin, O. A., Osthege, M., Vieira, R., Wiecki, T., & Zinkov, R. (2023). PyMC: a modern, and comprehensive probabilistic programming framework in Python. *PeerJ Computer Science*, 9, e1516. https://doi.org/10.7717/peerj-cs.1516

Acharya, A., Steiner, J. F., Walizada, K. M., Ali, S., Zakir, Z. H., Caiserman, A., & Watanabe, T. (2023). Review article: snow and ice avalanches in high mountain Asia — scientific, local and indigenous knowledge. *Natural Hazards and Earth System Sciences*, 23, 2569–2592. https://doi.org/10.5194/nhess-23-2569-2023

Chabot, D. (2024). Snow, rain, and an earthquake: a massive avalanche cycle in Tajikistan and Afghanistan. *Proceedings of the International Snow Science Workshop*, Tromsø, Norway.

Chabot, D., & Kaba, D. (2016). Avalanche forecasting in the central Asian countries of Afghanistan, Pakistan and Tajikistan. *Proceedings of the International Snow Science Workshop*, Breckenridge, CO, USA.

Elkan, C., & Noto, K. (2008). Learning classifiers from only positive and unlabeled data. *Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 213–220. https://doi.org/10.1145/1401890.1401920

Gelman, A., & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.

Lehning, M., Bartelt, P., Brown, B., Fierz, C., & Satyawali, P. (2002). A physical SNOWPACK model for the Swiss avalanche warning services. Part II: snow microstructure. *Cold Regions Science and Technology*, 35(3), 147–167.

Morin, S., Horton, S., Techel, F., Bavay, M., Coléou, C., Fierz, C., Gobiet, A., Hagenmuller, P., Lafaysse, M., Ližar, M., Mitterer, C., Monti, F., Müller, K., Olefs, M., Snook, J. S., van Herwijnen, A., & Vionnet, V. (2020). Application of physical snowpack models in support of operational avalanche hazard forecasting: a status report on current implementations and prospects for the future. *Cold Regions Science and Technology*, 170, 102190. https://doi.org/10.1016/j.coldregions.2019.102910

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Sheleg, I., Chabot, D., Snook, J., Peters, J., Steinkogler, W., & Simenhois, R. (2026). *Methods and Results: Avalanche Forecasting in Data-Sparse Central Asia (Supplementary Technical Report)*. GitHub repository, `caw-avalanche-mapping`. https://github.com/ronimos/caw-avalanche-mapping/blob/main/docs/methods_and_results.md

Skamarock, W. C., & Klemp, J. B. (2008). A time-split nonhydrostatic atmospheric model for research and NWP applications. *Journal of Computational Physics*, 227, 3465–3485.

Snook, J. S. (2016). Weather forecast model grid spacing — is smaller better? *Proceedings of the International Snow Science Workshop*, Breckenridge, CO, USA.

Snook, J. S., Cooperstein, M., & Greene, E. (2022). Snowpack modeling efforts at the Colorado Avalanche Information Center. *Proceedings of the International Snow Science Workshop*, Bend, OR, USA.
