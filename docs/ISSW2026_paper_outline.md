# ISSW 2026 Paper Outline

**Title:** Avalanche Forecasting in Data-Sparse Central Asia: A Weak-Supervision Framework Linking NWP-Driven Snowpack Simulations to Valley-Floor Observations

---

## 1. Introduction (~0.5 page)

- Avalanche risk in Central Asia: humanitarian scale, geographic/political data barriers
- AKAH program context: operational need without operational data
- The data-sparse forecasting problem: no snowpack pits, no automatic weather stations, no systematic avalanche records
- Thesis: heterogeneous public data (NWP, satellite imagery, social media) can support meaningful probabilistic guidance when framed as decision-support rather than autonomous prediction
- Paper scope: methods, two-season results, physical plausibility assessment, path to operationalization

---

## 2. Data (~0.75 page)

### 2.1 SNOWPACK Simulations

- NWP forcing (WARF or similar, 4 km grid) → SNOWPACK per virtual station
- Station network: 23 stations, Darvoz/Tajikistan region, two seasons (2024-25, 2025-26)
- Key outputs used: layer heights, Sn38 stability index, HS, HN24, TA, rain rate
- Acknowledged limitation: no in-situ validation of simulated snowpack state

### 2.2 Avalanche Observations

- AKAH structured observation log: multi-year, Pakistan + Tajikistan
- Social media extraction: methodology for identifying valley-floor events, confidence criteria
- Weak-label framing: reports as probabilistic indicators, not ground truth
- Total: ~30 events across two seasons; dataset size acknowledged as a primary constraint

### 2.3 Observation-to-Station Matching

- Valley-floor reports (social media + AKAH log) provide approximate avalanche runout locations, not start zones
- **Expert terrain interpretation**: a practitioner with regional knowledge identified the likely avalanche start zones for each event based on valley geometry, known paths, and observation context
- Matched each start zone to the nearest SNOWPACK station sharing the same **aspect** — ensuring the simulation reflects the relevant solar radiation exposure and wind loading regime
- This hybrid human-machine step is a deliberate design choice: it embeds domain knowledge where algorithmic matching would fail (nearest-by-distance alone can assign the wrong slope aspect, which drives fundamentally different snowpack evolution)
- Acknowledged limitation: single station per event; no weighting for elevation band differences between start zone and station

---

## 3. Methods (~1 page)

### 3.1 Feature Engineering

- Daily aggregation from sub-daily SMET: HS, HN24, HN72, rain sum, TA_max
- Upper/lower zone split (top 40% by burial depth): min Sn38 per zone, depth of weakest lower-zone layer
- Physical rationale for zone split: near-surface storm slabs vs. deep persistent weak layers

### 3.2 Labeling Under Weak Supervision

- Positive day definition: observed event OR day prior (one-day reporting lag)
- Class imbalance handling: balanced class weights in logistic regression
- Fallback: met-only feature set when zone features unavailable on positive days (thin early-season snowpack)

### 3.3 Per-Station Logistic Regression Classifier

- One model per station; trained on all available seasons combined
- Feature set: 8 features (met + zone stability)
- Model persistence: saved per station, reloaded for prediction

### 3.4 Cross-Season Evaluation

- 2025-26 season used for training; 2024-25 treated as approximate held-out evaluation
- Metrics: AUC, precision/recall at operational thresholds (50/65/80/95%)
- Caveat on small N; results interpreted as indicative, not conclusive

---

## 4. Results (~1.25 pages)

### 4.1 Classifier Performance

- Per-station and aggregate AUC / skill scores across both seasons
- Discrimination between avalanche and non-avalanche days
- Performance on 2024-25 held-out season (cross-season generalization)

### 4.2 Physical Plausibility of Learned Weights

- HN24 and TA_max as dominant predictors — consistent with loading and rain-on-snow physics
- Positive TA_max coefficient: rain-on-snow at least as important as new snow loading in this climate
- Sn38 zone features: contribution where data is sufficient
- Coefficient signs/magnitudes as indirect validation of physical coherence

### 4.3 Forecast Map and Probability Output

- Case study: known event window, map probability coloring, stability plot walkthrough
- Illustration of how the tool surfaces risk at the right time and location

---

## 5. Discussion (~0.75 page)

### 5.1 What the System Does Well

- Physically consistent response to weather drivers
- Reasonable discrimination with minimal labeled data
- Transparent, interpretable model (logistic regression coefficients readable by practitioners)

### 5.2 Known Limitations and Honest Uncertainty

- Small event dataset: statistical power is limited, results are hypothesis-generating
- Model chain risk: NWP → SNOWPACK → classifier — errors accumulate, no internal validation point
- No snowpack observations: simulated Sn38 unverified against real stratigraphy
- Social media labels: geographic imprecision, reporting bias, survivorship
- Expert terrain interpretation introduces subjectivity at the matching step; not reproducible without the same practitioner

### 5.3 Decision-Support Framing

- The system was designed with expert judgment embedded at a critical step (start-zone identification), not bypassed — this is a feature, not a gap
- Probability output + threshold bands give forecasters actionable signal with explicit uncertainty
- Tool is designed to inform expert judgment, not replace it
- AKAH integration pathway: how would this fit into an existing operational workflow?

---

## 6. Future Work (~0.5 page)

### 6.1 Satellite Avalanche Detection

**Near-term — optical and SAR debris mapping**
- Sentinel-1 SAR backscatter change detection and Sentinel-2 optical debris mapping: free, global, operationally mature
- Deep learning on SAR time series (CNN/transformer detectors) improving rapidly over simple threshold methods — more robust to wet snow noise
- Commercial high-resolution optical (Planet Labs, ~3–5 m daily) would shorten detection latency to ~24 hours in cloud-free conditions; cost declining
- Debris polygons could automate start-zone inference (backtrack from runout up the slope), reducing but not eliminating the expert bottleneck at the matching step

**Medium-term — NISAR**
- NASA-ISRO dual-frequency (L-band + S-band) SAR, 12-day global repeat; L-band penetrates deeper into snow than Sentinel-1 C-band
- Particularly relevant given the rain-on-snow signal identified in this study — L-band is more sensitive to wet snow conditions
- Could provide both improved debris detection and additional snowpack state information to feed the model chain

**Longer-term — InSAR precursor detection**
- Experimental frontier: slope-scale surface creep or settling in an unstable snowpack may produce a detectable InSAR phase signal prior to release
- ESA ROSE-L (L-band, ~2029) would have the penetration depth and coherence to make this plausible
- Not operationally demonstrated; included as a research horizon

### 6.2 Seismic Avalanche Detection

- Large avalanches generate distinctive low-frequency seismic signals detectable tens of kilometers away; well-documented in European and North American contexts
- Central Asia has existing seismic monitoring infrastructure (earthquake networks), making this a low-marginal-cost detection pathway in principle
- **Data access caveat**: sparse global network stations (IRIS/EarthScope, GEOFON/GFZ) are publicly available but spaced 100–300 km apart — likely insufficient for detecting all but the largest events; dense local networks operated by national agencies (e.g. Tajikistan IGEES) are not systematically open, presenting the same government data-sharing barrier identified as a core constraint in this study
- Pathway: leverage AKAH's existing governmental relationships to negotiate access, or evaluate deployment of independent low-cost MEMS seismic sensors at key valley locations
- Pairs naturally with the weak-label framework — seismic detections could feed the same observation pipeline currently supplied by social media

### 6.3 Ensemble NWP Forcing

- Run multiple NWP members through SNOWPACK → probability distribution over snowpack states
- Honest uncertainty propagation through the model chain rather than false single-value precision
- More appropriate product for operational forecasters

### 6.4 Richer Prediction Models

- More seasons → more events → gradient boosted models or neural approaches become viable
- Cross-station pooling or regional models to overcome per-station data poverty
- Incorporation of terrain parameters (aspect, elevation, slope) in station selection and feature set

---

## 7. Conclusions (~0.25 page)

- A data fusion framework combining NWP-forced SNOWPACK and weak observational labels produces physically consistent and statistically discriminating avalanche probability guidance in a region with near-zero conventional data
- Physical plausibility of learned predictors supports framework coherence even without direct snowpack validation
- Expert terrain knowledge embedded at the observation matching step is integral to system credibility, not a limitation to be automated away
- Best interpreted as a decision-support layer; satellite detection is the critical next investment for operational scaling

---

## References

- SNOWPACK model documentation (Lehning et al.)
- Sn38 stability index (Schweizer — ISSW 2016)
- Sentinel-1 SAR avalanche detection (Eckerstorfer et al., Bühler et al.)
- Weak supervision / noisy label learning framing
- AKAH program references (if available/permitted)
