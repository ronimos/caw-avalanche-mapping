# Forecasting Avalanches in Data-Sparse Central Asia using Snowpack and Weather Modeling with Avalanche Observations# 
Itai Sheleg1, Doug Chabot2, John Snook3, Jaime Peters1, Ron Simenhois3

- 1 Lake County High School, Leadville, Colorado
- 2 Latok, LLC, Bozeman, MT
- 3 Colorado Avalanche Information Center, CO, USA

Avalanches in Central Asia, particularly in Afghanistan, Tajikistan, and Pakistan, are a persistent hazard, causing hundreds of fatalities in severe winters and routinely destroying homes, infrastructure, and livestock essential for survival. To address this risk, in 2015, Aga Khan Agency for Habitat established a remote avalanche forecasting program supported by external expertise.

Following its implementation, it became clear that a primary constraint on forecasting capability is the limited availability of snowpack, weather, and avalanche observations, driven by both geographic remoteness and government restrictions on data sharing.

We address this gap by developing a proof-of-concept framework that fuses heterogeneous, publicly available data sources to support avalanche forecasting in data-sparse regions. Specifically, we combine numerical weather prediction (NWP)-forced SNOWPACK simulations (4 km grid spacing) with opportunistically extracted avalanche observations from social media posts shared by local residents. This approach treats informal reports as weakly labeled observations, enabling the reconstruction of avalanche occurrence in the absence of systematic records.
We identified 30 valley-floor avalanche events from social media and reconstructed the associated meteorological and snowpack conditions using nearby SNOWPACK outputs. Analyzed variables include precipitation totals and phase, air temperature, snow depth, weak-layer depth, and the natural stability index (Sn38). To distinguish avalanche from non-avalanche days, we trained per-station logistic regression classifiers on daily simulation-derived features.
Across 23 modeled stations, 24-hour new snow and maximum air temperature (TA_max) emerge as the most consistent predictors. Positive TA_max coefficients indicate that rain-on-snow events are at least as important as new-snow loading in driving avalanche activity.

This work incorporates a novel data-fusion and weak-supervision framework for avalanche forecasting, integrating NWP-driven snowpack modeling with opportunistic, publicly sourced observations. However, this framework is constrained by a limited event dataset and a model chain that lacks rigorous, component-wise validation, introducing uncertainty in both inputs and predictions. As such, the framework is best interpreted as a decision-support tool rather than a fully autonomous forecasting system. Coupled with expert validation and interpretation, it provides a scalable pathway for developing operational guidance in regions where conventional data streams are sparse or unavailable.
