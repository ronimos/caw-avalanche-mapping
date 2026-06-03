# Map Visualization Design — Open Discussion

**Status:** Pre-decision. Review with Doug (operational end-user) before implementing anything.
**Participants so far:** Ron, Claude Code session (2026-06-03)

---

## Context

The current avalanche map (`output/avalanche_map.html`) shows one circle marker per observation record. Marker color encodes forecast probability (gray → green → gold → orange → red → dark red). Clicking a marker opens the station's stability plot. Hovering shows a tooltip.

We now have two additional pieces of information that should be communicated visually but are not yet:

1. **Model confidence tier** — derived from the operational threshold analysis (§4.6 of methods_and_results.md). Each station's classifier falls into one of three tiers based on false alarm rate at its recall-maximizing threshold:
   - Ready: FA ≤ 10% (currently: 160942_res E, 164801_res N)
   - Marginal: FA 10–25% (currently: 165202_res E)
   - Not ready: FA > 25% (all others)

2. **Whether the probability came from a trained per-station model or the regional fallback** — stations without enough observations to train a per-station model currently show gray ("no classifier"), but the regional pooled model can still produce a probability for them.

---

## Open Questions to Discuss with Doug

### Q1 — What should a forecaster see on the map?

**Option A: Show debris/runout locations only (current)**
Markers sit where the avalanche was observed — valley floor or road crossing. This is where the hazard manifests and where AKAH field staff reported it. Operationally intuitive: the marker is at the place of concern.

**Option B: Show station locations only**
Markers sit at the virtual SNOWPACK station on the slope above. This is where the model runs and the probability is computed. More honest about model geography, but disconnects from where people are affected.

**Option C: Show both, linked**
A small station marker on the slope connected by a thin line to the debris marker in the valley. Shows the full chain: where the model lives → where the hazard reaches. More informative but potentially cluttered, especially where many events cluster near the same valley section.

**Option D: Debris marker with station information in tooltip**
Keep debris location as the primary marker; add station coordinates, distance, and aspect to the hover tooltip. Lightweight — no extra map geometry.

**Doug's question:** When you look at the map operationally, do you want to see where the avalanche ended up (to assess exposure) or where the slope is (to understand the snowpack)?

---

### Q2 — How do we show model confidence tier?

We discussed three visual channels. The current map uses fill color for probability, so confidence needs a separate channel:

**Option A: Fill opacity** *(preferred based on discussion)*
- Ready: `fillOpacity = 0.90` — vivid, full signal
- Marginal: `fillOpacity = 0.55`
- Not ready: `fillOpacity = 0.25` — washed out

Reads naturally: a pale red circle = "model says high risk but we don't trust this station's model yet." Vivid green = "low risk and we trust that." Two independent channels (color = what, opacity = how much to trust it).

**Option B: Marker size**
- Ready: radius 13, Not ready: radius 7
Simple but conflates "importance" with "confidence" in a way that may mislead.

**Option C: Icon shape** (circle / diamond / square)
Visually distinct but hard to implement cleanly in Folium and may be unfamiliar to forecast users.

**Doug's question:** Which encoding is clearest in your operational workflow? Do you read map opacity intuitively, or would a shape distinction or legend badge be more obvious?

---

### Q3 — What do we do with stations that have no trained model?

Currently stations without a per-station classifier show a gray marker ("no classifier"). Two upgrade paths:

**Option A: Use the regional model for all unclassified stations**
Every station on the map gets a probability — either from its per-station model (if trained) or from the regional pooled model (always available). The visual distinction between the two would be encoded via Q2's confidence tier (regional-only = "not ready" opacity).

Pros: no gray markers; full map coverage; regional model has AUC-ROC = 0.61.
Cons: the regional model may suggest spurious local risk. A forecaster might over-interpret a regional-derived red marker.

**Option B: Keep gray for no per-station model, add regional as a separate layer**
Keep the current behavior but add an optional Folium layer toggle: "Regional model background." Forecasters can turn it on to see regional estimates or leave it off for cleaner per-station-only view.

**Option C: Show nothing for stations without a trained model**
Only show markers where we have at least one LOO fold's worth of evidence. Clean and conservative, but leaves many map areas blank.

**Doug's question:** Would seeing a regional-model probability on a station with no local training data be helpful or confusing? Is "something is better than nothing" or "don't show what you can't support" the right operational philosophy here?

---

### Q4 — Should unobserved stations appear at all?

Currently only stations that match at least one observation get a marker. The full network has 30 stations across the region. Should the map show:

- **All 30 stations** with their current model probability (per-station where available, regional otherwise)? This gives full spatial coverage and lets forecasters see modeled risk everywhere, not just where we happen to have records.
- **Only observed stations** (current)? Keeps the map focused on validated locations.
- **All stations but visually subdued** (smaller radius, lower opacity, distinct color scale) for unobserved ones?

This connects directly to Q3: if we use the regional model for unobserved stations, showing all 30 provides a continuous probability surface across the region.

---

## Visual Channel Summary

| Channel | Currently used for | Proposed additional use |
|---|---|---|
| Fill color | Forecast probability | — (keep) |
| Fill opacity | — (always 0.88) | Model confidence tier |
| Marker radius | — (always 10) | Possibly size by confidence |
| Marker shape | — (always circle) | Possibly tier encoding |
| Stroke/border | Subtle outline | Could encode trained vs regional |
| Layer toggle | Base map selection | Could separate debris vs station markers |

---

## What We Want from the Doug Conversation

1. Which location to primary-mark: debris, station, or both?
2. Which confidence encoding reads best operationally (opacity / size / shape)?
3. Regional model as fallback for unclassified stations: yes or no?
4. Show all 30 stations or only observed ones?
5. Any other field-use context that should drive the design (e.g., is the map used on mobile? projected in a meeting room? printed?)?

---

## Notes from the Discussion So Far

- Probability color (current) stays — no changes to the risk scale itself.
- Fill opacity is the leading candidate for confidence tier encoding.
- The distinction between "trained per-station model" and "regional fallback" matters operationally and should be visually clear, not buried in a tooltip.
- The "always-on" model problem (180343, 176522 showed 100% LOO detection but 82–97% false alarms) means that a high forecast probability at a non-ready station should not carry the same visual weight as the same probability at a ready station — exactly the problem opacity encoding addresses.
- The map currently shows observation debris locations. That may be the right default since that is where AKAH staff observed and reported hazard.
