# Map Visualization Design — Open Discussion

**Status:** Pre-decision. Review with Doug (operational end-user) before implementing anything.
**Participants so far:** Ron, Claude Code session (2026-06-03)

---

## Context

The current avalanche map (`output/avalanche_map.html`) shows one circle marker per matched SNOWPACK **station** — not per observation record. Marker color encodes the forecast probability at that station (gray → green → gold → orange → red → dark red). Clicking a marker opens the station's stability plot. Hovering shows a tooltip with observation details.

**Tooltip and click-to-open-stability-plot must be preserved in any redesign.**

We now have two additional pieces of information that should be communicated visually but are not yet:

1. **Model confidence tier** — derived from the operational threshold analysis (§4.6 of methods_and_results.md). Each station's classifier falls into one of three tiers based on false alarm rate at its recall-maximizing threshold:
   - Ready: FA ≤ 10% (currently: 160942_res E, 164801_res N)
   - Marginal: FA 10–25% (currently: 165202_res E)
   - Not ready: FA > 25% (all others)

2. **Whether the probability came from a trained per-station model or the regional fallback** — stations without enough observations to train a per-station model currently show gray ("no classifier"), but the regional pooled model can still produce a probability for them.

---

## Open Questions to Discuss with Doug

### Q1 — What should a forecaster see on the map?

**Current behavior: Station locations only**
Markers sit at the virtual SNOWPACK station on the slope above the valley. The probability color reflects conditions at that slope. Debris/runout locations are not shown.

**Option A: Add debris/runout locations as a separate layer (toggleable)**
Show observation markers at the valley-floor/road locations where AKAH staff reported the event. This is where the hazard manifests and where people are exposed. Keep station markers as the probability layer. Forecasters can toggle each layer independently — turning on debris locations for exposure context, turning off clutter when not needed. Tooltip and click-to-open remain on both marker types.

**Option B: Replace station markers with debris markers**
Move the probability color to the debris location. More intuitive for field staff (marker is at the place of concern), but severs the visual link to the slope driving the model.

**Option C: Show both, linked, as a combined default layer**
A station marker on the slope with a thin line connecting to the debris marker in the valley. Shows the full chain: where the model runs → where the hazard reaches. Informative but potentially cluttered where multiple events share a station. Could also be offered as a toggleable layer.

**Option D: Debris location in tooltip only**
Keep station marker as the primary map element; add debris coordinates, distance from station, and observer notes to the hover tooltip. Lightweight — no extra map geometry.

**Layer toggle note:** Options A and C lend themselves naturally to Folium layer toggles, which already exist on the map for base layers. Adding "Debris locations" as a toggleable overlay avoids clutter while keeping the information available. Tooltip and click behavior must be preserved on all active layers.

**Doug's question:** When you open the map operationally, is your first instinct to locate the slope (to reason about snowpack) or the valley floor (to assess exposure and community risk)?

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

- The current map shows **station locations**, not debris/runout locations. This is a correction from the original document.
- Probability color (current) stays — no changes to the risk scale itself.
- Tooltip and click-to-open-stability-plot must be preserved in any redesign, on all active layers.
- Fill opacity is the leading candidate for confidence tier encoding.
- Layer toggles (already present for base maps) are the preferred way to add debris locations and other overlays without cluttering the default view.
- The distinction between "trained per-station model" and "regional fallback" matters operationally and should be visually clear, not buried in a tooltip.
- The "always-on" model problem (180343, 176522 showed 100% LOO detection but 70% false alarm rate) means a high forecast probability at a non-ready station should not carry the same visual weight as the same probability at a ready station — exactly what opacity encoding addresses.
