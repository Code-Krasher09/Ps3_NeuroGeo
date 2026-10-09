# NeuroGeo-Field: Spatial Inference Engine

## Overview
NeuroGeo-Field is an evidence-backed spatial inference engine that transforms raw descriptive address text, commercial priors, and operational field visit logs into accurate 2D metric coordinates. In physical loan collections across India, commercial geocoders (Google Maps, MapmyIndia) often fail on descriptive addresses, leading to high median errors (376.4m) and numerous lost visits. This system uses operational field visit trajectories and a hierarchical spatial graph to achieve an **83.7% reduction in median error**.

## Core Architecture: Retrieve → Rank → Calibrate
The architecture explicitly rejects continuous coordinate regression. Instead, it formulates geocoding as **discrete spatial hypothesis generation and evidence-weighted reranking**:

1. **Address Understanding & Entity Resolution**: Extracts multi-lingual Indian address tokens, canonicalizing streets and bounded landmarks.
2. **Spatial Candidate Generation**: Generates hypotheses including:
   - `C1`: Commercial Baseline Prior
   - `C2`: Direct Visit Doorstep Medoid (if previously visited)
   - `C3`: POI Anchor Coordinate (with relational cues)
   - `C4`: Hierarchically Scoped Neighbor Street Cluster
   - `C5`: Locality Centroid Fallback
3. **Candidate Scoring & Reranking**: Uses a deterministic linear scorer evaluating text-spatial semantic alignment, neighbor support, and negative search penalties.
4. **Uncertainty Calibration & Navigation**: Calibrates an empirical confidence radius and synthesizes offline, deterministic turn directions.

## Key Results & Pipeline Improvements
- **Median Error Reduction**: Reduced the overall median error from **376.40m to 61.26m** (an 83.7% reduction).
- **Visited Addresses**: Median error reduced from 383.67m to **21.10m**.
- **Cold-Start Addresses**: Median error reduced from 362.87m to **235.40m**.
- **Validation**: Strict leave-one-street-out (LOSO) and Simulated Cold-Start Validation (SCV) methodologies guarantee zero leakage, achieving ~90% empirical bounding radius coverage.
- **Independence**: The full pipeline is reproducible locally using a 100% self-contained Jupyter notebook (`master.ipynb`).

## Future Integrations & Roadmap
To further productionize the NeuroGeo-Field engine, the following structural improvements will be integrated:

1. **Upgrade the Parser: Replace Rule-Based Parsing with MuRIL NER**
   Replace the deterministic rule-based parser with a MuRIL Transformer NER approach (from the `Address-Parser-for-Geoencoder` repository). By extracting entities using a deep learning language model that understands contextual Hindi/Kannada transliterations, we will dramatically increase the extraction rate of landmarks and streets, boosting the availability of high-confidence candidates ($C_3$ and $C_4$) and lowering cold-start errors.

2. **Introduce a Bayesian/ML Fallback for "Graph Deserts"**
   Use a Bayesian Grid Posterior / Gradient Boosting ML model (from `CreditNirvana`) as a fallback Candidate ($C_6$). When exact street matches fail in rural areas, the ML model will interpolate a location based on textual similarity (TF-IDF embeddings) to regional clusters, bridging the gap between precise street matches and vague pincode centers.

3. **Add Human-in-the-Loop Validation (UI/Dashboard)**
   Integrate the Next.js/FastAPI Dashboard from `CreditNirvana`. Create a UI that visualizes the "Evidence Card", plotting the target, the commercial baseline, and the 5 candidate hypotheses on an interactive map. Show the confidence radius visually so operations managers can manually override if the system falls back to a low-confidence baseline.

4. **Snap to Navigable Road Networks (OSM Integration)**
   Integrate OpenStreetMap (OSM) via OSRM or GraphHopper. After predicting the final $(x,y)$ coordinate, "snap" it to the nearest navigable road segment. This will allow the engine to generate actual turn-by-turn driving/walking distances rather than just compass headings, making the offline navigator realistic for field agents.

5. **Build an Active "Address Decay" Feedback Loop**
   Add an API endpoint for field agents to report `location_moved` or `incorrect_pin`. When this feedback is received, dynamically penalize or purge that specific historical check-in from the Hierarchical Spatial Graph. This ensures the model actively "forgets" outdated spatial knowledge rather than perpetually generating coordinates for an address that no longer exists there.
