# NeuroGeo-Field: Spatial Inference Engine

## Overview
NeuroGeo-Field is an evidence-backed spatial inference engine that transforms raw descriptive address text, commercial priors, and operational field visit logs into accurate 2D metric coordinates. In physical loan collections across India, commercial geocoders (Google Maps, MapmyIndia) often fail on descriptive addresses, leading to high median errors (376.4m) and numerous lost visits. This system uses operational field visit trajectories and a hierarchical spatial graph to achieve an **83.8% reduction in median error**.

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
- **Median Error Reduction**: Reduced the overall median error from **376.40m to 60.81m** (an 83.8% reduction).
- **Visited Addresses**: Median error reduced from 383.67m to **21.10m**.
- **Cold-Start Addresses**: Median error reduced from 362.87m to **178.04m**.
- **Validation**: Strict leave-one-street-out (LOSO) and Simulated Cold-Start Validation (SCV) methodologies guarantee zero leakage, achieving **82.0%** empirical bounding radius coverage on the test set.
- **Independence**: The full pipeline is reproducible locally using a 100% self-contained Jupyter notebook (`submission.ipynb`).

## Implemented Integrations & Features
The following structural improvements from the roadmap have been fully implemented across the respective codebases:

1. **Parser Upgraded: MuRIL NER Integration**
   Replaced the deterministic rule-based parser with a MuRIL Transformer NER approach (from the `Address-Parser-for-Geoencoder` repository) in `src/neurogeo/address_parser.py` and `master.ipynb`. This deep learning language model understands contextual Hindi/Kannada transliterations, dramatically increasing the extraction rate of landmarks and streets, boosting the availability of high-confidence candidates ($C_3$ and $C_4$), and lowering cold-start errors.

2. **Bayesian/ML Fallback for "Graph Deserts"**
   Integrated a Bayesian Grid Posterior ML model as a fallback Candidate ($C_6$). When exact street matches fail in rural areas, the ML model serves as a fallback mechanism within `parse_address`, bridging the gap between precise street matches and vague pincode centers.

3. **Human-in-the-Loop Validation (UI/Dashboard)**
   Integrated with the Next.js Dashboard. The UI visualizes the "Evidence Card" and plots the target, the commercial baseline, and the candidates on an interactive map. We added action buttons directly into the frontend interface (`page.tsx`) to allow operations managers to override or snap coordinates.

4. **Snap to Navigable Road Networks (OSM Integration)**
   Integrated OpenStreetMap (OSM) via backend endpoints. After predicting the final $(x,y)$ coordinate, the new `/api/snap_to_osm` endpoint "snaps" it to the nearest navigable road segment, enabling actual turn-by-turn driving/walking distances.

5. **Active "Address Decay" Feedback Loop**
   Added an API endpoint (`/api/feedback`) for field agents to report `location_moved` or `incorrect_pin`. When this feedback is received, the engine dynamically penalizes or purges that specific historical check-in from the Hierarchical Spatial Graph, ensuring the model actively "forgets" outdated spatial knowledge.
