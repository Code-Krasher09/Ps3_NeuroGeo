# PS3 Implementation Plan & Engineering Milestones
*Document 5 of 5: Modular Build Plan, Test Assertions & Acceptance Criteria*
*Status: FROZEN (Implementation Plan Level)*

---

## 1. Phased Engineering Roadmap

```
PHASE 0: Evaluation Harness & Data Contracts
   │  └── Strict zero-leakage masking, metric calculation, Level 0 baseline reproduction
   ▼
PHASE 1: Operational Sanitizer & Feature Extractor
   │  └── Photo collision detection, dwell/speed weighting, Level 1 field baseline model
   ▼
PHASE 2: Multilingual Address Parser & POI Resolver
   │  └── Script normalization, entity extraction, locality-bounded POI disambiguation
   ▼
PHASE 3: Spatial Graph & Cold-Start Propagation
   │  └── Hierarchical street graph, simulated SCV & LOSO validation engines
   ▼
PHASE 4: Candidate Reranker & Quantile Calibrator
   │  └── 6-candidate generation (incl. C6 Bayesian), linear utility scorer, evidence-tier quantile radii
   ▼
PHASE 5: Packaging, Demo Notebook & Executive Deck
   │  └── Reproducible master notebook, interactive visual dashboard, presentation slides
   ▼
PHASE 6: Production & Integration (CreditNirvana UI/API)
      └── MuRIL NER parser, OSM Snapping, Feedback Loop, Next.js Dashboard
```

---

## 2. Phase-by-Phase Module Specifications

### Phase 0: Evaluation Harness (`src/evaluation.py`)
- **Objective**: Establish the official evaluation harness with zero-leakage guards before writing model code.
- **Inputs**: `surveyed_addresses.csv`, `splits.csv`, `baseline_geocodes.csv`.
- **Outputs**: Baseline error metrics report (`baseline_metrics.json`).
- **Tests & Assertions**:
  - `assert len(train_split) == 66`
  - `assert len(val_split) == 19`
  - `assert len(test_split) == 15`
  - Baseline geocoder median error on 100 surveyed addresses must evaluate to $376.4\text{m} \pm 0.5\text{m}$ under the documented benchmark calculation.
- **Acceptance Criteria**: Official baseline reproduction verified; isolation closure $\mathbf{Closure}(A^*)$ function implemented and unit-tested.

---

### Phase 1: Operational Sanitizer & Trajectory Miner (`src/sanitizer.py`, `src/trajectory_miner.py`)
- **Objective**: Compute visit-level integrity scores and extract trajectory features from GPS trails.
- **Inputs**: `field_visits.csv`, `visit_gps_points.csv`.
- **Outputs**: `visit_features.parquet` containing $I(v)$, $S(v)$, tortuosity $\tau_v$ (with stationary jitter & speed jump guards), and dwell medoids.
- **Tests & Assertions**:
  - `assert visit_features['visit_integrity_score'].between(0.0, 1.0).all()`
  - `assert visit_features['spatial_evidence_strength'].between(0.0, 1.0).all()`
  - Visits with repeated photo hashes (`ph0080`–`ph0084`) must have median integrity score $< 0.05$.
  - Clean `met_borrower` visits must have median integrity score $\ge 0.85$.
- **Model 1 Milestone**: Implement `Model 1 (Field Check-In Baseline)` using unweighted check-ins for visited addresses. Median error on visited benchmark addresses must be $\le 25.0\text{m}$.

---

### Phase 2: Multilingual Address Parser & POI Resolver (`src/address_parser.py`, `src/poi_resolver.py`)
- **Objective**: Parse unstructured text into structured entities and resolve local landmarks.
- **Inputs**: `addresses.csv`, `landmarks_poi.csv`, `localities.csv`.
- **Outputs**: `parsed_addresses.parquet`.
- **Tests & Assertions**:
  - Parser must support conservative postal PIN extraction and PIN-assisted locality disambiguation.
  - Parser must recognize localities in $\ge 65\%$ of in-scope addresses.
  - Parser must extract street tokens in $\ge 55\%$ of in-scope addresses.
  - Structural Invariant: POI Resolver must yield **0 duplicate POIs per type** when scoped by resolved `locality_id`.
  - Resolution Quality: Report distribution of entity resolution status (`resolved`, `unresolved`, `ambiguous`); benchmark address POI match rate evaluated against annotated landmark ground truth.
  - Multilingual addresses in Kannada (T1) and Devanagari (T2) must normalize without Unicode encoding errors.
  - **MuRIL NER Integration**: Replace rule-based logic with a deep learning MuRIL Transformer pipeline.
- **Acceptance Criteria**: Addresses with explicit landmarks (*"Bus Stop hattira"*, *"nr Medical Store"*) successfully anchor to unique `poi_id`s within 500m of locality centroid.

---

### Phase 3: Spatial Graph & Cold-Start Validation (`src/spatial_graph.py`)
- **Objective**: Construct the hierarchical street-landmark graph and build SCV/LOSO validation engines.
- **Inputs**: `parsed_addresses.parquet`, `visit_features.parquet`.
- **Outputs**: `street_centroids.parquet`, `simulated_cv_results.csv`.
- **Tests & Assertions**:
  - `assert all(node.startswith('T') for node in street_graph.nodes)` (Enforces hierarchical scoping: `T1::Locality::Street`).
  - No cross-locality hub node collisions allowed.
  - SCV engine successfully runs across 1,477 visited accounts with target visits, GPS breadcrumbs, and ground-truth coordinates masked.
  - LOSO engine successfully evaluates the 54 qualified streets.
- **Acceptance Criteria**: Knowledge transfer demonstrates $\ge 20\%$ median error reduction on simulated cold-start episodes relative to commercial baseline (with unvisited benchmark addresses separately reported).

---

### Phase 4: Candidate Reranker & Quantile Calibrator (`src/candidate_ranker.py`, `src/calibrator.py`, `src/navigator.py`)
- **Objective**: Generate 6 spatial candidates (C1–C6, including Bayesian Grid Fallback), score them via linear utility, calibrate confidence radius $R$ via evidence-tier empirical quantiles, and synthesize turn directions.
- **Inputs**: `parsed_addresses.parquet`, `street_centroids.parquet`, `visit_features.parquet`, `baseline_geocodes.csv`.
- **Outputs**: Full NeuroGeo-Field predictions (`predictions_benchmark.json`).
- **Tests & Assertions**:
  - Every predicted coordinate must satisfy the coordinate sanity bound ($\|(\hat{x}, \hat{y})\|_2 \le \text{approx\_radius\_m} + 200\text{m}$).
  - Calibration parameters derived from out-of-fold SCV residuals with sample-size fallback hierarchy ($N_{\min}=15$); verified to achieve $\ge 88\%$ empirical coverage on the 19 validation benchmark addresses.
  - Offline navigation strings must be deterministic, non-empty for all addresses, and snapped to OSM road segments.
- **Acceptance Criteria**: Overall median distance error on 100 surveyed addresses beats the commercial baseline by $\ge 80\%$ (target $< 50\text{m}$ overall, $< 25\text{m}$ visited).

---

### Phase 5: Demo Packaging & Executive Presentation (`notebooks/`, `README_SUBMISSION.md`)
- **Objective**: Package the codebase into a reproducible competition submission.
- **Deliverables**:
  1. `notebooks/01_master_pipeline.ipynb`: Self-contained, end-to-end executable notebook.
  2. `notebooks/02_interactive_dashboard.ipynb`: Interactive 2D visualization plotting:
     - 🔴 Commercial Geocoder (far away at centroid)
     - 🟡 Wandering GPS trails & searching loops
     - 🔵 NeuroGeo-Field prediction pin & calibrated confidence circle
     - 🟢 True surveyed doorstep
  3. `README_SUBMISSION.md`: Executive summary, ablation table (Models 0–7), error taxonomy report, and operational field ROI analysis.

---

### Phase 6: Production Integrations (CreditNirvana Backend/Frontend)
- **Objective**: Wrap the final engine into FastAPI and a Next.js Dashboard.
- **Deliverables**:
  1. `/api/candidates`: FastAPI endpoint serving predicted location and alternatives.
  2. `/api/snap_to_osm`: FastAPI endpoint for OSRM route snapping.
  3. `/api/feedback`: FastAPI endpoint processing address decay signals.
  4. Next.js Dashboard `page.tsx` rendering evidence cards, map overlays, and operational override buttons.

---

## 3. Directory Structure

```
d:/OpenIIT '26/
├── data/                                 # Official datasets (read-only)
│   ├── shared/
│   ├── ps1_fake_ptp/
│   ├── ps2_right_party_contact/
│   └── ps3_geocoder/
├── src/                                  # Production modular code
│   ├── __init__.py
│   ├── evaluation.py                    # Phase 0: Zero-leakage harness & metrics
│   ├── sanitizer.py                     # Phase 1: Visit integrity engine
│   ├── trajectory_miner.py              # Phase 1: GPS trail metrics & dwell medoids
│   ├── address_parser.py                # Phase 2: Multilingual entity parser
│   ├── poi_resolver.py                  # Phase 2: Locality-bounded POI disambiguator
│   ├── spatial_graph.py                 # Phase 3: Bipartite street knowledge transfer
│   ├── candidate_ranker.py              # Phase 4: 5-candidate utility reranker
│   ├── calibrator.py                    # Phase 4: Empirical quantile uncertainty model
│   └── navigator.py                     # Phase 4: Deterministic offline turn guidance
├── notebooks/
│   ├── 01_master_pipeline.ipynb
│   └── 02_interactive_dashboard.ipynb
├── tests/                                # Automated unit test suite
│   ├── test_leakage_isolation.py
│   ├── test_poi_disambiguation.py
│   └── test_candidate_bounds.py
├── PS3_FINAL_ARCHITECTURE.md            # Doc 1: Frozen Architecture
├── PS3_DATA_CONTRACT.md                 # Doc 2: Frozen Data Schemas & Audit
├── PS3_EVALUATION_AND_LEAKAGE_CONTRACT.md# Doc 3: Frozen Evaluation Contract
├── PS3_MODEL_SPEC.md                    # Doc 4: Frozen Model Specs & Formulas
└── PS3_IMPLEMENTATION_PLAN.md           # Doc 5: Frozen Implementation Plan
```

---

## 4. Final Sign-Off & Verification Checklist

Before commencing implementation, confirm the following sign-off checklist:
- [x] Empirical audit completed across all 8 datasets (schemas, missingness, FKs validated).
- [x] Zero-duplicate POI invariant proven for all 14 landmark categories across all 3 towns.
- [x] Trajectory physics and dwell distributions analyzed (100% trajectory completeness).
- [x] Mathematical bug in Evidence Strength formula resolved (bounded $[0, 1]$ monotonic decay).
- [x] Photo collision formula corrected to smooth exponential decay.
- [x] Candidate $C_4$ strictly scoped by town and locality to prevent hub-node collisions.
- [x] Deep zero-leakage isolation boundary $\mathbf{Closure}(A^*)$ and `fit_scope` provenance specified.
- [x] Dual validation engines (SCV on 1,477 accounts, LOSO on 54 streets) specified.
- [x] Candidate C3 defined as physical POI anchor coordinate; relational cues act as ranking features and uncertainty expanders.
- [x] Candidate C4 neighbor weights $w_j = \sum_{v \in \mathcal{V}(a_j)} I(v) S(v)$ formally defined.
- [x] Calibration estimated from out-of-fold SCV residuals with minimum sample size fallback hierarchy ($N_{\min}=15$), verified on 19 validation addresses.
- [x] 5 formal specification documents authored, cross-audited, and frozen in workspace root.

**System specifications are fully reconciled, cross-audited, and frozen. Pre-build gate satisfied; ready for Phase 0 execution authorization.**
