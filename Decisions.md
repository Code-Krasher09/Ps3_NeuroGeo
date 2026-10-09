# NeuroGeo-Field: Architectural & Implementation Decisions Log
*Authority: System Design Authority & Methodological Rationale*
*Location: Repository Root (`Decisions.md`)*

This document maintains an immutable, chronological register of engineering, algorithmic, and methodological decisions taken during the design and implementation of the **NeuroGeo-Field** engine for Problem Statement 3 (PS3).

---

## 1. System-Level Architecture Decisions

### Decision 0.1: Retrieve → Rank → Calibrate over Continuous Coordinate Regression
- **Action**: Formulate geocoding as discrete spatial candidate generation ($C_1$ to $C_5$), linear evidence reranking, and empirical uncertainty calibration, explicitly rejecting continuous neural regression $(x, y) = f(\text{address\_text})$.
- **Reason & Intuition**: Neural network regression in unconstrained 2D Euclidean space frequently outputs hallucinations outside administrative town limits or between disjoint settlements. By generating candidates anchored strictly to verified physical entities (commercial priors, visit medoids, resolved POIs, neighbor clusters, locality centroids), every prediction is mathematically bounded within the valid geographic domain.

### Decision 0.2: Dual-Path Inference Engine (Visited vs. Cold-Start)
- **Action**: Split inference into **Path A** (Direct Doorstep Evidence Pipeline for the 47.4% visited addresses) and **Path B** (Spatial Knowledge Transfer Pipeline for the 52.6% cold-start unvisited addresses).
- **Reason & Intuition**: Empirical audit revealed that 52.6% of addresses in `addresses.csv` (1,640 of 3,117) and 51 of the 100 benchmark surveyed addresses have exactly zero visit history. A single-path pipeline would either leak visit data or fail to generalize to unvisited addresses.

---

## 2. Phase 0: Evaluation Harness & Zero-Leakage Decisions

### Decision 0.3: Deep Isolation Closure $\mathbf{Closure}(A^*)$
- **Action**: Enforce that when evaluating target address $A^*$, the system purges not just direct visits to $A^*$, but all records derived from $A^*$ across neighbor graphs, street centroids, and calibration sets:
  $$\mathcal{D}_{\text{visible}}(A^*) = \mathcal{D}_{\text{full}} \setminus \mathbf{Closure}(A^*)$$
- **Reason & Intuition**: Standard random splitting leaks information in spatial graphs. If address $A^*$ contributes visits to a street centroid or neighbor cluster, predicting $A^*$ tests memory rather than spatial knowledge transfer.

### Decision 0.4: Separation of Model Weight Fitting from Uncertainty Calibration
- **Action**: The 19 validation benchmark addresses are used solely to verify model choices and check calibrated coverage ($\ge 88\%$). Calibration radii ($R_T$) are estimated strictly from out-of-fold calibration episodes generated during Simulated Cold-Start Validation (SCV, $N = 1,477$).
- **Reason & Intuition**: Using the same 19 validation addresses to simultaneously fit reranker weights and estimate P90 residual quantiles is statistically invalid and produces artificially optimistic uncertainty estimates.

### Decision 0.5: Fit-Scope Provenance Tags
- **Action**: Every intermediate artifact must carry an immutable provenance string tag (e.g., `unsupervised_all`, `train_fold_{k}`, `calibration_oof`).
- **Reason & Intuition**: Enforces machine-verifiable assertions preventing training artifacts from contaminating out-of-fold test evaluations.

---

## 3. Phase 1: Operational Sanitizer & Trajectory Mining Decisions

### Decision 1.1: Information Triad Formulation ($I(v)$ and $S(v)$ Decoupled)
- **Action**: Decouple operational visit trustworthiness $I(v) \in [0, 1]$ from doorstep spatial precision $S(v) \in [0, 1]$.
  - $I(v)$ evaluates device physics, photo duplication fraud, dwell times, and velocity.
  - $S(v)$ evaluates GPS accuracy radius and outcome semantics.
- **Reason & Intuition**: An agent can stand at the exact true doorstep for 10 minutes with 5m GPS accuracy (high spatial precision), but upload a duplicated stock photo to game incentives (compromised integrity). Conversely, an honest check-in might have poor GPS signal due to dense foliage. Multiplying $I(v) \cdot S(v)$ prevents untrustworthy data from contaminating spatial graphs while rewarding clean observations.

### Decision 1.2: Exponential Photo Collision Decay Kernel
- **Action**: Replace binary fraud thresholding with a smooth exponential decay:
  $$\text{PhotoPenalty}(v) = \exp\Big(-\lambda_{\text{photo}} \cdot (\text{CollisionCount}_v - 1)\Big)$$
- **Reason & Intuition**: Binary thresholding creates fragile cliffs. The exponential penalty smoothly preserves unique photos ($c=1 \implies 1.0$), discounts occasional 2-photo overlaps ($c=2 \implies 0.368$), and obliterates syndicates like Agent `FA009` ($c=26 \implies < 10^{-10}$).

### Decision 1.3: Dwell Sigmoid Centered at $t_{\text{dwell}} = 45\text{s}$
- **Action**: Apply a smooth sigmoid centered at 45 seconds with slope $k = 0.05$:
  $$\text{Sigmoid}(d) = \frac{1}{1 + \exp\big(-0.05 \cdot (d - 45)\big)}$$
- **Reason & Intuition**: Empirical audit of `field_visits.csv` showed that fraudulent drive-bys have dwell $< 15\text{s}$, while genuine debtor interactions (`met_family`, `locked_premises`, `met_borrower`) have median dwell of $65\text{s}$ to $180\text{s}$. The 45-second midpoint cleanly attenuates drive-bys without punishing brief doorstep check-ins.

### Decision 1.4: Decoupling `neighbour_says_shifted` as Address-State Evidence
- **Action**: Set $w_{\text{outcome}} = 0.40$ for `neighbour_says_shifted`, but explicitly exclude it from positive doorstep medoid calculations (`is_doorstep_confirmed == False`).
- **Reason & Intuition**: When a neighbor states the borrower shifted, the agent spoke to a neighbor at an adjacent house or street corner. The observation provides critical operational state (borrower no longer resides here), but does not prove the borrower's rooftop coordinates. Treating it as a positive doorstep observation would distort doorstep medoid pinning.

### Decision 1.5: Trajectory Preprocessing & Numerical Guards
- **Action**: Implement three mandatory numerical guards before computing tortuosity $\tau_v$:
  1. Instantaneous GPS velocity jump filter: discard points where implied speed $> 40.0\text{ m/s}$ ($144\text{ km/h}$).
  2. Stationary jitter guard: if net displacement $\le 15.0\text{m}$, set neutral $\tau_v = 1.0$.
  3. Minimum points guard: if $N_v < 3$, set neutral $\tau_v = 1.0$.
- **Reason & Intuition**: Path tortuosity $\tau = \frac{\text{path\_length}}{\text{net\_displacement} + \epsilon}$ diverges when an agent stands stationary in a doorway with GPS multipath jitter. A wandering agent walking 400m around lanes has genuine high tortuosity ($\tau > 3.0$), whereas a stationary agent pacing 3m must not be misclassified as a searching loop.

### Decision 1.6: Weighted Doorstep Medoid vs. Arithmetic Mean
- **Action**: Compute address doorstep coordinates as the spatial medoid among credible visits ($I(v) \ge 0.50$, positive outcomes), weighted by $w_v = I(v) \cdot S(v)$.
- **Reason & Intuition**: Arithmetic means ($(\bar{x}, \bar{y})$) are sensitive to outliers. If an address has two genuine visits at $(100, 100)$ and one compromised visit 5 km away, the mean places the pin in the middle of nowhere. The medoid is guaranteed to select a true, physically visited check-in point from the cluster.

### Decision 1.7: Null Photo Hash Isolation
- **Action**: In `compute_photo_collision_counts`, missing or null `photo_hash` values are assigned collision count $= 1$ (no penalty).
- **Reason & Intuition**: Grouping missing photo hashes into a shared cluster would falsely penalize every legitimate visit that lacked a photo hash as colliding with hundreds of other visits. Each missing photo is treated as an isolated observation.

---

## 4. Phase 2: Multilingual Address Parser & POI Resolver Decisions

### Decision 2.1: Non-Destructive Multilingual Normalization
- **Action**: Implement NFKC Unicode normalization, case-folding for Latin characters, whitespace standardization, and repeated-character collapse ($\ge 3$), while strictly preserving native Kannada (`\u0C80`–`\u0CFF`) and Devanagari (`\u0900`–`\u097F`) Unicode character sequences alongside Latin transliterations.
- **Reason & Intuition**: Many Indian NLP pipelines attempt naive Romanization that irreversibly corrupts native script into ambiguous phonetic representations. Preserving both native script entities (e.g., `ಬಸ್ ನಿಲ್ದಾಣ`, `राशन की दुकान`) and Latin loan tokens guarantees zero loss of linguistic fidelity for subsequent entity extraction.

### Decision 2.2: Town-Scoped Locality Resolution with Specificity Ranking
- **Action**: Scope locality candidate search strictly to `localities.csv` where `town_id == address.town_id`. Use exact normalized substring matching with length priority, backed by windowed character 3-gram Jaccard matching ($\ge 0.65$ threshold) with ambiguity detection.
- **Reason & Intuition**: Locality names are not globally unique across towns (e.g., `Nehru Colony` exists in both `T1` Kaveripura as `T1-L11` and `T2` Devgarh Nagar as `T2-L06`). Evaluating without town context causes cross-town misattribution. Furthermore, ranking by string length guarantees that specific compound names (e.g. "Gandhi Nagar") are preferred over generic substrings (e.g. "Nagar").

### Decision 2.3: Bidirectional Street Normalization and Standardized Ordinals
- **Action**: Standardize street representations so that alternate word orders and abbreviations normalize to identical canonical street keys (e.g., `D Block, 5th Road` and `Blk D, Rd 5` both map to `block d, 5th road`; `6th X, 1st MN` maps to `6th cross, 1st main`).
- **Reason & Intuition**: Street-level knowledge transfer in Phase 3 ($C_4$ neighbor cluster) requires grouping accounts by canonical street identifier. If identical physical streets have mismatched raw text, the bipartite spatial graph fractures into isolated singletons.

### Decision 2.4: Locality-Bounded POI Disambiguation (Zero-Duplicate Invariant)
- **Action**: Scope POI resolution strictly to the bounding disc of the resolved locality ($r \le 800\text{m}$). Within this disc, every landmark type is strictly unique ($\le 1$ POI). If locality is unresolved, mark generic landmark mentions as `ambiguous` rather than arbitrarily selecting a candidate across the town.
- **Reason & Intuition**: In Indian towns, landmarks like "Bus Stop" or "Ration Shop" occur in every neighborhood. Without locality bounding, picking one bus stop among 7 in a town is a blind guess with 2 km error. Locality bounding converts an intractable global disambiguation problem into a deterministic local lookup.

### Decision 2.5: Relational Terms as Pure Semantic Evidence (Zero Heuristic Coordinate Offsets)
- **Action**: Extract relational prepositions (`behind`, `opposite`, `near`, `beside`, `hinde`, `eduru`, `ke paas`, etc.) and store them as semantic evidence tags in `ParsedAddressEntity`. Candidate $C_3$ coordinate is strictly the physical POI catalog coordinate from `landmarks_poi.csv` with zero arbitrary displacement.
- **Reason & Intuition**: Heuristically shifting a coordinate (e.g. adding $-50\text{m}$ for "behind") is mathematically ungrounded because "behind" depends on the unknown orientation, street frontage, and cadastral boundary of the landmark. Fabricated displacements increase error rather than reducing it. Relational evidence is properly used during candidate ranking and uncertainty radius expansion in Phase 4.

### Decision 2.6: Precomputed Landmark Type Similarities for Sub-Second Processing
- **Action**: Factor textual matching into a 14-category type-level similarity evaluation rather than looping over all POIs and aliases per address.
- **Reason & Intuition**: All POIs belonging to a category share identical textual semantics. Computing category-level similarities once per address accelerates batch resolution from 75 seconds to $< 7$ seconds (450+ addresses/sec) while preserving exact match determinism.

---

## 5. Phase 3: Hierarchical Spatial Graph & Cold-Start Validation Decisions

### Decision 3.1: Deterministic Hierarchical Scoping over Global Street Matching
- **Action**: Scope the spatial graph strictly in a deterministic 3-level hierarchy: $\text{Town} \to \text{Locality} \to \text{Street}$, with canonical composite keys `Town::Locality::NormalizedStreet`.
- **Reason & Intuition**: Street names in Indian municipalities repeat frequently across localities (e.g., "1st Cross, 2nd Main" or "Station Road" occurs in multiple distinct wards). A flat global street match across the entire town would conflate accounts situated kilometers apart into a single corrupt cluster. Hierarchical scoping guarantees zero cross-locality hub node collisions and enforces that all node identifiers start with the town prefix `'T'`.

### Decision 3.2: Cumulative Information Triad Neighbor Weighting: $w_j = \sum_{v \in \mathcal{V}(a_j)} I(v) \cdot S(v)$
- **Action**: Weight each neighbor account $a_j$ by the exact cumulative sum of its operational visit integrity and spatial evidence strength:
  $$w_j = \sum_{v \in \mathcal{V}(a_j)} I(v) \cdot S(v)$$
  rather than naive uniform account averaging ($w_j = 1$) or learned parameter weights.
- **Reason & Intuition**: Neighbor accounts vary widely in evidentiary trustworthiness: an account verified three times with high-integrity positive doorstep outcomes (`met_family`, $I \ge 0.85$, $S \ge 0.85$) provides an immensely stronger spatial anchor than an account visited once with marginal GPS accuracy. The Information Triad weight naturally pulls Candidate $C_4$ toward rigorously verified doorstep locations while filtering noise.

### Decision 3.3: Strict Unavailability Protocol for Candidate $C_4$ (Zero Fallback Inside $C_4$)
- **Action**: If an address lacks a resolved locality, lacks a normalized street, or has zero eligible visited neighbors on that street segment ($|\mathcal{N}(a)| = 0$), Candidate $C_4$ is explicitly omitted (`None`). It does NOT fabricate a fallback coordinate inside $C_4$.
- **Reason & Intuition**: Conflating candidate generation with candidate fallback violates modular separation. If $C_4$ secretly defaulted to a locality centroid or town center when neighbors were missing, downstream candidate ranking and evidence-tier uncertainty calibration would misclassify a weak locality guess as a confident street-level prediction. $C_4$ must represent strictly street-level neighbor knowledge transfer.

### Decision 3.4: $O(1)$ Leave-One-Out Isolation Boundary for SCV
- **Action**: During Simulated Cold-Start Validation across $N = 1,477$ visited accounts, precompute street-level cumulative totals $(\sum w, \sum w x, \sum w y)$ and subtract target account $a_i$'s own contribution $(w_i, w_i x_i, w_i y_i)$ to compute $C_4(a_i)$:
  $$C_4(a_i) = \left( \frac{\sum_{j \ne i} w_j x_j}{\sum_{j \ne i} w_j}, \frac{\sum_{j \ne i} w_j y_j}{\sum_{j \ne i} w_j} \right)$$
- **Reason & Intuition**: Evaluating 1,477 leave-one-out models via naive graph reconstruction would take minutes. Because the weighted centroid formula is linear, subtracting the target observation's weight and moment yields the exact out-of-fold prediction in $O(1)$ time while guaranteeing zero self-contamination.

### Decision 3.5: Multi-Account Street Masking for Leave-One-Street-Out (LOSO)
- **Action**: In LOSO evaluation across eligible streets ($\ge 5$ visited accounts), simultaneously mask ALL visited accounts on the held-out street from the graph. Confirm that $C_4$ availability is strictly $0.0\%$ and evaluate non-learned spatial baselines (Locality Centroid $C_5$ and Commercial Baseline $C_1$).
- **Reason & Intuition**: While SCV evaluates spatial interpolation along a known street segment with existing visited neighbors, LOSO stress-tests spatial extrapolation to a completely unvisited street where zero direct or neighbor doorstep evidence exists. Masking the entire street simultaneously ensures that spatial knowledge transfer bounds are rigorously characterized.

### Decision 3.6: Immutable `fit_scope` Provenance Enforcement
- **Action**: Every graph artifact (`neighbor_graph.parquet`, `street_centroids.parquet`) carries an immutable provenance tag (`production_all`, `train_fold_{k}`). Functions assert validity against `VALID_FIT_SCOPE_PATTERNS`.
- **Reason & Intuition**: Geographic models are prone to silent leakage where global summaries generated from all data are accidentally evaluated out-of-fold. Programmatic fit_scope validation prevents cross-fold contamination at runtime.

### Decision 3.7: Production (`production_all`) vs. Evaluation Fold Provenance Separation
- **Action**: Distinguish full-corpus production graph artifacts (`fit_scope="production_all"`) from evaluation artifacts (`fit_scope="train_fold_{k}"` for LOSO, and `evaluation_scope="scv_leave_one_out"` with `graph_fit_scope="scv_leave_one_out"` for SCV). Furthermore, designate the 1,477 SCV residuals with `reserved_for_calibration = True` ("leakage-safe SCV residuals reserved for empirical calibration"), explicitly separating them from Phase 4 model parameter fitting.
- **Reason & Intuition**: Declaring `train_all` on evaluation outputs masks the fact that evaluation must be out-of-fold. An evaluation output must explicitly declare that the target was masked during prediction, preserving full auditability.

### Decision 3.8: Empirical Resolution of the LOSO 54-vs-13-vs-17 Eligibility Discrepancy
- **Action**: Formally audit the exact frozen contract definition of LOSO eligibility (`PS3_EVALUATION_AND_LEAKAGE_CONTRACT.md` Section 5: "streets with $\ge 5$ visited accounts") across all potential clustering dimensions in the empirical data:
  1. Canonical street keys with $\ge 5$ ALL portfolio addresses: **52**
  2. Canonical street keys with $\ge 5$ VISITED addresses: **13** (80 accounts)
  3. Canonical street keys with $\ge 5$ CREDIBLE DOORSTEP addresses: **11**
  4. Canonical street keys with $\ge 5$ distinct VISITED ACCOUNTS: **17** (102 accounts)
  5. Total unique visited accounts overall: **1,339**
  6. Total unique visited addresses overall: **1,477**
- **Reason & Intuition**: The approximate "~54 streets" in Document 3 was derived from exploratory portfolio-wide address clustering (which yields 52 canonical streets with $\ge 5$ total addresses). The prior implementation used visited address count ($\ge 5$ visited addresses), yielding 13 streets. However, the frozen contract explicitly specifies *visited accounts* (`account_id`), which dynamically and cleanly evaluates to **17 eligible streets**. Rather than forcing 54 or relying on address count, the implementation now strictly adheres to the frozen account-level contract unit.

### Decision 3.9: Strict Account-Level LOSO Masking Across the Entire Portfolio
- **Action**: For each of the 17 LOSO folds ($k=1 \dots 17$):
  1. Identify all distinct visited accounts on held-out street $S_j$ (`held_out_account_count`, totaling 102 accounts across the 17 folds).
  2. Identify all addresses belonging to those accounts both on $S_j$ (102 addresses) and across the entire portfolio (136 addresses).
  3. Purge ALL visits, GPS breadcrumbs, and doorstep medoids belonging to those accounts, as well as all addresses on street $S_j$, from the training graph for fold $k$.
  4. Enforce that street $S_j$ has zero neighbor accounts, zero weight, and no learned centroid in `train_graph` (`fit_scope="train_fold_{k}"`), ensuring Candidate $C_4$ availability is strictly $0.0\%$.
  5. Clearly distinguish in output tables: `street_key`, `fold_id`, `fit_scope`, `held_out_account_count`, `held_out_address_count`, `evaluated_episode_count` (102 episodes), and `evaluated_medoid_episode_count` (96 episodes with ground-truth medoids).
- **Reason & Intuition**: If a held-out account owns multiple addresses, leaving an address on another street as training evidence could leak the account's operational check-in pattern into locality or street clusters. Purging the entire account portfolio-wide ensures the held-out street is genuinely and completely unvisited.

---

## Phase 4: Candidate Generation, Deterministic Linear Reranking & Empirical Quantile Calibration

### Decision 4.1: Strict Candidate Schema and Zero Coordinate Fabrication
- **Action**: Enforce that every candidate hypothesis ($C_1 \dots C_5$) maintains an explicit availability flag, exact Cartesian coordinates $(x, y)$, metadata, provenance `fit_scope`, and raw feature vectors. If evidence is missing (e.g. no credible doorstep for $C_2$, no resolved POI for $C_3$, unresolved street or no neighbors for $C_4$, unresolved locality for $C_5$), the candidate is marked `available = False` and coordinates are set strictly to `None`. Under no circumstances are fallback coordinates substituted into $C_2$, $C_3$, or $C_4$.
- **Reason & Intuition**: Coordinate fabrication creates phantom evidence that corrupts downstream reranking and uncertainty calibration. Preserving explicit unavailability allows the reranker to distinguish between absent evidence and zero-weight evidence.

### Decision 4.2: Physical POI Coordinates Preserved Without Arbitrary Preposition Offsets
- **Action**: Candidate $C_3$ adopts the exact physical coordinate of the resolved landmark from `landmarks_poi.csv`. Relational prepositions (*"behind"*, *"opposite"*, *"beside"*, *"near"*) modulate textual compatibility scores in `TextMatch` and status tier ambiguity, but do NOT displace the physical coordinate.
- **Reason & Intuition**: In the absence of building footprints or vector road centerlines, heuristic vector displacements (e.g. adding $+25\text{m}$ for "behind") introduce uncontrolled spatial errors and degrade accuracy. Keeping $C_3$ anchored to the physical POI preserves ground-truth landmark precision.

### Decision 4.3: Deterministic Coordinate Deduplication ($\Delta \le 5.0\text{m}$) with Multi-Source Agreement
- **Action**: If multiple candidate types resolve to effectively identical coordinates (distance $\le 5.0\text{m}$), they are deterministically collapsed into a single candidate pin. The retained pin adopts the type with the highest prior evidence priority ($C_2 > C_4 > C_3 > C_1 > C_5$), while `agreed_sources` preserves all merged types (e.g. `("C4", "C1")`). Candidates belonging to the same locality that are $> 5.0\text{m}$ apart remain strictly distinct.
- **Reason & Intuition**: When commercial baseline and spatial graph neighbor clusters point to the same doorstep, treating them as separate competing candidates creates split-vote artifacts and risk of double counting. Deduplication consolidates agreement into a unified evidence pin.

### Decision 4.4: Interpretable Bounded Linear Utility Scorer
- **Action**: Candidate utility is computed via the frozen linear formulation:
  $$\text{Score}(C_k) = w_{\text{text}} \cdot \text{TextMatch} + w_{\text{visit}} \cdot \text{DirectEvidence} + w_{\text{neigh}} \cdot \text{NeighborSupport} + w_{\text{base}} \cdot \text{BaselineAgreement} - w_{\text{neg}} \cdot \text{NegativeSearchPenalty}$$
  where all raw features are deterministically bounded in $[0.0, 1.0]$. The feature `neighbour_says_shifted` is treated as address-state negative evidence (borrower no longer at address) and contributes strictly to `NegativeSearchPenalty`, never to positive doorstep evidence.
- **Reason & Intuition**: Complex non-linear neural rerankers risk overfitting to benchmark anomalies and lack field explainability. A linear utility model guarantees auditability, monotonicity, and clear attribution in evidence cards.

### Decision 4.5: Permitted Training-Set-Only Parameter Fitting (Train $N=66$)
- **Action**: Reranker weights are fitted exclusively on the 66 Train benchmark addresses using deterministic pairwise soft-margin ranking loss (`L-BFGS-B`, `random_state=42`), optimizing candidate ranking preferences against surveyed ground truth on Train only. The 19 Validation and 15 Test benchmark addresses are strictly excluded from parameter fitting.
- **Reason & Intuition**: Fitting or tuning weights against test benchmark addresses causes severe data snooping. Fitting strictly on Train $N=66$ and recording provenance `fit_scope="benchmark_train"` ensures the benchmark test evaluation is 100% untouched.

### Decision 4.6: Evidence-Tier Empirical Quantile Calibration with $N_{\min}=15$ Fallback Hierarchy
- **Action**: Calibration radii $R_T$ are computed as the empirical 90th percentile (P90) residual from the 1,477 held-out SCV leave-one-out episodes (`reports/metrics/scv_results.csv`, `reserved_for_calibration = True`), excluding all benchmark test accounts ($N=15$). For tiers with sample size $N_T < 15$:
  - Level 1: Tier-specific empirical P90 (used for `STREET_RESOLVED` $N=283$, `LOCALITY_BOUNDED` $N=1284$, `BASELINE_FALLBACK` $N=1457$).
  - Level 2: Evidence-Family pooled P90 (Visited family vs. Cold-Start family).
  - Level 3: Global pooled empirical P90 (used for `DOORSTEP_VERIFIED` where SCV masked all target visits, yielding $R=642.25\text{m}$).
- **Reason & Intuition**: SCV residuals represent honest leakage-safe held-out SCV residuals where target visit history was masked. The 3-level fallback hierarchy prevents small-sample degenerate quantiles while maintaining valid empirical coverage ($\ge 90\%$).

### Decision 4.7: Separation of Ablation Ladder Stages (M0 to M6)
- **Action**: Implement the exact 7-stage ablation progression:
  - $M_0$: Commercial Baseline only ($C_1$)
  - $M_1$: + Raw Direct Visit Check-ins (unweighted)
  - $M_2$: + Integrity-Weighted Doorstep Medoids ($C_2$)
  - $M_3$: + Doorstep Medoids + NLP/POI Anchor ($C_3$)
  - $M_4$: + Doorstep Medoids + POI + Spatial Graph Centroid ($C_4$)
  - $M_5$: + Learned Deterministic Linear Reranker ($w_{\text{neg}}=0$)
  - $M_6$: + Negative Search Penalty ($w_{\text{neg}} > 0$)
- **Reason & Intuition**: Each stage isolates the exact marginal accuracy lift of an architectural component. Evaluating both Visited and Cold-Start segments across the ladder provides definitive empirical evidence of where accuracy gains originate.

### Decision 4.8: Phase 4 Correction Pass — Calibration Provenance, Split Isolation & Address-Level Disjointness
- **Action**:
  1. Enforced strict address-level disjointness across all benchmark splits:
     $$\text{calibration\_address\_ids} \cap \text{train\_benchmark\_address\_ids} = \emptyset$$
     $$\text{calibration\_address\_ids} \cap \text{val\_benchmark\_address\_ids} = \emptyset$$
     $$\text{calibration\_address\_ids} \cap \text{test\_benchmark\_address\_ids} = \emptyset$$
     All 100 benchmark addresses (66 Train, 19 Validation, 15 Test) are strictly excluded from calibration fitting, purging all 49 overlapping visited benchmark addresses and retaining exactly 1,428 clean, non-benchmark SCV visited-address episodes.
  2. Provenance disjointness guarantee:
     $$\text{calibration residual source} \cap \text{reranker-fitting target labels} = \emptyset$$
     Reranker weights are fitted exclusively on the 66 Train addresses (using their surveyed coordinates). Calibration residuals are estimated exclusively on the 1,428 non-benchmark SCV episodes (using their held-out doorstep medoids).
  3. Terminology clarification:
     Replaced legacy "visited accounts" with frozen contract terminology:
     - 1,477 SCV visited-address episodes
     - 1,339 unique visited accounts
  4. Residual vs. Episode Accounting:
     Explicitly track both `unique_episode_count` and `residual_count` in `reports/metrics/calibration_quantiles.json` across all tiers:
     - `STREET_RESOLVED` ($C_4$): 279 unique episodes, 279 residuals ($R=615.93\text{m}$, Level 1 `tier_direct`, family `cold_start`).
     - `LOCALITY_BOUNDED` ($C_5$): 1,258 unique episodes, 1,258 residuals ($R=559.95\text{m}$, Level 1 `tier_direct`, family `cold_start`).
     - `BASELINE_FALLBACK` ($C_1$): 1,428 unique episodes, 1,428 residuals ($R=982.55\text{m}$, Level 1 `tier_direct`, family `cold_start`).
     - `DOORSTEP_VERIFIED` ($C_2$): 0 direct residuals (since target visits were masked out during SCV to avoid trivial circular leakage). Triggers Level 3 fallback (`global_p90`), pooling across all candidate tiers ($279 + 1,258 + 1,428 = 2,965$ residuals across 1,428 unique episodes), yielding $R=642.18\text{m}$.
  5. Invariance of predictions:
     Reranker weights ($w_{\text{text}}=5.0, w_{\text{visit}}=3.6926, w_{\text{neigh}}=1.5, w_{\text{base}}=0.1, w_{\text{neg}}=1.5427$), benchmark predictions, and ablation errors remain 100% numerically identical. Overall empirical coverage remains 91.0% (target ~90%).
  6. Candidate Availability Policy Clarification:
     Test addresses are excluded from candidate selection tuning, candidate-prior fitting, reranker fitting, and calibration estimation. Candidate availability may be measured during the final untouched evaluation for reporting, but no test truth is used to alter availability rules.
- **Reason & Intuition**: Absolute separation between calibration residual sources and reranker-fitting target labels eliminates any possibility of circular information flow or benchmark split cross-talk, while maintaining robust sample sizes ($N \ge 279$) and backward compatibility.

### Decision 5.1: Offline Deterministic Navigation, Vector Geometry & Auditable Evidence Cards
- **Action**:
  1. Implemented offline deterministic navigator and evidence card generator in `src/neurogeo/navigator.py` strictly consuming frozen Phase 4 predictions without altering model behavior, candidate rankings, or predicted coordinates.
  2. Adhered strictly to the repository's native coordinate convention: all address and geocoder coordinates are metric Cartesian coordinates in meters $(x, y)$ centered at $(0, 0)$ per town. Distances are computed via Euclidean vector geometry, bearings via 2D cartographic trigonometry ($+Y$ North, $+X$ East), and directions mapped to a deterministic 8-sector compass partition. Standard geodetic haversine is provided for reference but not conflated with projected metric coordinates.
  3. Enforced status-aware and uncertainty-aware guidance across 4 distinct navigation modes: `DOORSTEP`, `STREET`, `LOCALITY`, and `BASELINE`. Language strictly reflects evidence strength: `LOCALITY_BOUNDED` and `BASELINE_FALLBACK` carry explicit caveats and are never described as verified.
  4. Negative field signals (`neighbour_says_shifted`, `address_not_traceable`) are surfaced as prominent warnings in both navigation output and evidence cards; shifted outcomes are treated strictly as address-state evidence and never converted to positive doorstep pins.
  5. Built auditable evidence cards (`reports/metrics/phase5_evidence_cards.json`) capturing candidate provenance, calibration tier, sample count, score margins, positive/negative evidence, and human-readable field explanations.
  6. Guaranteed Phase 4 immutability: verified 100/100 benchmark predictions are bit-for-bit identical before and after navigation execution. Verified 100% offline operation with zero network calls.
- **Reason & Intuition**: Field agents require clear, actionable, and honest guidance that conveys uncertainty accurately. Offline deterministic vector geometry provides reliable navigation assistance in remote areas with poor connectivity without making unsupported road routing claims.

### Decision 5.2: Final Packaging — Reproducible Master Notebook, Interactive Dashboard & Submission README
- **Action**:
  1. Implemented self-contained, reproducible master pipeline notebook `notebooks/01_master_pipeline.ipynb` containing all 19 standardized presentation sections, loading authoritative artifacts and rendering figures A–H.
  2. Implemented interactive demonstration dashboard `notebooks/02_interactive_dashboard.ipynb` supporting offline inspection of any benchmark address, rendering spatial plots, calibrated confidence radii, navigation instructions, auditable evidence cards, and 4 fixed representative case studies.
  3. Authored comprehensive, judge-facing executive submission document `README_SUBMISSION.md` with full technical specification, Mermaid architecture diagram, executive results table (83.7% error reduction), ablation ladder M0–M6, zero-leakage proofs, failure analysis, operational ROI story, and exact reproducibility commands.
  4. Generated publication-quality figures A through H (`reports/figures/`) covering overall error, ablation ladder, visited vs. cold-start breakdown, candidate availability, status tiers, calibration radii, error CDF, and spatial case study.
  5. Strictly enforced the Packaging Boundary: zero modification to candidate generation, reranker weights, calibration mathematics, navigation logic, evaluation splits, benchmark predictions, or source datasets. Zero introduction of Phase 6+ logic (no web apps, no external geocoders, no LLMs, no online learning).
- **Reason & Intuition**: An exemplary engineering submission must be immediately accessible and transparent to competition evaluators. Programmatic notebook generation and end-to-end execution validation guarantee that all artifacts execute reliably without missing dependency crashes while preserving the integrity of frozen modeling decisions.

---

## 7. Submission Consolidation Decisions

### Decision 6.1: Submission Consolidation — Integrated PIN Parsing, C1–C5 Candidate Pipeline, and Entity-Contract Alignment
- **Action**:
  1. **Standardized PIN Extraction and Disambiguation**: Formalized postal PIN parsing (`extract_and_normalize_pincode`) and PIN-assisted locality disambiguation (`resolve_locality`) as native components of `src/neurogeo/address_parser.py`. The parser performs house/unit number pre-masking, six-digit postal pattern extraction, town-compatibility validation against `localities.csv`, and conservative single-digit repair only when an unambiguous town-valid assignment exists. Conflicting PINs trigger abstention, and localities are never assigned from PIN alone.
  2. **Frozen C1–C5 Candidate Architecture**: Finalized the active spatial candidate set strictly as Candidates $C_1$ through $C_5$:
     - $C_1$: Commercial baseline prior
     - $C_2$: Sanitized direct visit medoid
     - $C_3$: Locality-bounded physical POI catalog anchor
     - $C_4$: Hierarchical street neighbor weighted centroid
     - $C_5$: Locality polygon centroid
  3. **Exclusion of Candidate C6 (Town-Scoped Landmark Hypotheses)**: Formally excluded Candidate $C_6$ from the submission architecture and removed all production code, wiring, and tests across `poi_resolver.py`, `candidate_ranker.py`, `calibrator.py`, `navigator.py`, `schemas.py`, and `run_phase4.py`.
     - *Empirical Evaluation Record*: During controlled validation evaluation on the 19 validation split addresses, Candidate $C_6$ produced exactly **0 wins, 0 losses, and 0 candidate selections**. It introduced resolver complexity without providing incremental spatial accuracy over $C_3$ and $C_4$. Historical experiment logs honestly preserve that $C_6$ was prototyped, rigorously evaluated, and definitively rejected.
  4. **Entity-Contract Alignment in Phase 4**: Corrected the column selection (`ent_cols`) in `scripts/run_phase4.py` to pass all parsed address entity attributes—specifically `poi_resolution_status` and `locality_resolution_status`—into `df_benchmark` and downstream candidate feature extraction for both training and inference.
     - *Causal Attribution*: Baseline `run_phase4.py` previously omitted `poi_resolution_status` from `ent_cols`, causing Candidate $C_3$ to default to `"unresolved"` (capping its `TextMatch` score at 0.65 instead of the intended 0.90/0.95). In the validation audit, supplying the complete entity dictionary allowed $C_3$ to receive its true score, yielding 3 validation improvements (`AD000525`, `AD001315`, `AD002536`). This was caused by restoring the intended POI resolution status to $C_3$, **not** by PIN normalization.
  5. **Evidence-Tier Quantile Calibration Retained**: Preserved the empirical quantile calibration method across status tiers (`DOORSTEP_VERIFIED`, `STREET_RESOLVED`, `LOCALITY_BOUNDED`, `BASELINE_FALLBACK`).
  6. **Future Work Reservation**: Explicitly recorded that geometry-aware local-region confidence calibration is proposed future work and has not been implemented or evaluated in this submission.
- **Reason & Intuition**: Under strict submission constraints, architectural clarity, code hygiene, and strict schema contract alignment are paramount. Eliminating redundant candidate hypotheses reduces maintenance burden while fixing the entity-contract ensures production inference matches the intended mathematical specification.


