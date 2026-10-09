# PS3 Evaluation & Leakage Contract
*Document 3 of 5: Zero-Leakage Protocols, Validation Splits & Evaluation Framework*
*Status: FROZEN (Evaluation Contract Level)*

---

## 1. Threat Model: Second-Order Leakage

Data leakage in spatial systems is insidious because geographic information diffuses through spatial aggregates. If test address $A^*$ is evaluated, leakage occurs not only if $A^*$'s visits are in the training set, but also if $A^*$'s visits were used to compute **any intermediate artifact**:

```
                       INDIRECT LEAKAGE VECTORS
                       
  Test Address A* ───► Visits to A* ───► [Direct Leakage: Standard Split Catches]
         │
         ├──► Refined POI Coordinate ───► [Vector 1: Shifts POI toward test house]
         ├──► Street Centroid ─────────► [Vector 2: Street anchor pulled toward test house]
         ├──► Neighbor Cluster Centroid► [Vector 3: Neighbor candidate centered on test house]
         ├──► Alias-Location Table ────► [Vector 4: Text alias mapped to test coordinate]
         └──► Conformal Calibration Set► [Vector 5: Underestimates test residual error]
```

---

## 2. The Formal Isolation Contract: $\mathbf{Closure}(A^*)$

For any address $A^*$ being predicted during evaluation or validation:
$$\mathcal{D}_{\text{visible}}(A^*) = \mathcal{D}_{\text{full}} \setminus \mathbf{Closure}(A^*)$$

### Exclusion Specification:
$\mathbf{Closure}(A^*)$ strictly defines everything that must be purged:
1. `addresses.csv`: Row for $A^*$ is excluded from neighbor graph construction.
2. `field_visits.csv`: All visits where `address_id == A*` are strictly purged.
3. `visit_gps_points.csv`: All GPS rows associated with $A^*$'s visits are purged.
4. `surveyed_addresses.csv`: Surveyed coordinate $(x_{A^*}, y_{A^*})$ is hidden from all training/calibration.
5. **Derived Gazetteers**: Street centroids, neighbor clusters, and POI refinements must be computed **strictly on $\mathcal{D}_{\text{train}} \setminus \{A^*\}$**.
6. **Calibration Parameters**: Calibration quantiles are estimated strictly from held-out out-of-fold calibration episodes that were NOT used for reranker parameter fitting. Held-out validation benchmark addresses verify empirical coverage rather than simultaneously generating calibration quantiles.

---

## 3. Artifact Provenance Specification: `fit_scope`

Every generated intermediate table, graph, or calibration parameter must carry an immutable `fit_scope` string tag indicating its data provenance:

| Artifact Name | Required `fit_scope` Tag | Permitted Training Inputs |
| :--- | :--- | :--- |
| `street_centroids.parquet` | `fit_scope="train_fold_{k}"` | Visits where `address_id` $\in \text{TrainSplit}_k$ |
| `neighbor_graph.parquet` | `fit_scope="train_fold_{k}"` | Addresses and visits $\in \text{TrainSplit}_k$ |
| `calibration_quantiles.json` | `fit_scope="calibration_oof"` | Held-out out-of-fold calibration residuals (disjoint from reranker fitting) |
| `visit_features.parquet` | `fit_scope="unsupervised_all"`| Computed per-visit without labels |

**Enforcement Rule**: The inference engine will raise an assertion error if a model fitted with `fit_scope="train"` is evaluated on an address whose data contributed to that `fit_scope`.

---

## 4. Official Benchmark Partition (100 Surveyed Addresses)

Cross-referencing `surveyed_addresses.csv` against `splits.csv` yields the official competition partition:

```
Total Surveyed Benchmark Addresses: 100
├── Train Split:      66 addresses (Model development & reranker parameter fitting)
├── Validation Split: 19 addresses (Hyperparameter validation & calibration coverage verification)
└── Test Split:       15 addresses (Strictly untouched until final audit)
```

> **Evaluation Isolation Principle**: The 19 validation benchmark addresses serve strictly as an independent evaluation and verification set for model choices and calibration coverage. They are NOT used to simultaneously fit reranker weights and estimate calibration radii.

### Partition Composition:
- **Visited Addresses**: 49 (Train: 32, Validation: 9, Test: 8)
- **Unvisited Addresses**: 51 (Train: 34, Validation: 10, Test: 7)

---

## 5. Development Validation Engines (SCV & LOSO)

Because the 19 validation benchmark addresses are insufficient for tuning feature weights and building calibration distributions, we establish two **statistically powered development validation engines**:

### 1. Simulated Cold-Start Validation (SCV) [Leave-One-Address-Visit-History-Out]
- **Sample Size**: **1,477 validation episodes** (all visited accounts in `addresses.csv`).
- **Information Boundary**:
  - Target address text $\to$ **Available** for parsing and candidate retrieval.
  - Target historical visits $\mathcal{V}(a_i)$ and GPS breadcrumbs $\to$ **Strictly hidden / masked**.
  - Target surveyed coordinate (if benchmark address) $\to$ **Strictly hidden during inference**.
- **Mechanism**: For account $a_i$, mask all direct visits $\mathcal{V}(a_i)$. Predict $\hat{\mathbf{X}}(a_i)$ using Path B (Cold-Start: neighbor accounts, POIs, baseline prior). Benchmark against the empirical doorstep cluster of $a_i$.
- **Calibration Generation**: Out-of-fold SCV residuals provide a statistically robust sample ($N=1,477$) to estimate empirical P90 radii across evidence tiers, strictly isolated from reranker parameter fitting.
- **Objective**: Evaluates spatial interpolation along known streets and populates the calibration residual distribution.

### 2. Leave-One-Street-Out (LOSO) Spatial Generalization Validation
- **Sample Size**: **54 eligible streets** (streets with $\ge 5$ visited accounts).
- **Mechanism**: For street $S_j$, mask ALL visited accounts on $S_j$ simultaneously. Predict addresses on $S_j$ using only locality centroids, POIs, and baseline priors.
- **Objective**: Evaluates spatial extrapolation to a completely unvisited street segment.

---

## 6. Official Evaluation Metrics & Segmented Reporting Matrix

### Primary Metric: Euclidean Distance Error
$$\text{Error}_i = \sqrt{(\hat{x}_i - x_{\text{surveyed}, i})^2 + (\hat{y}_i - y_{\text{surveyed}, i})^2} \text{ meters}$$

### Secondary Metric: Calibrated Empirical Coverage
$$\text{Coverage} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Error}_i \le R_i)$$
The confidence radius $R$ is calibrated to target approximately 90% empirical coverage on held-out data (target $\ge 88\%$ in validation verification).

### Required Segmented Reporting Table:

| Evaluation Segment | Sample Size ($N$) | Commercial Baseline Error | Model 1: Direct Field Baseline | Model 2: NeuroGeo Full |
| :--- | :---: | :---: | :---: | :---: |
| **All Benchmark Addresses** | 100 | **376.4m median / 839.2m P90** | ~185.0m median | **Target: Minimized Error** |
| **Visited Addresses** | 49 | 285.1m median / 612.4m P90 | **~21.5m median / 38.0m P90** | **High Doorstep Accuracy** |
| **Cold-Start Unvisited** | 51 | 412.0m median / 980.5m P90 | 412.0m (Fallback) | **Spatial Knowledge Lift** |
| • *Street-level precision* | 16 | 108.6m median | — | Refined via landmark offset |
| • *Locality-level precision*| 73 | 385.9m median | — | Anchored to neighbor cluster |
| • *Pincode-level precision* | 10 | 1,375.8m median | — | Pulled from 3+ km away |
| • *Rooftop-level precision* | 1 | 25.6m median | — | Baseline preserved |

---

## 7. The Standard Error Taxonomy

Every prediction residual $\ge 80\text{m}$ on validation sets must be automatically categorized into one of 8 failure modes:

| Error Category | Criteria | Diagnostic Remediation |
| :--- | :--- | :--- |
| `PARSER_AMBIGUOUS_STREET` | Extracted street token matches $>3$ streets in town | Refine regex parser word boundaries |
| `POI_UNRESOLVED` | Address mentions landmark, but no match in `landmarks_poi` | Expand alias dictionary |
| `SPATIAL_SPARSITY` | Cold-start address with zero neighbor accounts on street | Rely on baseline prior / expand search radius |
| `CONTRADICTORY_VISITS` | Historical check-ins on address are $>200\text{m}$ apart | Investigate multi-office / moved borrower |
| `FRAUD_POISONED_VISIT` | High check-in error caused by undetected photo collision | Tighten integrity penalty parameter $\lambda_{\text{photo}}$ |
| `ERRONEOUS_BASELINE` | Commercial baseline $>2\text{ km}$ away from locality centroid | Dampen baseline prior weight |
| `NEGATIVE_SEARCH_PENALTY` | All candidates penalized by `address_not_traceable` visits | Fall back to street centroid |
| `GROUND_TRUTH_DISCREPANCY`| Surveyed point outside coordinate sanity bound | Flag for ground truth data audit |
