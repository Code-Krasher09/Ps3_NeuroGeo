# PS3 Model Specification: Algorithmic & Mathematical Specifications
*Document 4 of 5: Formula Families, Parameter Spaces & Algorithms*
*Status: FROZEN (Model Specification Level)*

---

## 1. Specification Paradigm: Architecture vs. Parameters

We strictly separate **Architectural Formulations** (which are frozen) from **Tuned Parameters** (which must be fit on training/validation splits):

```
┌────────────────────────────────────────────────────────┐
│ ARCHITECTURE (Frozen)                                  │
│ • Bounded monotonic kernels for GPS accuracy           │
│ • Exponential penalty for photo collisions             │
│ • Continuous dwell sigmoid attenuation                 │
│ • Linear utility candidate reranking                   │
│ • Evidence-tier empirical quantile calibration         │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ CONFIGURABLE PARAMETERS (Learned / Grid-Searched)      │
│ • λ_photo (photo collision decay rate)                 │
│ • σ_gps (GPS accuracy scaling factor)                  │
│ • dwell_midpoint, dwell_slope (dwell sigmoid params)   │
│ • w_1 ... w_5 (candidate reranker utility weights)     │
│ • Q_tier (empirical P90 calibration radii per tier)    │
└────────────────────────────────────────────────────────┘
```

---

## 2. Visit Integrity Engine: $I(v)$

Computes the operational trustworthiness of visit $v \in [0.0, 1.0]$:

$$I(v) = \exp\Big(-\lambda_{\text{photo}} \cdot (\text{CollisionCount}_v - 1)\Big) \times \frac{1}{1 + \exp\big(-k_{\text{dwell}} \cdot (\text{dwell}_v - t_{\text{dwell}})\big)} \times \exp\left(-\frac{\text{speed}_v^2}{2 \cdot v_{\text{max}}^2}\right)$$

### Parameter Specifications:
- $\text{CollisionCount}_v$: Frequency of `photo_hash` across all visits in dataset.
- $\lambda_{\text{photo}} \in [1.0, 2.0]$: Collision penalty coefficient (Default baseline: $1.0$).
  - For unique photos ($\text{count}=1$): $e^0 = 1.0$.
  - For photo reused 2 times: $e^{-1.0} = 0.368$.
  - For photo reused 5 times: $e^{-4.0} = 0.018$.
  - For FA009 duplicate clusters ($\ge 26$ collisions): $I(v) < 10^{-10} \approx 0.0$.
- $k_{\text{dwell}} = 0.05$, $t_{\text{dwell}} = 45\text{ seconds}$: Dwell midpoint threshold.
- $v_{\text{max}} = 60.0\text{ km/h}$: Smooth velocity attenuation kernel.

---

## 3. Spatial Evidence Strength: $S(v)$

Computes the doorstep spatial precision of observation $v \in [0.0, 1.0]$:

$$S(v) = w_{\text{outcome}}(v) \times \exp\left(-\frac{\text{gps\_accuracy}_v}{\sigma_{\text{gps}}}\right)$$

### Mathematical Guarantees:
- **Monotonically Decreasing**: As GPS accuracy radius widens ($10\text{m} \to 80\text{m}$), $S(v)$ decays smoothly towards $0$.
- **Strictly Bounded**: $S(v) \in [0.0, 1.0]$ for all possible inputs (eliminates negative value bugs).

### Parameter Specifications:
- $\sigma_{\text{gps}} \in [20.0, 35.0]$ meters (Default baseline: $25.0\text{m}$).
- Outcome Weights $w_{\text{outcome}}(v)$:
  - `met_family`: $1.00$ (Doorstep confirmed by spouse/parent; median surveyed error $15.9\text{m}$).
  - `locked_premises`: $0.95$ (Doorstep confirmed; borrower absent; median error $20.2\text{m}$).
  - `met_borrower`: $0.85$ (High accuracy; occasional meetings at roadside shops; median error $34.3\text{m}$).
  - `cash_collected`: $0.90$ (Receipt generated at premises).
  - `neighbour_says_shifted`: $0.40$ (Address-state evidence; neighbor confirmed adjacent house; median error $20.1\text{m}$).
  - `no_such_person`: $0.00$ (Doorstep disconfirmed).
  - `address_not_traceable`: Evaluated separately via Negative Search Penalty (see Section 6).

---

## 4. Trajectory Feature Extractor

From `visit_gps_points.csv`, for each visit $v$, extract:
1. **Path Tortuosity $\tau_v$**:
   $$\tau_v = \frac{\sum_{i=1}^{N_v-1} \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}}{\sqrt{(x_{N_v} - x_1)^2 + (y_{N_v} - y_1)^2} + \epsilon}$$
   - *Tunable Operational Thresholds*: Initial candidate boundaries are $\tau_v \le 2.5$ (direct transit) and $\tau_v > 3.0$ (wandering / searching loop), subject to empirical calibration.
   - *Trajectory Preprocessing & Numerical Guards*:
     - **Stationary Jitter Guard**: If net displacement $\le 15\text{m}$, or total points $N_v < 3$, or mean horizontal accuracy $> 50\text{m}$, tortuosity is ill-conditioned due to GPS drift; assign default neutral value $\tau_v = 1.0$.
     - **Speed Jump Filter**: Points with instantaneous implied speed $> 40\text{ m/s}$ ($144\text{ km/h}$) are filtered out as GPS multi-path anomalies prior to path summation.
2. **Radius of Gyration $R_{g,v}$**:
   $$R_{g,v} = \sqrt{\frac{1}{N_v} \sum_{i=1}^{N_v} \left( (x_i - \bar{x})^2 + (y_i - \bar{y})^2 \right)}$$
3. **Doorstep Dwell Medoid $(x_{\text{med}}, y_{\text{med}})$**:
   - Filter GPS points where instantaneous speed $< 1.0\text{ m/s}$ during the check-in stationary window ($\pm 5\text{ mins}$ of `checkin_ts`).
   - Compute spatial median of stationary points. If no stationary window exists, fall back to `(checkin_x, checkin_y)`.

---

## 5. Candidate Generation Engine: $\mathcal{C}(a)$

For any target address $a$, generate up to 5 spatial candidates:

```
Candidate Set C(a)
├── C1: Commercial Baseline Prior (baseline_geocodes.csv)
├── C2: Doorstep Dwell Medoid (if Path A: Visited)
├── C3: POI Anchor Coordinate (if landmark resolved)
├── C4: Hierarchically Scoped Neighbor Street Centroid (Path B: Cold Start)
└── C5: Locality Geometric Centroid (localities.csv fallback)
```

### Mathematical Definitions of Candidates:
- **Candidate $C_1$ (Commercial Baseline)**: $C_1 = (\text{geocoder\_x}, \text{geocoder\_y})$ directly from `baseline_geocodes.csv`.
- **Candidate $C_2$ (Doorstep Dwell Medoid)**: $C_2 = (x_{\text{med}}, y_{\text{med}})$ computed across all clean, high-integrity visits to address $a$.
- **Candidate $C_3$ (POI Anchor Coordinate)**:
  $$C_3 = (x_{\text{poi}}, y_{\text{poi}})$$
  $C_3$ is the physical coordinate of the resolved landmark from `landmarks_poi.csv`.
  *Relational Semantics*: Spatial prepositions (e.g., *"behind"*, *"opposite"*, *"beside"*) are **not** mapped to arbitrary heuristic coordinate displacements in V1. Instead, the extracted relation acts as an evidence match score in $\text{TextMatch}(a, C_3)$ and modulates the confidence radius. (Learned metric offset displacements $C_{3.1}$ remain an empirical post-V1 candidate extension).
- **Candidate $C_4$ (Hierarchically Scoped Neighbor Street Centroid)**:
  $$C_4 = \frac{\sum_{j \in \mathcal{N}(a)} w_j \cdot \mathbf{X}_j}{\sum_{j \in \mathcal{N}(a)} w_j}$$
  where:
  - $\mathcal{N}(a) = \{ \text{Visited accounts } j \mid \text{town}_j = \text{town}_a \land \text{locality}_j = \text{locality}_a \land \text{street}_j = \text{street}_a \}$.
  - $\mathbf{X}_j$ is the Doorstep Dwell Medoid coordinate of neighbor address $j$.
  - The neighbor weight $w_j$ is formally defined via the Information Triad:
    $$w_j = \sum_{v \in \mathcal{V}(a_j)} I(v) \cdot S(v)$$
    representing the cumulative integrity-weighted spatial evidence on neighbor account $a_j$.
  - If $|\mathcal{N}(a)| = 0$, Candidate $C_4$ is **not generated**.
- **Candidate $C_5$ (Locality Centroid)**: $C_5 = (\text{centroid\_x}, \text{centroid\_y})$ from `localities.csv`.

---

## 6. Candidate Scorer & Reranker

Each candidate $C_k = (x_k, y_k)$ is evaluated using an interpretable linear utility function:

$$\text{Score}(C_k) = w_{\text{text}} \cdot \text{TextMatch}(a, C_k) + w_{\text{visit}} \cdot \text{DirectEvidence}(C_k) + w_{\text{neigh}} \cdot \text{NeighborSupport}(C_k) + w_{\text{base}} \cdot \text{BaselineAgreement}(C_k) - w_{\text{neg}} \cdot \text{NegativeSearchPenalty}(C_k)$$

### Feature Definitions:
1. $\text{TextMatch}(a, C_k) \in [0, 1]$: Token and character n-gram match between address entities and candidate source.
2. $\text{DirectEvidence}(C_k) = \sum_{v \in \mathcal{V}(a)} I(v) \cdot S(v) \cdot \exp\left(-\frac{\|C_k - \mathbf{X}_v\|_2^2}{2 \cdot 25^2}\right)$.
3. $\text{NeighborSupport}(C_k) = \sum_{j \in \mathcal{N}(a)} \exp\left(-\frac{\|C_k - \mathbf{X}_j\|_2^2}{2 \cdot 50^2}\right)$.
4. $\text{BaselineAgreement}(C_k) = \mathbb{I}(\text{precision} \in \{\text{'street'}, \text{'rooftop'}\}) \times \exp\left(-\frac{\|C_k - \mathbf{X}_{\text{base}}\|_2^2}{2 \cdot 100^2}\right)$.
5. $\text{NegativeSearchPenalty}(C_k) = \sum_{v \in \mathcal{V}_{\text{not\_traceable}}} I(v) \cdot \mathbb{I}(\|C_k - \mathbf{X}_v\|_2 \le 100\text{m}) \cdot \min(1.0, \tau_v / 3.0)$.

### Final Coordinate Output:
$$\hat{\mathbf{X}} = \arg\max_{C_k \in \mathcal{C}(a)} \text{Score}(C_k)$$

---

## 7. Empirical Quantile Uncertainty Calibration

The confidence radius $R$ is calibrated to target approximately 90% empirical coverage on held-out data:

1. **Evidence Strata Classification**:
   - **Tier 1 (`DOORSTEP_VERIFIED`)**: Target address has $\ge 1$ visit with $I(v) \ge 0.70$ and $w_{\text{outcome}} \ge 0.85$. (Initial target $R \le 35\text{m}$).
   - **Tier 2 (`STREET_RESOLVED`)**: Cold-start address; Candidate $C_4$ selected with $\ge 2$ supporting neighbor accounts, or $C_3$ anchored to resolved landmark. (Initial target $R \le 85\text{m}$).
   - **Tier 3 (`LOCALITY_BOUNDED`)**: Cold-start address; Candidate $C_5$ (Locality Centroid) selected. (Initial target $R \le 250\text{m}$).
   - **Tier 4 (`BASELINE_FALLBACK`)**: Commercial baseline geocoder selected with low precision (`locality` or `pincode`). (Initial target $R > 400\text{m}$).

2. **Calibration Sample Isolation**:
   - Calibration radii $R_T$ are estimated from out-of-fold calibration episodes generated during Simulated Cold-Start Validation (SCV) across $N = 1,477$ visited accounts, strictly disjoint from reranker weight fitting.
   - The 19 benchmark validation addresses independently verify empirical coverage rather than simultaneously fitting reranker weights and estimating quantiles.

3. **Sample-Size Fallback Hierarchy**:
   To prevent degenerate quantiles on tiers with small sample sizes ($N_T < N_{\text{min}} = 15$):
   ```python
   def get_calibrated_radius(tier: str, residuals_by_tier: dict, N_min: int = 15) -> float:
       # Level 1: Tier-specific empirical P90
       if len(residuals_by_tier.get(tier, [])) >= N_min:
           return float(np.quantile(residuals_by_tier[tier], 0.90))
       
       # Level 2: Evidence-Family pooled P90 (Visited Family: T1 vs Cold-Start Family: T2, T3, T4)
       family = "visited" if tier == "DOORSTEP_VERIFIED" else "cold_start"
       family_residuals = [r for t, res in residuals_by_tier.items() 
                           if ("visited" if t == "DOORSTEP_VERIFIED" else "cold_start") == family 
                           for r in res]
       if len(family_residuals) >= N_min:
           return float(np.quantile(family_residuals, 0.90))
       
       # Level 3: Global pooled empirical P90
       global_residuals = [r for res in residuals_by_tier.values() for r in res]
       return float(np.quantile(global_residuals, 0.90))
   ```

4. **Coverage Target**:
   Assign $R_a = \text{get\_calibrated\_radius}(\text{Tier}(a))$. The radius is calibrated to target approximately 90% empirical coverage on held-out data.

---

## 8. Deterministic Offline Navigation Synthesizer

Generates auditable, natural language guidance for field app display without network calls:

```python
def generate_navigation_guidance(entity: ParsedAddressEntity, resolved_poi_name: str) -> str:
    parts = []
    if entity.street_normalized:
        parts.append(f"Follow {entity.street_normalized}")
    if resolved_poi_name:
        relation = entity.spatial_relation or "near"
        parts.append(f"house is situated {relation} {resolved_poi_name}")
    if entity.locality_name_raw:
        parts.append(f"in {entity.locality_name_raw}")
    return ". ".join(parts) + "."
```
*Example Output*: `"Follow 6th Cross, 5th Main. House is situated near Church in Kuvempu Layout."*
