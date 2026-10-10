# NeuroGeo-Field: Comprehensive Architecture & Data Flow

This document provides a deep, detailed explanation of the NeuroGeo-Field Spatial Inference Engine, visualizing how data flows through the system, how values are transformed, and the sources of each data point.

## High-Level Architecture Overview

The system rejects continuous coordinate regression and instead uses a **Retrieve → Rank → Calibrate** paradigm:
1. **Retrieve** evidence from historical visits, raw text, and spatial anchors to form discrete hypotheses (Candidates C1-C6).
2. **Rank** these spatial candidates using a deterministic linear scorer based on semantic alignment and neighbor support.
3. **Calibrate** an empirical confidence radius and generate turn-by-turn directions.

---

## 1. System Master Flow Diagram

```mermaid
graph TD
    %% Data Sources
    A1[(addresses.csv)] -->|Raw Text| P2
    A2[(field_visits.csv)] -->|Visit Metadata| P1
    A3[(visit_gps_points.csv)] -->|GPS Trails| P1
    A4[(baseline_geocodes.csv)] -->|Commercial API Priors| P4
    A5[(landmarks_poi.csv)] -->|Known POIs| P2
    A6[(localities.csv)] -->|Verified Areas| P2

    %% Phases
    subgraph "Phase 1: Operational Sanitizer"
        P1["Trajectory Miner & Sanitizer<br/>(Photo Hash, Dwell Time)"]
    end
    
    subgraph "Phase 2: Entity Resolution"
        P2["MuRIL NER Address Parser<br/>& POI Resolver"]
    end
    
    subgraph "Phase 3: Spatial Graph"
        P3["Hierarchical Street Graph<br/>& Centroid Extractor"]
    end
    
    subgraph "Phase 4: Reranker & Calibrator"
        P4["Candidate Generator (C1-C6)<br/>Linear Utility Scorer"]
        P5["Quantile Calibrator &<br/>OSM Navigator"]
        P4 --> P5
    end

    %% Flow of internal Data
    P1 -->|visit_features.parquet| P3
    P1 -->|visit_features.parquet| P4
    
    P2 -->|parsed_addresses.parquet| P3
    P2 -->|parsed_addresses.parquet| P4
    
    P3 -->|street_centroids.parquet| P4
    
    %% Outputs
    P5 -->|predictions.json| OUT1["Final Coordinates (x,y)"]
    P5 -->|predictions.json| OUT2["Confidence Radius (R)"]
    P5 -->|predictions.json| OUT3["Navigation Directions"]
```

---

## 2. Deep Dive: Phase-by-Phase Breakdown with Examples

### Phase 1: Operational Sanitizer & Trajectory Miner
This block cleanses the operational field agent data to establish ground truth "Check-ins" (Direct Visit Doorstep Medoids) and assigns integrity weights.

- **Inputs**: 
  - `field_visits.csv`: Agent logs (e.g., visit_id, address_id, outcome, photo_hash).
  - `visit_gps_points.csv`: Raw GPS pings taken every ~30 seconds during a visit.
- **Process**:
  - **Collision Detection**: Discards visits if an agent reused the exact same photograph (`photo_hash`) across multiple visits (Spoofing).
  - **Dwell/Speed Weighting**: Calculates tortuosity and dwell medoids to assign an Integrity Score $I(v)$ and Spatial Evidence Strength $S(v)$.
- **Output**: `visit_features.parquet` (Cleaned medoid coordinates and reliability weights for each visit).

```mermaid
flowchart LR
    IN1["Raw GPS Points: <br> (x: 120, y: 150), (122, 149), ... "] --> PROC["Calculate Dwell Medoid &<br>Check Photo Hash"]
    IN2["Visit Log:<br> outcome='met_borrower', photo='ph0080'"] --> PROC
    PROC --> OUT["Clean Visit Feature:<br>medoid=(121.5, 149.5),<br> I(v)=0.95, S(v)=0.88"]
```

---

### Phase 2: Multilingual Address Parser & POI Resolver
This block uses Deep Learning (MuRIL Transformer NER) to understand messy, multi-lingual Indian addresses and map them to physical anchor databases.

- **Inputs**: 
  - `addresses.csv`: e.g., "near hanuman mandir, behind ration shop, gandhi nagar 560001"
  - `landmarks_poi.csv` & `localities.csv`: Known mapping datasets.
- **Process**: 
  - MuRIL extracts entities (Street, Landmark, Locality). 
  - The POI Resolver canonicalizes these entities (e.g., mapping "hanuman mandir" to a specific `poi_id` bounded within "gandhi nagar").
- **Output**: `parsed_addresses.parquet`

```mermaid
flowchart LR
    IN["Raw Address: <br>Bus Stop hattira, Vinayaka Nagar"] --> NLP["MuRIL NER"]
    NLP --> ENTS["Landmark: 'Bus Stop'<br>Locality: 'Vinayaka Nagar'"]
    ENTS --> RESOLV["POI Resolver + <br>localities.csv"]
    RESOLV --> OUT["Parsed Features:<br>poi_id: LMK_45 (x:310, y:410)<br>locality_id: LOC_12"]
```

---

### Phase 3: Spatial Graph & Cold-Start Propagation
Builds a hierarchical relationship graph (`Town -> Locality -> Street`) to allow unvisited "cold-start" addresses to borrow location knowledge from visited neighbors on the same street.

- **Inputs**: `parsed_addresses.parquet` and `visit_features.parquet`.
- **Process**: Groups all high-integrity visits (`I(v) > 0.85`) by their extracted Street and Locality tokens to compute robust `street_centroids`.
- **Output**: `street_centroids.parquet`

```mermaid
flowchart TD
    IN1["Visit 1 on 'MG Road' at (100, 200)"] --> AGG["Group by Street:<br>'MG Road' in 'Locality A'"]
    IN2["Visit 2 on 'MG Road' at (105, 195)"] --> AGG
    AGG --> OUT["street_centroids.parquet:<br>MG Road Centroid = (102.5, 197.5)"]
```

---

### Phase 4: Candidate Reranker & Quantile Calibrator
The core decision engine. It generates 6 potential coordinates (Candidates), ranks them, and assigns mathematical uncertainty.

#### The 6 Spatial Candidates:
1. **C1 (Commercial Baseline Prior)**: Sourced from `baseline_geocodes.csv` (e.g., Google Maps pin).
2. **C2 (Direct Visit Doorstep Medoid)**: Sourced from Phase 1 if the address was historically visited.
3. **C3 (POI Anchor Coordinate)**: Sourced from Phase 2 if a landmark was extracted.
4. **C4 (Neighbor Street Cluster)**: Sourced from Phase 3 street centroids.
5. **C5 (Locality Centroid)**: Safe fallback to the neighborhood center.
6. **C6 (Bayesian Grid Fallback)**: Machine Learning fallback for "Graph Deserts" where streets aren't matched.

- **Ranking Process**: A deterministic linear scorer evaluates semantic alignment and neighbor support to pick the best Candidate.
- **Calibration Process**: Based on the chosen Candidate tier, it assigns a dynamic Confidence Radius ($R$) using empirical quantiles (e.g., if C2 is chosen, R is small; if C5 is chosen, R is large).
- **Outputs**: Final $(x,y)$, Radius, and offline Directions (snapped to OSM).

```mermaid
flowchart TD
    subgraph "Candidate Generation"
        C1["C1: Baseline (300, 400)"]
        C2["C2: Null - Unvisited"]
        C3["C3: Temple POI (310, 410)"]
        C4["C4: Main St Centroid (312, 405)"]
    end
    
    C1 --> SCORER{"Linear Scorer"}
    C2 -.-> SCORER
    C3 --> SCORER
    C4 --> SCORER
    
    SCORER -->|Winner: C3| CALIB["Quantile Calibrator"]
    CALIB --> OUT1["Final Coord: (310, 410)"]
    CALIB --> OUT2["Radius: 45m (Evidence Tier 3)"]
    CALIB --> OUT3["Directions: '10m SW of Temple'"]
```

---

## 3. Data In/Out Example (End-to-End)

**Scenario**: A new loan applicant (Cold-Start / Unvisited).
**Input Record**:
- `address_id`: `ACC_999`
- `address_text`: *"Flat 2B, opposite Apollo Pharmacy, 2nd Cross, HSR Layout"*
- `baseline_geocode`: `(x: 1500, y: 2500)` (Google Maps guessed a generic HSR Layout center).

**Flow Execution**:
1. **Phase 1**: Skips (No historical visits).
2. **Phase 2 (Parser)**: 
   - *Leaves block as*: `landmark='Apollo Pharmacy'`, `street='2nd Cross'`, `locality='HSR Layout'`.
   - `poi_id` resolved to `(x: 1800, y: 2850)`.
3. **Phase 3 (Spatial Graph)**:
   - *Leaves block as*: Finds previous visits by other people on '2nd Cross' in 'HSR Layout' resulting in street centroid `(x: 1810, y: 2840)`.
4. **Phase 4 (Reranker)**:
   - Evaluates:
     - C1: `(1500, 2500)`
     - C3: `(1800, 2850)` [Apollo]
     - C4: `(1810, 2840)` [2nd Cross]
   - *Leaves block as*: Chooses **C4** as the winner due to strong neighbor support on the street network.
5. **Phase 4 (Calibrator/OSM)**:
   - *Leaves block as Final Output*: 
     - Coordinates: `(1810, 2840)`
     - Radius: `35m` (Tight confidence because street matches).
     - Directions: *"14m South of Apollo Pharmacy on 2nd Cross"*.

---

## 4. Complete System Component Graph

```mermaid
flowchart TD
    %% Global styling
    classDef inputFile fill:#2C3E50,stroke:#34495E,stroke-width:2px,color:#FFFFFF;
    classDef outputData fill:#8E44AD,stroke:#9B59B6,stroke-width:2px,color:#FFFFFF;
    classDef component fill:#27AE60,stroke:#2ECC71,stroke-width:2px,color:#FFFFFF;

    %% Data Inputs
    F_visits[("field_visits.csv<br/>Fields: visit_id, outcome, photo_hash<br/>Example: V123, 'met_borrower', 'ph0080'")]:::inputFile
    F_gps[("visit_gps_points.csv<br/>Fields: lat, lng, timestamp<br/>Example: 12.971, 77.594, '10:05'")]:::inputFile
    F_addr[("addresses.csv<br/>Fields: raw_text<br/>Example: 'near hanuman mandir, gandhi nagar'")]:::inputFile
    F_poi[("landmarks_poi.csv<br/>Fields: poi_id, name, x, y<br/>Example: LMK_45, 'hanuman mandir', 310, 410")]:::inputFile
    F_loc[("localities.csv<br/>Fields: loc_id, name<br/>Example: LOC_12, 'gandhi nagar'")]:::inputFile
    F_base[("baseline_geocodes.csv<br/>Fields: geocoder_x, geocoder_y<br/>Example: 300, 400")]:::inputFile

    %% Components
    subgraph Phase1["Phase 1: Operational Sanitizer"]
        C_Miner["Trajectory Miner & Sanitizer<br/>(Calculates Dwell Medoids,<br/>Filters GPS Spoofing & Photo Duplicates)"]:::component
    end

    subgraph Phase2["Phase 2: Entity Resolution"]
        C_Parser["MuRIL NER Address Parser<br/>& POI Resolver<br/>(Extracts Entities, Resolves to Database)"]:::component
    end

    subgraph Phase3["Phase 3: Spatial Graph"]
        C_Graph["Hierarchical Street Graph<br/>& Centroid Extractor<br/>(Groups visits by Street/Locality)"]:::component
    end

    subgraph Phase4["Phase 4: Reranker & Calibrator"]
        C_Ranker["Candidate Generator & Scorer<br/>(Generates C1-C6, Linear Utility Reranking)"]:::component
        C_Calib["Quantile Calibrator & OSM Navigator<br/>(Calculates Confidence Radius, Snaps to Roads)"]:::component
    end

    %% Internal Data Passes
    D_visitFeat[/"visit_features.parquet<br/>Outputs: Medoid (x,y), I(v), S(v)<br/>Example: (121.5, 149.5), 0.95, 0.88"/]:::outputData
    D_parsedAddr[/"parsed_addresses.parquet<br/>Outputs: Locality_ID, Street_ID, POI_ID<br/>Example: LOC_12, null, LMK_45"/]:::outputData
    D_streetCent[/"street_centroids.parquet<br/>Outputs: Street Centroid (x,y)<br/>Example: 'MG Road' -> (102.5, 197.5)"/]:::outputData

    %% Flow: Phase 1
    F_visits --> |Visit Metadata| C_Miner
    F_gps --> |GPS Pings| C_Miner
    C_Miner --> |Outputs| D_visitFeat

    %% Flow: Phase 2
    F_addr --> |Raw String| C_Parser
    F_poi --> |Anchor DB| C_Parser
    F_loc --> |Anchor DB| C_Parser
    C_Parser --> |Outputs| D_parsedAddr

    %% Flow: Phase 3
    D_visitFeat --> |Clean Check-ins| C_Graph
    D_parsedAddr --> |Extracted Street/Locality| C_Graph
    C_Graph --> |Outputs| D_streetCent

    %% Flow: Phase 4
    F_base --> |C1: Baseline Prior| C_Ranker
    D_visitFeat --> |C2: Direct Visit Prior| C_Ranker
    D_parsedAddr --> |C3: POI Anchor & C5: Locality| C_Ranker
    D_streetCent --> |C4: Neighbor Support| C_Ranker

    C_Ranker --> |Winning Spatial Candidate| C_Calib

    %% Final Outputs
    O_Coord(("[Final Output] Coordinates (x, y)<br/>Example: (310, 410)")):::outputData
    O_Radius(("[Final Output] Confidence Radius (m)<br/>Example: 45m (Evidence Tier 3)")):::outputData
    O_Dir(("[Final Output] Turn-by-turn Navigation<br/>Example: '10m SW of hanuman mandir'")):::outputData

    C_Calib --> O_Coord
    C_Calib --> O_Radius
    C_Calib --> O_Dir
```
