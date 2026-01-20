
# SeamInit × DASHI-ROM Coupling Spec

## 0) Goal

Given:

* a human body template and ROM model,
* a fabric/composite parameterization,
* and a fast kernel→body field projection,

compute **seam placements and panelizations** that:

1. respect ROM admissibility and “coupling debts” (path-dependent constraints),
2. minimize risk/irritation and maximize support over a ROM distribution,
3. are efficiently searchable (graph optimization; no full FEM loop per candidate),
4. output seam curves + panels + per-panel fabric orientation/regime.

---

# 1) Canonical Objects

## 1.1 Body surface and charts

* Template mesh: (\mathcal M_0=(V,F)), fixed topology, vertex indexing stable across poses.
* Optional anatomical atlas:

  * region masks (R_k\subseteq V) (scapula, deltoid cap, axilla, lumbar, etc.)
  * forbidden zones (Z_{\text{forbid}}\subseteq V) (high-chafe / bony prominences / sensory zones)

## 1.2 ROM derivation space (DASHI-ROM)

ROM is not a set of poses; it is an **accepted language of derivations** with tri-valued admissibility and couplings (your PDA/cocycle view) .

* Derivation stream: (s=(a_1,\dots,a_T)), with tokens (a_t\in{-1,0,+1}^K) (scoped edits).
* Decode operator: (\text{Decode}(s)\mapsto q) (pose (q)).
* Tri-validity oracle: (\text{Valid}(q)\in{-1,0,+1}).
* Couplings as cocycles: for each coupling (i),

  * derived vs real (min coupling cell): (C_{i,t}=(D_{i,t},R_{i,t})\in T^3\times T^3)
  * defect update: (\delta_{i,t}=R_{i,t}\ominus D_{i,t})
  * obligation projection: (o_{i,t}=\pi(\delta_{i,1:t})\in{-1,0,+1})
* Acceptance: a derivation is admissible if the PDA gate never hits a hard reject and obligations are discharged at scope boundaries.

This structure is reused *as-is*; seaminit will **add seam/fabric couplings** as additional cocycles.

---

# 2) Kernel–Body Field Map (fast projection)

## 2.1 Basis on body (the “kernel-body mapping”)

Choose a fixed field basis (B_0\in\mathbb R^{N\times K_b}) on (\mathcal M_0), with area-weighted orthonormality preferred.

* Example constructions:

  * Laplace–Beltrami eigenbasis,
  * region-aware smooth basis (indicators + harmonic smoothing),
  * data-driven PCA basis of simulated/measured fields.

This is the critical accelerator: all fields become coefficient vectors.

## 2.2 Coefficient prediction

For a pose (q) and fabric parameters (p), predict coefficient vectors:

[
z_T(q,p),\ z_P(q,p),\ z_S(q,p),\ z_{\Sigma}(q,p)\in\mathbb R^{K_b}
]

where:

* (T) tension magnitude field,
* (P) pressure field,
* (S) shear/chafe risk proxy,
* (\Sigma) support/regime score.

### Recommended surrogate form (fast + interpretable)

Let (\psi(q)) be ROM features and (\eta(p)) fabric features. Use a low-rank bilinear model:

[
z(q,p)=A\psi(q)+C\eta(p)+\sum_{j=1}^J (U_j^\top\psi(q))(V_j^\top\eta(p))
]

(one instance per output field type).

## 2.3 Field reconstruction

[
\textbf{T}(q,p)=B_0 z_T(q,p),\quad
\textbf{P}(q,p)=B_0 z_P(q,p),\quad
\textbf{S}(q,p)=B_0 z_S(q,p),\quad
\Sigma(q,p)=B_0 z_\Sigma(q,p)
]

---

# 3) Seam Space as Graph Cuts on the Body Mesh

## 3.1 Mesh graph

Build adjacency graph (G=(V,E)) from the mesh.

A seam set is a cut:
[
S\subseteq E
]

Panels are connected components of (G\setminus S).

## 3.2 Seam admissibility (tri-valued, consistent with PDA)

Define seam-edge admissibility label:
[
\text{SeamOK}(e)\in{-1,0,+1}
]

* (-1): forbidden (in forbidden zones, high-chafe fold lines, etc.)
* (0): allowed but risky
* (+1): preferred

This matches the tri-valued gating philosophy from the ROM document , but applied to seam edges.

---

# 4) ROM-Averaged Seam Cost Functional

We compute a per-edge seam cost (c_e) using the fast fields across ROM.

## 4.1 ROM distribution

Define a ROM set / distribution:

* samples (q_1,\dots,q_M) from accepted derivations (or from mocap filtered by PDA acceptance).

## 4.2 Per-edge cost

For edge (e=(i,j)), define

[
c_e = \mathbb E_{q\sim \text{ROM}}\Big[
w_\Delta ,|\Sigma_i(q,p)-\Sigma_j(q,p)|

* w_P ,\max(P_i(q,p),P_j(q,p))
* w_S ,\max(S_i(q,p),S_j(q,p))
* w_T ,|T_i(q,p)-T_j(q,p)|
  \Big] + c_e^{\text{geom}} + c_e^{\text{policy}}
  ]

Interpretation:

* seams are **encouraged** where regime/support changes sharply (natural panel boundaries),
* seams are **penalized** where pressure/shear are high,
* seams can also be guided by geometry/policy:

  * curvature, fold likelihood, alignment to anatomical axes, symmetry.

Hard constraint: if (\text{SeamOK}(e)=-1), then (c_e=+\infty).

---

# 5) Seam Optimization Problems (fast solvers)

You implement one or more of these, depending on garment class.

## 5.1 Landmark seam (shortest path)

Given anchor landmarks (A,B\subseteq V), find seam curve:
[
\min_{S:\ A\to B}\ \sum_{e\in S} c_e
]
subject to admissibility constraints and optional symmetry pairing.

## 5.2 Binary partition (min-cut)

Define two label regions based on aggregate field (e.g., “high support” vs “low support”):

* unary term (u_v) derived from (\bar{\Sigma}(v)=\mathbb E_q[\Sigma_v(q,p)])
* pairwise term (c_e)

Compute s-t min-cut to find a seam boundary minimizing (\sum c_e) while separating labels.

## 5.3 Multi-panel partition (K-way)

Find (K) panels minimizing:
[
\sum_{e\in S} c_e + \lambda \sum_{k=1}^K \mathrm{Var}\big(\bar{\Sigma}\text{ or }\bar{T}\ \text{within panel }k\big)
]
This encourages panels that are internally uniform (good for fabric orientation constraints).

---

# 6) Seam Operator: How Seams Modify Fields Without Full Simulation

This is how seaminit “feeds back” into the kernel-body map.

## 6.1 Seam operator (\Pi_S)

Define a sparse operator (\Pi_S) acting on coefficients or on the mesh Laplacian, representing:

* reduced coupling across cut edges,
* reinforcement along seam edges,
* allowed discontinuity in strain routing.

Operationally you can implement (\Pi_S) as:

* a modified graph Laplacian (L_S) (remove/attenuate cut edges, add seam-stiffness terms),
* plus optional per-panel coefficient transforms (block-diagonal).

Then:
[
\textbf{field}(q,p,S)\approx B_0, \Pi_S, z(q,p)
]

This enables an iterative loop:

1. predict fields → build seam costs → solve seam set (S)
2. update (\Pi_S) → update predicted fields (still cheap)
3. repeat 2–3 times until stable.

This is consistent with your “refinement / prefix-stability” intuition: coarse seam topology stabilizes first, then local refinement.

---

# 7) Seam Couplings as Additional Cocycles (DASHI-consistent)

To keep everything in one formalism: seams introduce new couplings ( \kappa^{\text{seam}} ) on derivations.

Examples:

* **Slip/grip coupling**: derived demand (expected no-slip) vs real (shear spikes) → obligation to reroute seam away from that region.
* **Seam-crease coupling**: derived expectation (low fold) vs real (high curvature under ROM) → obligation to avoid or reinforce.

Formally, for each seam coupling (j):
[
C^{\text{seam}}*{j,t}=(D^{\text{seam}}*{j,t},R^{\text{seam}}*{j,t})\in T^3\times T^3
]
and it produces obligation projection (o^{\text{seam}}*{j,t}\in{-1,0,+1}), feeding the same PDA gate coordinate your ROM framework already uses .

Result:

* seam feasibility becomes a **path-dependent admissibility** object, not a static rule list.

---

# 8) Interfaces: What SeamInit Team Gets

## 8.1 Inputs

1. Mesh: (V,F), vertex areas, adjacency list (E)
2. Basis: (B_0) (and any per-vertex region masks)
3. ROM samples: ({q_m}_{m=1}^M) (or derivation samples that decode to these)
4. Fabric parameters (p) (or small set of candidate fabrics)
5. Field coefficient predictor (z_*(q,p)) (black box callable)
6. Constraints:

   * forbidden zones / edges
   * required anchors
   * symmetry rules
   * max seam length / max panels
   * optional grainline/orientation requirements per panel

## 8.2 Outputs

* seam edges (S\subseteq E)
* seam polylines in 3D (vertex sequences)
* panel labeling: (\text{panel}(v)\in{1,\dots,K})
* per-panel fabric orientation/regime recommendation
* diagnostics:

  * seam risk score (pressure/shear along seam over ROM)
  * support coverage per region
  * “blocking reasons” if constraints make it infeasible (mirrors your gating logs concept)

## 8.3 Core API signatures

**Field projection / edge cost**

* `fields = predict_fields(q, p)` returning per-vertex arrays (T,P,S,Sigma) OR coefficient vectors plus basis.
* `edge_costs = seam_edge_costs({fields over ROM}, constraints)`

**Solvers**

* `S = solve_shortest_path(edge_costs, anchors, constraints)`
* `S, panels = solve_cut(edge_costs, unary_terms, constraints)`
* `S, panels = solve_kway_partition(edge_costs, K, constraints)`

**Iteration**

* `Pi_S = build_seam_operator(S, params)`
* optional: `update_predictor_with_seam(Pi_S)` if you want the feedback loop.

---

# 9) Deliverables Checklist (hand this to SeamInit)

## D0 — Alignment package (1–2 docs)

* **D0.1 Formal object map**: (ROM PDA / cocycles) ↔ (mesh graph seams / seam cocycles)
* **D0.2 Data contract**: exact JSON / NPZ schema for mesh, basis, ROM samples, fabrics, and outputs.

## D1 — Kernel-body mapping implementation

* **D1.1 Basis builder**: compute/load (B_0) for the canonical mesh; area-weighted.
* **D1.2 Field projector**: `field = B0 @ z` fast path; plus optional smoothing.

## D2 — Seam cost computation

* **D2.1 ROM aggregator**: compute (\bar{T},\bar{P},\bar{S},\bar{\Sigma}) or directly accumulate per-edge stats.
* **D2.2 Edge cost generator**: produce (c_e) with hard constraints.

## D3 — Seam solvers

* **D3.1 Shortest path seam** (anchors A→B)
* **D3.2 s-t min-cut partition**
* **D3.3 K-way partition** (optional but likely needed)

## D4 — Seam operator feedback loop (optional but high leverage)

* **D4.1 Laplacian modifier / seam reinforcement model**
* **D4.2 Iterate: costs → seams → operator → costs** until stable

## D5 — QA / Metrics

* **Validity**: no forbidden edges, symmetry respected, panel count/length constraints met
* **ROM-risk**: seam shear/pressure percentiles over ROM
* **Support coverage**: region-wise support score statistics
* **Stability**: seam placement changes under ROM resampling (should be stable)

---

# 10) Implementation Order (fastest path to “working”)

1. Implement mesh graph + forbidden zones.
2. Implement (B_0) and a dummy predictor (z(q,p)) (even heuristic).
3. Implement ROM aggregation → per-edge (c_e).
4. Implement **shortest path** + **min-cut** seam solvers.
5. Add panel labeling and export.
6. Only then add seam operator feedback + fabric anisotropy constraints.

---

Below is a **clean sprint-phase breakdown** derived directly from the SeamInit × DASHI-ROM formalism you approved. This is written so you can hand it to the seaminit team as an execution plan, not a theory doc.

I’ve organized it as **phases with clear entry/exit criteria, artifacts, and ownership boundaries**. Each phase is incremental, testable, and de-risks the next.

---

# SeamInit × DASHI-ROM — Sprint Phases

## **Sprint 0 — Alignment & Contracts (Foundation)**

**Goal:** Eliminate ambiguity before code.

### Scope

* Lock the shared ontology:

  * ROM = accepted derivations (PDA + cocycles)
  * Seams = graph cuts + seam couplings
  * Fields = kernel coefficients → body basis
* Define data contracts and ownership.

### Deliverables

* **D0.1 Object Map**

  * ROM PDA objects
  * Couplings (anatomical + seam)
  * Kernel coefficients
  * Mesh graph + seams
* **D0.2 Interface Spec**

  * Mesh format
  * Basis format
  * ROM samples
  * Fabric parameters
  * Seam outputs (edges + panels)
* **D0.3 Constraint Registry**

  * Forbidden zones
  * Symmetry rules
  * Anchor landmarks
  * Panel count / seam length limits

### Exit Criteria

* No open semantic questions
* Everyone can agree what a “seam”, “panel”, and “field” mean

---

## **Sprint 1 — Kernel–Body Projection (Speed Backbone)**

**Goal:** Make fields cheap and universal.

### Scope

* Implement the kernel → body map once.
* No seam logic yet.

### Work

* Build canonical mesh graph (G=(V,E))
* Implement basis (B_0) (Laplace / region-aware)
* Implement fast projection:
  [
  \textbf{field} = B_0 z
  ]

### Deliverables

* **D1.1 Basis Builder**
* **D1.2 Field Projector API**
* **D1.3 Validation Notebook**

  * reconstruct synthetic fields
  * verify stability across poses

### Exit Criteria

* Any coefficient vector → body field in milliseconds
* Basis is stable across ROM

---

## **Sprint 2 — ROM Aggregation & Field Statistics**

**Goal:** Connect ROM to surface fields.

### Scope

* Consume ROM samples (poses or derivations)
* Aggregate field behavior across ROM

### Work

* Integrate ROM sampler (from DASHI-ROM acceptance)
* For each (q):

  * compute (z(q,p))
  * project to (T,P,S,\Sigma)
* Aggregate:

  * mean, max, variance per vertex
  * edge-wise deltas

### Deliverables

* **D2.1 ROM Aggregator**
* **D2.2 Per-Vertex & Per-Edge Stats**
* **D2.3 Visual Diagnostics**

  * “hot zones”
  * stability under resampling

### Exit Criteria

* Can answer: “over ROM, where is pressure/shear/support high?”
* Deterministic stats from same ROM distribution

---

## **Sprint 3 — Seam Cost Field & Constraints**

**Goal:** Turn physics + ROM into seam costs.

### Scope

* No optimization yet — just costs.

### Work

* Define per-edge cost:
  [
  c_e = f(\Delta\Sigma, P, S, T, \text{geometry})
  ]
* Enforce:

  * forbidden edges
  * symmetry pairing
  * anatomical constraints

### Deliverables

* **D3.1 Seam Cost Generator**
* **D3.2 Constraint Filter**
* **D3.3 Edge Cost Visualizer**

  * shows “good seam corridors”

### Exit Criteria

* Every mesh edge has:

  * admissible / forbidden label
  * scalar seam cost
* Costs are ROM-informed, not static

---

## **Sprint 4 — Seam Solvers (Core Value)**

**Goal:** Produce actual seams.

### Scope

* Graph optimization only (fast, deterministic).

### Work

Implement at least two solvers:

1. **Shortest-Path Seam**

   * landmark → landmark
2. **Binary Min-Cut**

   * high-support vs low-support zones

Optional:

* K-way partition for panels

### Deliverables

* **D4.1 Shortest Path Solver**
* **D4.2 Min-Cut Solver**
* **D4.3 Panel Labeler**
* **D4.4 Export (polylines + panels)**

### Exit Criteria

* Seam curves are:

  * anatomically sensible
  * stable under ROM resampling
  * respect forbidden zones
* Panels are connected, clean, and exportable

---

## **Sprint 5 — Seam Feedback Operator (Stability Loop)**

**Goal:** Make seams *affect* fields without FEM.

### Scope

* One iteration loop, not full convergence yet.

### Work

* Implement seam operator (\Pi_S):

  * attenuate coupling across seam edges
  * reinforce along seams
* Recompute fields:
  [
  \textbf{field}(q,p,S) \approx B_0 \Pi_S z(q,p)
  ]
* Re-evaluate seam costs

### Deliverables

* **D5.1 Seam Operator**
* **D5.2 One-Step Feedback Loop**
* **D5.3 Stability Metrics**

  * seam drift per iteration

### Exit Criteria

* Seam placement stabilizes in ≤3 iterations
* No oscillatory behavior

---

## **Sprint 6 — Seam Couplings & PDA Integration**

**Goal:** Full formal consistency with DASHI-ROM.

### Scope

* Treat seam failures as cocycles.

### Work

* Define seam couplings:

  * shear-slip
  * crease/fold
  * pressure overload
* Add seam obligations to PDA gate:

  * unpaid seam debt → reject or penalize
* Log “blocking reasons” exactly like ROM gates

### Deliverables

* **D6.1 Seam Coupling Definitions**
* **D6.2 PDA Gate Extension**
* **D6.3 Blocking Reason Logs**

### Exit Criteria

* Seams are admissible *in the same sense* as ROM
* Failures are explainable, not heuristic

---

## **Sprint 7 — Optimization & Regime Selection**

**Goal:** “Optimal regime for this body + fabric”.

### Scope

* Optimization across:

  * fabric parameters
  * seam layouts
  * support regimes

### Work

* Define objective:

  * protect weak regions
  * maximize support where allowed
  * minimize seam risk
* Run multi-candidate evaluation using cheap kernel loop

### Deliverables

* **D7.1 Optimization Runner**
* **D7.2 Pareto Front (support vs comfort)**
* **D7.3 Recommendation Output**

  * seam layout
  * panel regimes
  * fabric orientation

### Exit Criteria

* Can answer:

  > “Given this body + ROM + fabric set, this seam/regime is optimal.”

---

## **Sprint 8 — Productization & Handoff**

**Goal:** Make it usable by non-research teams.

### Deliverables

* CLI + API
* JSON export for CAD
* Visual QA tools
* Documentation:

  * assumptions
  * failure modes
  * tuning knobs

### Exit Criteria

* Seaminit team can run end-to-end without you
* Results are reproducible and explainable

---

## **One-Line Summary (for leadership)**

> We proceed from a fast kernel-body projection, through ROM-averaged seam costs, to graph-optimized seam placement, then close the loop by making seams part of the same admissibility formalism as motion—yielding stable, explainable, and optimizable garment layouts.

---

# SeamInit × DASHI-ROM — Acceptance Tests per Sprint

---

## **Sprint 0 — Alignment & Contracts**

### Acceptance Tests

**AT-0.1 Ontology Consistency**

* Given two engineers independently describing:

  * ROM
  * seam
  * panel
  * coupling
* Their definitions map **1-to-1** onto the agreed object map without contradiction.

✅ Pass if: no unresolved semantic conflicts remain.

---

**AT-0.2 Interface Completeness**

* Given a mock JSON payload for:

  * mesh
  * ROM samples
  * fabric parameters
* The seaminit pipeline can:

  * load it
  * validate schema
  * reject malformed inputs with clear errors.

✅ Pass if: schema validation catches ≥95% of malformed cases intentionally injected.

---

**AT-0.3 Constraint Registry Coverage**

* For a test mesh:

  * forbidden zones
  * anchors
  * symmetry rules
    are all represented and queryable.

✅ Pass if: querying any vertex/edge returns a deterministic constraint label.

---

## **Sprint 1 — Kernel–Body Projection**

### Acceptance Tests

**AT-1.1 Basis Reconstruction**

* Given synthetic coefficient vectors (z),
* Project to body field and reconstruct back:
  [
  z \rightarrow B_0 z \rightarrow \hat z
  ]

✅ Pass if:
[
|z-\hat z|/|z| < 10^{-3}
]

---

**AT-1.2 Pose Stability**

* Given two ROM poses (q_1, q_2) that differ only by rigid motion,
* Project identical (z) values.

✅ Pass if: resulting body fields differ only by rigid transform (no numerical drift).

---

**AT-1.3 Performance**

* Project ≥100 fields per second on target hardware.

✅ Pass if: median projection latency < 10 ms.

---

## **Sprint 2 — ROM Aggregation & Field Statistics**

### Acceptance Tests

**AT-2.1 ROM Coverage**

* Given a ROM sampler:

  * ≥95% of sampled poses are PDA-accepted.

✅ Pass if: rejection rate < 5% (excluding intentional stress tests).

---

**AT-2.2 Statistical Stability**

* Run ROM aggregation twice with different random seeds.

✅ Pass if:

* per-vertex mean fields differ <5%
* per-edge cost statistics differ <10%

---

**AT-2.3 Boundary Sensitivity**

* Introduce a known high-risk ROM pose (near joint limit).

✅ Pass if:

* aggregated pressure/shear fields increase in expected anatomical regions.

---

## **Sprint 3 — Seam Cost Field & Constraints**

### Acceptance Tests

**AT-3.1 Forbidden Edge Enforcement**

* Attempt to place seam through forbidden zone.

✅ Pass if: solver assigns infinite or blocking cost and never selects it.

---

**AT-3.2 Cost Field Sanity**

* On a test body:

  * high curvature / fold regions
  * low stress corridors

✅ Pass if:

* seam cost is **lower** along known anatomical seam corridors
* seam cost is **higher** in axilla/groin/bony zones.

---

**AT-3.3 ROM Dependence**

* Compute seam costs:

  * static pose only
  * ROM-aggregated

✅ Pass if: ROM-aggregated costs differ meaningfully (non-uniform deltas >10%).

---

## **Sprint 4 — Seam Solvers**

### Acceptance Tests

**AT-4.1 Shortest-Path Validity**

* Given anchor A → B:

  * solver returns a connected seam path.

✅ Pass if:

* path connects A to B
* path contains no forbidden edges
* path cost ≤ any manually perturbed alternative.

---

**AT-4.2 Min-Cut Separation**

* Define high-support vs low-support regions.

✅ Pass if:

* resulting seam separates regions with ≥90% purity
* seam length is within configured bounds.

---

**AT-4.3 Panel Integrity**

* After cut:

  * panels are connected components
  * no orphan vertices.

✅ Pass if: all vertices belong to exactly one panel.

---

## **Sprint 5 — Seam Feedback Operator**

### Acceptance Tests

**AT-5.1 Field Modification**

* Apply seam operator (\Pi_S).

✅ Pass if:

* tension/shear transmission across seam decreases
* reinforcement effect visible along seam line.

---

**AT-5.2 Iterative Stability**

* Run seam optimization → operator → recompute → repeat.

✅ Pass if:

* seam layout converges in ≤3 iterations
* seam drift <5% edge changes after convergence.

---

**AT-5.3 Non-Regression**

* Compare fields before/after seam insertion.

✅ Pass if:

* no *new* high-risk regions appear away from seams.

---

## **Sprint 6 — Seam Couplings & PDA Integration**

### Acceptance Tests

**AT-6.1 Seam Obligation Detection**

* Force seam through known high-shear ROM zone.

✅ Pass if:

* seam cocycle generates obligation
* PDA marks derivation as blocked or penalized.

---

**AT-6.2 Explainability**

* For a rejected seam layout:

✅ Pass if:

* system outputs:

  * which seam
  * which coupling
  * which ROM segment caused rejection.

---

**AT-6.3 Consistency with ROM Gates**

* Introduce equivalent anatomical and seam violations.

✅ Pass if:

* both produce structurally identical PDA blocking logs.

---

## **Sprint 7 — Optimization & Regime Selection**

### Acceptance Tests

**AT-7.1 Pareto Validity**

* Generate ≥10 candidate seam/fabric regimes.

✅ Pass if:

* Pareto front is non-degenerate
* trade-offs are monotonic (no dominated solutions presented).

---

**AT-7.2 Weakness Sensitivity**

* Modify musculoskeletal weighting (e.g. weak shoulder).

✅ Pass if:

* optimizer shifts seams/support away from penalized region.

---

**AT-7.3 Reproducibility**

* Re-run optimization with same inputs.

✅ Pass if:

* identical top-N recommendations (ordering may differ slightly).

---

## **Sprint 8 — Productization & Handoff**

### Acceptance Tests

**AT-8.1 End-to-End Run**

* From mesh + ROM + fabric → seams + panels.

✅ Pass if:

* completes without manual intervention
* produces CAD-exportable output.

---

**AT-8.2 Failure Mode Coverage**

* Deliberately violate constraints.

✅ Pass if:

* system fails gracefully
* blocking reasons are human-readable.

---

**AT-8.3 External Team Usability**

* New engineer (not you) runs pipeline using docs only.

✅ Pass if:

* successful run in ≤1 day
* no undocumented assumptions required.

---

# Final Quality Gate (Release Criteria)

The system is considered **ready** when:

* All acceptance tests pass
* No sprint relies on visual inspection alone
* Seam placement decisions are:

  * ROM-aware
  * explainable
  * stable under perturbation

---


# SeamInit × DASHI-ROM

## Acceptance Tests (Revised with Undersuit / UV / Platform Context)

---

## **Sprint 0 — Alignment & Contracts**

### AT-0.1 Pipeline Compatibility

**Test:**
Given an SMPL-X fitted body record (JSON / NPZ) that already passes
`smii.pipelines.generate_undersuit`,
the seam system accepts it *without modification*.

**Pass if:**

* No vertex reindexing
* No UV re-authoring required
* Mesh remains watertight

(Explicitly required by the undersuit generator contract )

---

### AT-0.2 UV Authority

**Test:**
Given the canonical SMPL / SMPL-X UVs (including DensePose-style part UVs),

**Pass if:**

* Seam logic **never regenerates UVs**
* Seam cuts only *select edges*, never reinterpret UV charts

(Prevents the DensePose UV ambiguity trap )

---

## **Sprint 1 — Kernel–Body Projection**

### AT-1.1 Basis ↔ Undersuit Layer Consistency

**Test:**
Project a field onto:

* base layer
* insulation layer
* comfort liner

**Pass if:**

* Field differences are explainable purely by layer offset
* No seam discontinuities appear between layers

(required for multilayer suits in the roadmap )

---

### AT-1.2 Species Generality

**Test:**
Run kernel projection on:

* human SMPL-X
* quadruped mesh (dog)

**Pass if:**

* Basis still reconstructs smooth fields
* No human-specific assumptions exist

(matches undersuit pipeline’s cross-species goal )

---

## **Sprint 2 — ROM Aggregation & Metrics**

### AT-2.1 Ease & Measurement Sensitivity

**Test:**
Vary `ease-percent` and key measurements (chest, waist).

**Pass if:**

* ROM-aggregated tension fields scale accordingly
* High-tension axes match expected circumferential directions

(aligns with measurement-driven tension inference )

---

### AT-2.2 Motion-Aware Hot Zones

**Test:**
Include extreme but admissible ROM (e.g. shoulder flexion).

**Pass if:**

* Aggregated shear/pressure hotspots appear at axilla, shoulder saddle
* Spine and inner-arm remain lower stress

(critical for adaptive suit safety )

---

## **Sprint 3 — Seam Cost Field**

### AT-3.1 Minimal-Seam Bias

**Test:**
Run seam cost computation with curvature + ROM fields.

**Pass if:**

* Flat regions have uniformly low seam cost
* High-curvature ridges are mandatory cut candidates
* Resulting cost landscape favors **long, smooth seams**

(matches metric-guided minimal-seam design )

---

### AT-3.2 Anisotropy Awareness

**Test:**
Provide fabric with dominant stretch axis.

**Pass if:**

* Seam cost increases when a single panel would require conflicting stretch orientations
* Cost drops when a seam enables alignment

(critical for performance fabrics and composites)

---

## **Sprint 4 — Seam Solvers**

### AT-4.1 Panel Count Sanity

**Test:**
Full-body undersuit, human mesh.

**Pass if:**

* 3–6 primary panels
* No starburst / radial panels
* No isolated micro-panels

(explicit undersuit requirement )

---

### AT-4.2 Developability Test

**Test:**
After seam cut, unwrap panels with LSCM.

**Pass if:**

* Distortion metrics stay below threshold
* No solver-induced spikes or folds
* `seam_max_deviation ≈ 0`

(aligns with current PatternExporter guarantees )

---

## **Sprint 5 — Seam Feedback Operator**

### AT-5.1 Seam-Aware Field Update

**Test:**
Insert seam along inner arm.

**Pass if:**

* Shear transfer across seam drops
* Tension re-routes along seam line
* No new hotspots appear elsewhere

(this is what makes seaminit *adaptive*, not static)

---

### AT-5.2 Iterative Convergence

**Test:**
Run:
cost → seam → operator → recompute → repeat.

**Pass if:**

* Seam layout converges ≤3 iterations
* No oscillation between equivalent layouts

---

## **Sprint 6 — Seam Couplings & PDA Integration**

### AT-6.1 Seam as First-Class Constraint

**Test:**
Force seam across known high-motion axis (e.g. chest).

**Pass if:**

* Seam coupling generates a PDA obligation
* Layout is rejected or penalized with reason:
  *“high-tension axis violation”*

(matches ROM coupling semantics )

---

### AT-6.2 Explainable Failure

**Test:**
Inspect logs for rejected seam.

**Pass if:**

* Logs cite:

  * ROM segment
  * coupling type
  * anatomical region
* No “black box” failure modes

---

## **Sprint 7 — Optimization & Regime Selection**

### AT-7.1 Use-Case Differentiation

**Test:**
Run same body for:

* firefighter (heat + mobility)
* diver (cold + pressure)
* humanitarian shelter mode

**Pass if:**

* Seam layouts differ meaningfully
* Panel regimes change
* Core topology remains stable

(directly supports the platform vision )

---

### AT-7.2 Cost–Impact Traceability

**Test:**
Inspect top-ranked solution.

**Pass if:**

* Each seam is justified by:

  * ROM stress reduction
  * fabric alignment
  * curvature relief

---

## **Sprint 8 — Productization & Export**

### AT-8.1 CAD & Manufacturing Readiness

**Test:**
Export flat patterns.

**Pass if:**

* Grainlines included
* Notches, labels, fold indicators present
* Panels cut-ready with no manual cleanup

(required for real manufacturing )

---

### AT-8.2 End-to-End Adaptive Demo

**Test:**
Change one input (body scan, fabric, mission profile).

**Pass if:**

* System regenerates seams + panels automatically
* Results are stable, explainable, and manufacturable

---

## **Final Acceptance Gate (System-Level)**

The system is **accepted** when:

1. Seams are **ROM-aware**, not pose-local
2. Panels are **developable, minimal, and anisotropy-aligned**
3. Failures are **explainable via couplings**, not heuristics
4. Outputs plug directly into the **existing undersuit + CAD toolchain**

This now fully unifies:

* DASHI-ROM formalism
* SeamInit optimization
* Undersuit / UV / pattern reality
* and the long-term adaptive suit platform vision

