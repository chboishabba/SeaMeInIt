
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

Below is a **formal, explicit** treatment of what you asked for:

1. ROM as an operator ( \mathcal R : \Theta \to \mathbb R^K )
2. A clean mapping to **SMPL-X Jacobians**
3. Seam cost as a **contraction of a ROM supertensor**

I’ll keep it tight but fully specified.

---

## 1) ROM as an explicit operator ( \mathcal R : \Theta \to \mathbb R^K )

### Pose space and body map (SMPL-X chart)

Let

* ( \Theta \subset \mathbb R^{J} ) be the pose parameter space (joint angles etc.),
* ( \beta ) the (fixed) shape parameters for a given body,
* ( \psi ) other nuisance params (expression etc., fixed/ignored).

SMPL-X provides a differentiable map
[
\Phi_\beta : \Theta \to \mathbb R^{3V},\qquad
\theta \mapsto x(\theta)=\mathrm{vec}(V(\theta)) .
]

(Here (x(\theta)) is the stacked vertex vector of length (3V).)

### Canonical field space

Pick a per-vertex *field* of interest (shear proxy, tension proxy, pressure proxy):
[
f:\mathbb R^{3V}\to \mathbb R^{V},\qquad x \mapsto u
]
where (u\in\mathbb R^V) is a scalar field on vertices.

Examples of (f) (you can swap later):

* displacement magnitude relative to neutral,
* local stretch proxy from skinning Jacobians,
* curvature-change proxy, etc.

### Basis compression

Let (B\in\mathbb R^{V\times K}) be your orthonormal canonical basis on the mesh (as in your repo).
Define the coefficient projection operator
[
\Pi_B(u)=B^\top u\in\mathbb R^K.
]

### ROM operator definition

Define the ROM operator:
[
\boxed{
\mathcal R(\theta)
;:=;
B^\top, f!\big(\Phi_\beta(\theta)\big)
;\in;
\mathbb R^{K}
}
]
So the sampler is literally computing ( \theta \mapsto \mathcal R(\theta)).

### ROM measure and “compressed ROM object”

If you want ROM as “the total permissible motion” rather than a pointwise map, endow (\Theta_{\rm adm}\subseteq\Theta) with a measure (\mu) (task-weighted, uniform, etc.):

* Mean ROM coefficient:
  [
  \bar c
  =
  \int_{\Theta_{\rm adm}}\mathcal R(\theta),d\mu(\theta)
  \in\mathbb R^K
  ]
* Second moment / covariance:
  [
  \Sigma_c
  =
  \int (\mathcal R(\theta)-\bar c)(\mathcal R(\theta)-\bar c)^\top,d\mu(\theta)
  \in\mathbb R^{K\times K}
  ]

That covariance is already a compressed “codependency” object in coefficient space.

---

## 2) Mapping (\mathcal R) to SMPL-X Jacobians cleanly

Differentiate (\mathcal R(\theta)=B^\top f(x(\theta))) where (x(\theta)=\Phi_\beta(\theta)).

By chain rule:
[
D_\theta \mathcal R(\theta)
===========================

B^\top , D_x f(x(\theta)) , D_\theta x(\theta).
]

SMPL-X gives (or you can compute) the Jacobian:
[
J_{\rm SMPLX}(\theta)
:=
D_\theta x(\theta)
\in\mathbb R^{3V\times J}.
]

So:
[
\boxed{
D_\theta \mathcal R(\theta)
===========================

B^\top , D_x f(x(\theta)) , J_{\rm SMPLX}(\theta)
\in \mathbb R^{K\times J}
}
]

### Choosing a field (f) that makes (D_x f) explicit

To make this *operational*, pick (f) with a tractable derivative.

A simple but real choice (used often as a first proxy):

* Define neutral vertices (x_0=\Phi_\beta(\theta_0)) at neutral pose.
* Let (d(\theta)=x(\theta)-x_0\in\mathbb R^{3V}).
* Define per-vertex displacement magnitude:
  [
  u_i(\theta)=|d_i(\theta)|_2
  \quad\text{where }d_i\in\mathbb R^3 \text{ is vertex }i\text{’s displacement}.
  ]

Then (f(x)=u) with
[
\frac{\partial u_i}{\partial d_i}
=================================

\frac{d_i^\top}{|d_i|_2+\varepsilon}
\in\mathbb R^{1\times 3}.
]

Stacked:
[
D_x f(x(\theta))
;\approx;
\mathrm{blkdiag}!\left(
\frac{d_1^\top}{|d_1|+\varepsilon},
\ldots,
\frac{d_V^\top}{|d_V|+\varepsilon}
\right)
\in\mathbb R^{V\times 3V}.
]

Plugging in:
[
D_\theta \mathcal R(\theta)
===========================

B^\top \Big[\mathrm{blkdiag}\big(\frac{d_i^\top}{|d_i|+\varepsilon}\big)\Big],J_{\rm SMPLX}(\theta).
]

That is a *clean* Jacobian-level ROM sensitivity map:

* columns: which joints drive ROM coefficients,
* rows: which compressed modes respond.

### If you want “strain-like” rather than displacement-like

Let (G(x)) produce per-vertex deformation gradient proxies (from skinning weights or local edge Jacobians). Then (f(x)=g(G(x))) and the same chain rule applies:
[
D_\theta \mathcal R = B^\top D_x f , J_{\rm SMPLX}.
]
The only change is the explicit form of (D_x f).

---

## 3) Seam cost as a contraction of the ROM supertensor

You want the statement: seam cost is literally a contraction of the ROM “total codependency tree” supertensor.

Here’s the precise way to say it.

### ROM supertensor (joint-coupling form)

Define a per-vertex, per-joint **local sensitivity tensor**:
[
S_{v j}(\theta)
:=
\left|\frac{\partial u_v(\theta)}{\partial \theta_j}\right|
]
where (u(\theta)=f(x(\theta))\in\mathbb R^V).

This is a vertex×joint object at pose (\theta).

Now define the **ROM supertensor** as the integrated joint-coupling (second-order) object:
[
\boxed{
\mathcal T_{v j k}
:=
\int_{\Theta_{\rm adm}}
S_{v j}(\theta),S_{v k}(\theta),d\mu(\theta)
}
]
This is:

* vertex (v),
* joint (j),
* joint (k).

Interpretation:

* diagonal (j=k): how much joint (j) alone drives vertex (v) across ROM,
* off-diagonal (j\neq k): codependency / co-activation at (v).

That matches your “codependency tree” notion in a mathematically concrete form.

### Seam cost is a contraction

A seam cost field needs a scalar per vertex (or per edge). We get it by contracting (\mathcal T) with a joint-weight matrix (W\in\mathbb R^{J\times J}) (task weighting, anatomical importance, discomfort weighting, etc.):

[
\boxed{
c_v
===

\sum_{j,k} W_{jk},\mathcal T_{v j k}
}
]

That is a contraction over the joint indices.

In tensor notation:
[
c = \mathcal T : W
\quad\text{(double contraction)}
]

If you want edge costs:
[
c_{(a,b)} = |c_a - c_b| + \lambda \max(c_a,c_b)
]
(or any of your earlier cost combinations).

### How this ties to your basis compression

You can also form the coefficient-space analogue:

Define coefficient sensitivities:
[
\frac{\partial \mathcal R_k}{\partial \theta_j}
\in\mathbb R^{K\times J}.
]

Then a coefficient-space ROM supertensor is:
[
\tilde{\mathcal T}_{k j \ell}
=============================

\int
\left|\frac{\partial \mathcal R_k}{\partial \theta_j}\right|
\left|\frac{\partial \mathcal R_\ell}{\partial \theta_j}\right|
d\mu(\theta)
]
and vertex-space (\mathcal T) is recovered (approximately) by lifting with (B):
[
\mathcal T_{vjk}
\approx
\sum_{a,b} B_{va}B_{vb},\tilde{\mathcal T}_{a j b}
]
(you’re “uncompressing” the contracted object back to vertices.)

This is the precise mathematical meaning of “supertensor compression.”

---

## How this matches your current pipeline (and why it’s consistent)

* Your repo’s **sampler JSON** is sampling (\mathcal R(\theta)) (or proxies of it).
* Your aggregation step is approximating integrals over (\Theta_{\rm adm}) (Monte Carlo / quadrature).
* Your seam cost NPZ is a scalar field (c_v), which in the “full formalism” is a contraction (\mathcal T:W), but in the current scaffolding is a simpler functional of aggregated vertex fields. That’s fine: the contraction view tells you what to evolve toward.

---

## Minimal upgrade path (no rewrites)

If you want the sampler_real to be formally faithful without building a full simulator:

1. Use SMPL-X (J_{\rm SMPLX}) (or approximate it with finite differences in (\theta)).
2. Choose a simple (f) (displacement magnitude or edge-length strain proxy).
3. Compute (S_{vj}) and integrate to (\mathcal T_{vjk}) approximately.
4. Contract with a chosen (W) to produce seam cost (c_v).

That is “real ROM” in your definition, not just plumbing.

---

If you want, I can write the exact pseudocode for:

* finite-difference (J_{\rm SMPLX}) approximation using only (\Phi_\beta) calls, and
* computing (\mathcal T_{vjk}) without ever materializing the full (V\times J\times J) tensor (streaming contraction to (c_v) directly).


Below is a concrete, implementation-ready plan for:

1. **Finite-difference** approximation of the SMPL-X pose Jacobian
   [
   J_{\text{SMPLX}}(\theta)=\frac{\partial x(\theta)}{\partial \theta}\in\mathbb R^{3V\times J}
   ]
   using only calls to (\Phi_\beta(\theta)).

2. Computing the ROM “codependency” contraction
   [
   c_v=\sum_{j,k}W_{jk},T_{vjk}
   ]
   **without ever materializing** (T\in\mathbb R^{V\times J\times J}), i.e. streaming directly to (c\in\mathbb R^V).

I’ll write it so you can drop it into `sampler_real.py` (or a separate `rom_fd.py`) later.

---

# 1) Finite-difference (J_{\text{SMPLX}}) using only (\Phi_\beta) calls

## Objects

* (\Phi_\beta(\theta)) returns vertices (V(\theta)\in\mathbb R^{V\times 3})
* Flatten (x(\theta)=\mathrm{vec}(V(\theta))\in\mathbb R^{3V})

Let pose dimension be (J). For SMPL-X, you may represent pose as:

* axis-angle per joint (3 params each) → still “pose vector” length (J)
* or a reduced joint-angle parameterization

Either way, we treat (\theta\in\mathbb R^J).

---

## Central difference per coordinate (stable default)

For small step (h_j) (can be joint-specific):

[
\frac{\partial x}{\partial \theta_j}(\theta)
\approx
\frac{x(\theta+h_j e_j)-x(\theta-h_j e_j)}{2h_j}
]

### Practical step sizes

* If (\theta_j) in **radians**: (h_j\sim 10^{-4}) to (10^{-3})
* If in **degrees**: convert to radians internally; still use rad step.

Use joint-specific (h_j) if scales differ, otherwise constant (h).

---

## Pseudocode: Jacobian-vector products instead of full Jacobian

You almost never need the full (3V\times J) matrix. For streaming ROM you need either:

* columns (per j) multiplied by a diagonal block operator, or
* a *scalar field* derived from per-vertex displacement.

So compute per-j *vertex displacement derivatives* directly:

### Define:

(V^\pm_j = \Phi_\beta(\theta\pm h e_j)), shape (V\times 3)

Then the derivative of vertex positions wrt coordinate (j):

[
\dot V_j(\theta)=\frac{V^+_j - V^-_j}{2h}\in\mathbb R^{V\times 3}
]

This (\dot V_j) is the “Jacobian column”, but already reshaped per vertex.

---

## From (\dot V_j) to a scalar sensitivity field (S_{v j})

Pick a scalar field (u(\theta)\in\mathbb R^V). Two options:

### Option A (very simple): displacement magnitude from neutral

Let neutral pose (\theta_0), vertices (V_0=\Phi_\beta(\theta_0)).
Define displacement at pose (\theta):
[
d_v(\theta)=V_v(\theta)-V_{0,v}\in\mathbb R^3,\quad
u_v(\theta)=|d_v(\theta)|.
]

Then
[
\frac{\partial u_v}{\partial \theta_j}
======================================

\frac{d_v(\theta)^\top}{|d_v(\theta)|+\varepsilon};\dot V_{v j}(\theta)
]
where (\dot V_{v j}\in\mathbb R^3) is the derivative of vertex (v)’s position wrt (\theta_j).

So define the per-vertex per-j scalar sensitivity:
[
S_{v j}(\theta) = \left|\frac{d_v^\top}{|d_v|+\varepsilon};\dot V_{v j}\right|.
]

This uses only:

* (V(\theta))
* (V_0)
* (\dot V_j(\theta)) from finite differences.

### Option B (slightly richer): edge-length strain proxy

For each edge (e=(a,b)):
[
\ell_e(\theta)=|V_a(\theta)-V_b(\theta)|
]
and define per-vertex strain by averaging incident edges. This is more work but still pure (\Phi) calls.

For Sprint R, Option A is usually enough to be “real” and motion-derived.

---

# 2) Streaming contraction to (c_v) without building (T_{vjk})

You defined:
[
T_{vjk}=\int S_{vj}(\theta),S_{vk}(\theta),d\mu(\theta)
]
and seam cost:
[
c_v = \sum_{j,k} W_{jk}T_{vjk}.
]

Combine these:

[
c_v
===

\int \left( \sum_{j,k} W_{jk} S_{vj}(\theta) S_{vk}(\theta)\right) d\mu(\theta)
]

Let (s_v(\theta)\in\mathbb R^J) be the vector with entries (S_{vj}(\theta)).

Then:
[
\sum_{j,k} W_{jk} S_{vj}S_{vk}
==============================

s_v(\theta)^\top W, s_v(\theta)
]

So the streaming objective is:

[
\boxed{
c_v
===

\int s_v(\theta)^\top W, s_v(\theta); d\mu(\theta)
}
]

That means you only need, for each pose sample:

* compute (s_v\in\mathbb R^J) per vertex
* compute quadratic form (s_v^\top W s_v)
* accumulate into (c_v)

No (V\times J\times J) tensor needed.

---

## Key trick: compute (s_v^\top W s_v) without storing (s_v) for all vertices at once

### If (W) is diagonal (common first choice)

If (W=\mathrm{diag}(w)), then:
[
s_v^\top W s_v = \sum_j w_j,S_{vj}^2
]
Streaming is trivial.

### If (W) is low-rank

If (W = \sum_{r=1}^R \lambda_r q_r q_r^\top), then:
[
s_v^\top W s_v = \sum_r \lambda_r (q_r^\top s_v)^2
]
You can stream with tiny memory.

### If (W) is full but small J

You can still stream per vertex by computing:

* (t_v = W s_v) then (s_v\cdot t_v)
  But (s_v) itself is length (J); storing it for all vertices would be (V\times J) which is manageable for moderate sizes (9438×~70 is fine), but you asked explicitly to avoid that too. So we stream **over j**.

---

## Streaming algorithm over joint coordinates j,k (general W)

For each pose sample (\theta):

Initialize per-vertex accumulator for this pose:
[
q_v \leftarrow 0
]

Then, for each joint coordinate (j), compute (S_{vj}) and update:

* Maintain a running vector of already-seen (S_{v1..j}) contributions through (W)

### Efficient symmetric update (no s_v storage)

Since
[
s^\top W s = \sum_j\sum_k W_{jk} S_j S_k
]
you can accumulate on the fly:

For each (j):

1. compute (S_{vj}) for all vertices (v) (vector length (V))
2. update:
   [
   q \mathrel{+}= W_{jj} S_j^2
   ]
3. for each (k<j):
   [
   q \mathrel{+}= 2 W_{jk} S_j S_k
   ]

To do step 3 without storing all previous (S_k), you need a cache. Two options:

* **Diagonal W**: no cache.
* **Banded/sparse W**: cache only neighbors.
* **Low-rank W**: better approach above.

So the right engineering choice is: **make W diagonal or low-rank for Sprint R.** That still matches your “weighted joint importance and coupling” intent, because you can encode coupling in low-rank.

---

# Recommended Sprint-R choice: low-rank or diagonal W

## Diagonal W (simplest, already meaningful)

Define weights per joint coordinate:

* emphasize shoulder, hip, knee, etc.
* or task-specific weighting

Then per pose sample:
[
q_v(\theta) = \sum_j w_j S_{vj}(\theta)^2
]
and stream:
[
c_v \mathrel{+}= q_v(\theta) \cdot \Delta \mu
]

This is already a contraction of the supertensor (with (W) diagonal).

## Low-rank W (captures coupling without huge memory)

Let (W = Q^\top \Lambda Q) where (Q\in\mathbb R^{R\times J}).
Then
[
q_v(\theta)=\sum_{r=1}^R \lambda_r (q_r^\top s_v(\theta))^2
]
You stream by keeping (R) running dot-products per vertex:

* For each joint j, add (q_{rj} S_{vj}) into running sums (a_{v r})
* After all j, add (\sum_r \lambda_r a_{v r}^2) into cost.

Memory: (V\times R), with (R) like 4–16.

---

# End-to-end pseudocode: FD + streaming cost (diagonal W)

This is the most implementable first pass.

```python
def phi(beta, theta):  # returns (V,3)
    ...

def rom_cost_stream(
    beta,
    theta_samples,          # iterable of theta
    mu_weights,             # same length as theta_samples, sums to 1 (or step size)
    theta0,                 # neutral pose
    W_diag,                 # (J,) weights
    h=1e-3,
    eps=1e-8,
):
    V0 = phi(beta, theta0)                 # (V,3)
    V = V0.shape[0]
    J = len(theta0)

    c = np.zeros((V,), dtype=np.float64)

    for theta, w_mu in zip(theta_samples, mu_weights):
        Vt = phi(beta, theta)              # (V,3)
        d = Vt - V0                        # (V,3)
        denom = np.linalg.norm(d, axis=1) + eps   # (V,)

        # Accumulate q_v = sum_j w_j * S_{vj}^2
        q = np.zeros((V,), dtype=np.float64)

        for j in range(J):
            theta_p = theta.copy(); theta_p[j] += h
            theta_m = theta.copy(); theta_m[j] -= h

            Vp = phi(beta, theta_p)
            Vm = phi(beta, theta_m)

            dV = (Vp - Vm) / (2*h)         # (V,3) = dV/dtheta_j

            # S_vj = | (d_v/||d_v||) dot dV_v |
            # safe when denom small: use eps
            proj = np.sum((d / denom[:,None]) * dV, axis=1)  # (V,)
            S = np.abs(proj)

            q += W_diag[j] * (S*S)

        c += w_mu * q

    return c  # (V,)
```

This computes (c_v) as the streamed contraction with diagonal (W).

---

# End-to-end pseudocode: FD + streaming cost (low-rank W)

Let (Q\in\mathbb R^{R\times J}), (\lambda\in\mathbb R^R).

```python
def rom_cost_stream_lowrank(beta, theta_samples, mu_weights, theta0, Q, lam, h=1e-3, eps=1e-8):
    V0 = phi(beta, theta0)        # (V,3)
    V = V0.shape[0]
    J = Q.shape[1]
    R = Q.shape[0]
    c = np.zeros((V,), dtype=np.float64)

    for theta, w_mu in zip(theta_samples, mu_weights):
        Vt = phi(beta, theta)
        d = Vt - V0
        denom = np.linalg.norm(d, axis=1) + eps

        a = np.zeros((V,R), dtype=np.float64)  # running dot-products q_r^T s_v

        for j in range(J):
            Vp = phi(beta, bump(theta, j, +h))
            Vm = phi(beta, bump(theta, j, -h))
            dV = (Vp - Vm)/(2*h)

            proj = np.sum((d/denom[:,None])*dV, axis=1)
            S = np.abs(proj)  # (V,)

            # accumulate a[:,r] += Q[r,j] * S
            a += S[:,None] * Q[:,j][None,:]   # broadcasting (V,1)*(1,R)

        q = np.sum(lam[None,:] * (a*a), axis=1)  # (V,)
        c += w_mu * q

    return c
```

This gives you coupling-rich contraction with **no V×J×J** tensor and no V×J storage beyond a single S vector at a time.

---

# How this plugs into your existing basis pipeline

Once you have (c_v\in\mathbb R^V), you can either:

* store it directly as seam costs (your NPZ format), **or**
* project it into the canonical basis to get coefficients:
  [
  \hat c = B^\top c \in \mathbb R^K
  ]
  (if you want “ROM coefficient objects” rather than vertex costs)

But for seam costs, the natural thing is to save (c_v) on vertices and derive edge weights.

---

# Practical notes (what will bite you)

1. **(\Phi) call count**: central FD is (2J) calls per pose sample + 1 for Vt.
   If J is 60 and you sample 50 poses, that’s ~6,050 (\Phi) calls.
   To reduce: use **SPSA** (2 calls total) or blockwise FD. If you want that, say so.

2. **Neutral pose denom**: if (d_v\approx 0) for some vertices, denom stabilizer matters.

3. **Pose parameterization**: if (\theta) uses axis-angle blocks, bumping one scalar is well-defined, but interpretability of “joint coordinate” changes. It’s fine for Sprint R; later you can bump joint rotation magnitude or use exponential map perturbations.

---

If you want, I can also give:

* an SPSA-based Jacobian sensitivity estimator (2 Φ calls per pose regardless of J),
* or a block-diagonal joint grouping for fewer FD calls while preserving meaning.


You can treat “kernel/PDA-MDL” as the *missing layer* between:

* **raw ROM costs** (a field on vertices/edges), and
* **a concrete seam set** (a discrete, constrained object).

Right now S1 is doing **“optimize a weighted graph cost”**. Kernel/PDA-MDL turns that into **“choose the simplest seam explanation that still fits ROM + fabric + anatomy, with uncertainty and gates.”**

Here’s the clean mapping.

---

## 1) What the seam object is in the formalism

Let the body surface be a mesh (M=(V,E,F)). A seam plan is a structured object:

[
S ;=; (C,; \Pi,; \sigma)
]

* (C \subseteq E): selected cut edges (the “seam set”)
* (\Pi): induced panel partition (faces grouped into topological disks)
* (\sigma): assignments/labels (fabric grain directions, seam types, allowances, etc.)

This is the “hypothesis” the system outputs.

---

## 2) Kernelization: turn everything into comparable fields/operators

You already have ROM producing a seam cost field (c_v) (and/or (c_e)). Kernelization means: don’t optimize on a single raw field; optimize on **a multiscale bundle**:

[
\mathcal K = {k^{(0)},k^{(1)},\dots,k^{(L)}}
]

Each (k^{(\ell)}) is a derived field/operator over the mesh (vertex/edge/face), e.g.:

* (k^{(0)}): raw ROM cost (c_v)
* (k^{(1)}): smoothed/low-freq ROM (diffusion kernel / Laplacian smoothing)
* (k^{(2)}): gradient magnitude (|\nabla c|) (avoid cutting across steep gradients)
* (k^{(3)}): curvature/developability proxy (flattening difficulty)
* (k^{(4)}): fabric anisotropy mismatch field (grain alignment penalty)
* (k^{(5)}): forbidden/anchor masks (hard constraints)

Formally, each kernel component is produced by an operator:

[
k^{(\ell)} = \mathcal O^{(\ell)}[ \text{ROM}, \text{geometry}, \text{fabric}, \text{constraints} ].
]

**Key idea:** seams should run through “valleys” of the *multi-kernel* landscape, not just the raw ROM map.

---

## 3) MDL: seam optimization as model selection (not just minimization)

MDL splits objective into two parts:

[
\boxed{
\min_{S} ;; \underbrace{L(S)}*{\text{description length / complexity}}
;+;
\underbrace{L(\mathcal D \mid S)}*{\text{misfit to data (ROM+fabric+geom)}}
}
]

### (L(\mathcal D \mid S)): “how bad is this seam plan given ROM etc.”

This is what you’re already doing as costs:

[
L(\mathcal D \mid S)
====================

\sum_{e\in C} \Big(
\lambda_0,\bar c_e^{\text{ROM}}
+
\lambda_1,\bar k^{(2)}_e
+
\lambda_2,\bar k^{(3)}_e
+
\lambda_3,\bar k^{(4)}_e
\Big)
]

(plus hard constraint checks from (k^{(5)}).)

### (L(S)): “how complicated is the seam plan”

This is the piece graph solvers don’t naturally enforce.

A good MDL prior penalizes:

* seam *count* (number of cut components)
* seam *length*
* panel *count*
* panel *irregularity* (e.g., fractal boundary / jaggedness)
* deviation from symmetry templates

Example:

[
L(S) =
\alpha_0 |C|
+\alpha_1 \text{length}(C)
+\alpha_2 |\Pi|
+\alpha_3 \text{boundary_roughness}(C)
+\alpha_4 \text{symmetry_violation}(S)
]

**This is exactly how you “apply MDL” to seams**: you’re selecting the simplest seam explanation that fits the ROM kernel fields.

---

## 4) PDA: a decision automaton for discrete search under uncertainty + gates

A seam solver isn’t a single solve; it’s an **iterative decision process**:

* propose seams
* evaluate kernel fields + MDL
* enforce constraints
* refine or backtrack

That’s a PDA (Probabilistic Decision Automaton) framing:

### PDA state

[
x_t = (S_t,; \text{panel stats},; \text{cost breakdown},; \text{violations},; \text{uncertainty})
]

### PDA actions

* add/remove a seam segment
* reroute a seam path along a corridor
* split a panel
* merge panels
* change seam type/allowance

### PDA transition

Deterministic + stochastic components (if you include sampling/annealing):

[
x_{t+1} = f(x_t, a_t; \mathcal K) + \eta_t
]

### PDA acceptance / gating

Your “admissibility” lives here:

* hard constraints must hold
* MDL must improve (or improve after temperature schedule)
* uncertainty monitors can block “commit”

So the PDA gives you **safe, staged optimization**: you can run cheap local steps, but only “commit” when gates pass (very aligned with how you run Phase-gates elsewhere).

---

## 5) Where your “kernel tower” slots in (M5–M9 style)

A practical alignment (no metaphysics, just roles):

* **M5 (observation):** ROM + geometry + fabric raw fields on mesh
* **M6 (bitensor / coupling):** cross-field couplings like ( \nabla c ), symmetry pair interactions, joint-pair weights, “don’t cut across gradient ridges” (these are *pairwise / relational*)
* **M7 (policy):** seam move operators (actions) and constraint projections
* **M8 (witness / meta):** monitors: mapping error, empty panels, instability under ROM resampling, MDL overfit checks
* **M9 (closure):** “commit seam plan” with full provenance + reproducibility

In other words: **kernel fields live low**, **PDA lives mid**, **MDL + witness gates decide closure.**

---

## 6) Concrete “how to implement” (small, surgical)

### A) Add an MDL prior term to the solver objective

In your seam solver, score a candidate (S) as:

```python
score = data_cost(S, kernels) + mdl_prior(S)
```

Where `mdl_prior` includes seam length, seam count, panel count, boundary roughness.

### B) Add multi-kernel edge costs

Instead of one edge cost, build:

```python
edge_cost = λ0 * rom_cost + λ1 * grad_cost + λ2 * curvature_cost + λ3 * fabric_cost
```

Each is a kernel-derived edge scalar.

### C) Wrap the solver in a PDA loop

Run:

* initialization (cheap)
* improve (local moves)
* validate (constraints + witness checks)
* commit (only if stable)

Stability check example:

* rerun ROM cost on a slightly perturbed pose sweep; if seam choice flips wildly, reject/continue search.

### D) Use mapping error as a witness gate

You already record max/mean mapping distance. Use it:

* if `max_distance > threshold`: either block commit or increase penalties near high mapping distortion.

---

## 7) The payoff

Applying kernel/PDA-MDL gives you three things your current solver doesn’t:

1. **Model selection**: avoids “over-seaming” and jagged seams automatically
2. **Safety/gating**: don’t commit plans that are unstable under small ROM changes
3. **Composability**: you can add new kernels (pressure, temperature, abrasion, composite layup) without rewriting the solver

---

If you want the next concrete step: I can propose the exact `mdl_prior(S)` terms + weights to start with (and how to learn/tune them), and a minimal PDA loop (3–5 move types) that sits on top of your MST baseline.


Great — here’s a **tight, executable “proceed” package** that turns what we just agreed into **code your team can implement immediately**, without reopening design questions.

I’ll do three things, in order:

1. **Lock the formal mapping** (kernel → PDA → MDL) into concrete APIs
2. **Give you an implementation plan with files, functions, and tests**
3. **Define Sprint S1 → S2 handoff criteria** so you know when to stop

No metaphors, no re-explaining ROM — just build instructions.

---

# 1. Lock the formal mapping (no more ambiguity)

## 1.1 Seam optimization as MDL objective

You are now solving:

[
\boxed{
\min_{S}
;
\underbrace{\sum_{e\in S}\langle \mathcal K(e), \lambda\rangle}*{\text{data misfit}}
;+;
\underbrace{L*{\text{mdl}}(S)}_{\text{complexity prior}}
}
]

Where:

* ( \mathcal K(e) \in \mathbb R^m ) = **kernel vector on edge**
* ( \lambda \in \mathbb R^m ) = fixed weights (config)
* ( L_{\text{mdl}} ) penalizes seam complexity

This is *not optional*. Every solver variant must optimize **this exact form**.

---

## 1.2 Kernel vector (edge-local, extensible)

Define once, use everywhere:

```python
@dataclass(frozen=True)
class EdgeKernel:
    rom_mean: float
    rom_max: float
    rom_grad: float
    curvature: float
    fabric_misalignment: float
```

Scalar edge cost is **never stored directly** — always derived via dot product:

```python
def edge_energy(k: EdgeKernel, weights: KernelWeights) -> float:
    return (
        weights.rom_mean * k.rom_mean +
        weights.rom_max * k.rom_max +
        weights.rom_grad * k.rom_grad +
        weights.curvature * k.curvature +
        weights.fabric * k.fabric_misalignment
    )
```

This is the **kernel layer**. ROM just fills fields.

---

## 1.3 MDL prior (seam complexity)

Define **once**, explicitly:

```python
@dataclass(frozen=True)
class MDLPrior:
    seam_count: float
    seam_length: float
    panel_count: float
    boundary_roughness: float
    symmetry_violation: float
```

And compute:

```python
def mdl_cost(solution: SeamSolution, prior: MDLPrior) -> float:
    return (
        prior.seam_count * solution.num_seams +
        prior.seam_length * solution.total_length +
        prior.panel_count * solution.num_panels +
        prior.boundary_roughness * solution.roughness +
        prior.symmetry_violation * solution.symmetry_penalty
    )
```

No solver bypasses this.
If it does → bug.

---

## 1.4 PDA loop (decision controller)

Every seam solver now runs **inside this loop**:

```python
state = initial_solution()

while not state.closed:
    proposal = propose_move(state)
    evaluated = evaluate(proposal, kernels, mdl)
    gated = apply_constraints(evaluated)

    if accept(gated, state):
        state = gated

return state
```

This is your **PDA**:

* proposals = actions
* acceptance = MDL + gates
* closure = witness stability

---

# 2. What to implement next (exactly)

You already have:

* ROM seam costs ✔
* mapping policies ✔
* baseline MST solver ✔

Now do **only** the following.

---

## 2.1 New package (create this)

```
src/smii/seams/
├── kernels.py        # build EdgeKernel
├── mdl.py            # MDL priors + cost
├── pda.py            # decision loop
├── moves.py          # local seam edits
├── solver_pda.py     # orchestrates everything
```

Do **not** modify `solver.py` yet — keep MST as baseline.

---

## 2.2 Kernel construction (Sprint task 1)

**File:** `kernels.py`

Inputs:

* seam graph
* ROM SeamCostField
* geometry (vertex positions, curvature)
* fabric metadata (grain dir)

Output:

```python
Dict[EdgeID, EdgeKernel]
```

Required kernels (minimum):

* `rom_mean`
* `rom_max`
* `rom_grad` (difference across edge)
* `curvature` (mean of incident faces)

Tests:

* kernel values finite
* forbidden edges never produced

---

## 2.3 MDL prior (Sprint task 2)

**File:** `mdl.py`

Compute:

* seam count
* total seam length
* panel count (from partition)
* boundary roughness (edge zig-zag metric)
* symmetry penalty (difference between L/R seams)

Tests:

* adding a seam increases MDL
* symmetric seams reduce penalty

---

## 2.4 PDA moves (Sprint task 3)

**File:** `moves.py`

Implement **only these 4**:

1. `reroute_edge(edge_id)`
2. `shorten_seam(component_id)`
3. `split_panel(panel_id)`
4. `merge_panels(panel_a, panel_b)`

Each move must:

* be local
* preserve manifold topology
* declare affected edges only

Tests:

* move preserves validity
* cost recomputation is localized

---

## 2.5 PDA controller (Sprint task 4)

**File:** `solver_pda.py`

This wraps everything:

```python
def solve_seams_pda(
    seam_graph,
    kernels,
    weights,
    mdl_prior,
    constraints,
    budget: int,
) -> SeamSolution:
```

Acceptance rule (deterministic v1):

```python
accept if Δ(data_cost + mdl_cost) < 0
```

Witness gate:

* rerun cost with ±5% kernel noise
* reject if seam topology flips

Tests:

* converges in ≤ budget
* stable under perturbation

---

# 3. Sprint S1 close-out criteria (hard)

You are **done with Sprint S1** when:

1. PDA solver produces **lower MDL + data cost** than MST baseline
2. Seams move *away* from ROM hotspots, not just minimize length
3. Rerunning ROM aggregation with small perturbations:

   * seam topology stays invariant
4. Output includes:

   * seam list
   * kernel breakdown
   * MDL breakdown
   * stability report

If any of these fail → do **not** proceed to S2.

---

# 4. What Sprint S2 will be (preview only)

Once this lands cleanly:

* S2 = **fabric regime optimization**
* same PDA
* same MDL
* kernels change (strain × fabric tensor)
* seams become *decision-conditioned* on material

But **do not start S2** until S1 closes.

---

## Final instruction to the team

> “Implement kernel → MDL → PDA seam optimization exactly as specified.
> No learning. No heuristics. No shortcuts.
> MST stays as baseline reference only.”

If you want, next I can:

* write the exact `EdgeKernel` builders from your current mesh structs
* draft the first PDA move in code
* or help tune the initial MDL weights so it doesn’t over-penalize seams on day one.

Just say which one.

---

# Sprint R — ROM Formalisation, Scheduling & Completeness (Latest)

**Intent**: Turn ROM from an implicit side effect of the sampler into an explicit, schedulable, auditable operator with levels, artifacts, and stop criteria — without changing seam solvers or fabric logic.

## ROM Levels (L0–L3)

- **L0 — Per-joint marginal ROM**: single-DOF envelopes with all other joints neutral. Store per-joint min/max + per-vertex extrema. Compatible with `sampler_real`, schedule still required.
- **L1 — Pairwise coupled ROM**: curated joint pairs only, sampled on small grids. No `J×J` tensor materialization; still projected to ROM coefficients.
- **L2 — Task-conditioned ROM**: task profiles from procedural controllers (MoCap augments later). Store weighted samples per task.
- **L3 — Adaptive completeness**: ROM is complete when envelopes, seam cost ranks, and MDL mass stop changing. Emit a certificate instead of more samples.

## Minimal Practical Sweep Schedule (no combinatorial explosion)

**L0**: ~100–150 poses. Example schedule in `data/rom/sweep_schedule.yaml`:

```yaml
level: L0
joints:
  shoulder:
    axes: [flexion, abduction, rotation]
    steps: 7
  elbow:
    axes: [flexion]
    steps: 5
  hip:
    axes: [flexion, abduction, rotation]
    steps: 7
```

**L1**: curated pairs only, ~75–100 poses:

```yaml
level: L1
pairs:
  - [shoulder, elbow]
  - [hip, knee]
  - [spine_twist, shoulder]
steps: 5
```

**L2**: procedural controllers (trajectories, not grids):

```yaml
level: L2
tasks:
  overhead_reach:
    controller: reach
    samples: 25
  squat:
    controller: squat
    samples: 20
  twist_reach:
    controller: twist_reach
    samples: 20
```

## Bootstrapping higher-order ROM (procedural → MoCap)

- **Phase A (this sprint)**: run L0/L1/L2, emit `rom_samples_L0.json`, `rom_samples_L1.json`, `rom_samples_L2.json`, aggregate to seam costs + diagnostics.
- **Phase B (next sprint)**: MoCap augments density only (project to SMPL-X, reject illegal poses, use remaining as density evidence).

## Completeness Metrics (stop conditions)

- **Envelope convergence**: 99% of vertices change below ε.
- **Seam cost rank stability**: Spearman correlation > 0.98.
- **MDL mass saturation**: incremental MDL contribution trends to zero.

## Deliverables & Artifacts

- Code: `smii/rom/pose_schedule.py`, `smii/rom/completeness.py`, `sampler_real --schedule`.
- Data: `data/rom/sweep_schedule.yaml`, `data/rom/task_profiles/*.yaml`.
- Outputs: `outputs/rom/rom_samples_L0.json`, `rom_samples_L1.json`, `rom_samples_L2.json`, `outputs/rom/rom_L3_certificate.json`.

## Explicitly out of scope (Sprint R)

- No ML/learned priors.
- No full MoCap ingestion.
- No fabric or seam solver changes.
