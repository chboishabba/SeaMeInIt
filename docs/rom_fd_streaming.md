# ROM Jacobian estimation and streaming seam-cost contraction

This note captures the finite-difference (FD) approach to approximate SMPL-X
pose Jacobians and stream seam costs without materializing large tensors.

## ROM operator recap

Given a canonical basis `B ∈ R^{V×K}` and a per-vertex field `u(θ)=f(Φβ(θ))`,
the ROM operator is `R(θ)=Bᵀ u(θ) ∈ R^K`. Aggregating over admissible poses
approximates integrals of `R(θ)` (mean, covariance) for ROM-aware costs.

## FD approximation of SMPL-X pose Jacobian

We only need access to `Φβ(θ)` (pose → vertices). For pose dimension `J`, use
central differences per coordinate:

```
V⁺ = Φβ(θ + h e_j)
V⁻ = Φβ(θ - h e_j)
Ṽ_j = (V⁺ - V⁻) / (2h)       # shape (V,3)
```

Pick `h` in radians (e.g., 1e-3). Neutral vertices `V₀ = Φβ(θ₀)` are reused.

### Per-vertex scalar sensitivity

Using displacement magnitude from neutral:

```
d = V(θ) - V₀                 # (V,3)
denom = ||d||₂ + ε            # (V,)
S_vj = | (d_v / denom_v) · Ṽ_vj |   # scalar sensitivity per vertex/joint
```

Other fields (e.g., edge strain) can replace this proxy; the chain rule form is
unchanged.

## Streaming seam-cost contraction (no V×J×J tensor)

Define joint weight matrix `W` (diagonal or low-rank recommended). For a pose
sample with sensitivities `S_{vj}`, the per-vertex quadratic form is:

```
q_v = s_vᵀ W s_v    where s_v = [S_{v1}, …, S_{vJ}]
```

Accumulate seam cost over pose samples with measure weights `w(θ)`:

```
c_v ← c_v + w(θ) * q_v
```

### Diagonal W (simplest)

```
q_v = Σ_j W_j * S_{vj}²
```

### Low-rank W (captures coupling, memory-light)

Let `W = Qᵀ Λ Q` with `Q∈R^{R×J}`, `Λ=diag(λ)`. Stream dot products:

```
a_vr = Σ_j Q_{rj} * S_{vj}
q_v  = Σ_r λ_r * a_vr²
```

This stores only `V×R`, not `V×J×J`.

## Pseudocode (diagonal W)

```
V0 = Phi(beta, theta0)
c  = zeros(V)
for theta, w in samples:
  Vt = Phi(beta, theta); d = Vt - V0; denom = ||d|| + eps
  q = zeros(V)
  for j in range(J):
    Vp = Phi(beta, theta + h e_j)
    Vm = Phi(beta, theta - h e_j)
    dV = (Vp - Vm)/(2*h)
    S  = abs(((d/denom) * dV).sum(axis=1))
    q += W_diag[j] * S * S
  c += w * q
return c  # per-vertex seam cost proxy
```

This is a direct contraction of the ROM supertensor with diagonal `W`, streamed
over poses. Low-rank `W` swaps the inner loop per the dot-product sketch above.
