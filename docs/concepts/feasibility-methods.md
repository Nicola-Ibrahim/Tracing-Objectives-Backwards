# 🛡️ Feasibility Validation Methods

This document details the strategies used to determine if a target objective $Y^*$ is reachable and trustworthy within the learned inverse mapping.

---

## 🧠 Local Spherical Neighborhood Feasibility (LSNF)

The primary method for feasibility checking. It defines a "trust zone" around the observed Pareto points.

### Core Idea
Define a sphere (ball) of radius $ r $ around each normalized Pareto point. A point is feasible if it lies within **any** such sphere. This ensures:

- **Smoothness**: Continuous transition across boundaries.
- **Rotation-invariance**: No axis-aligned bias (unlike cubes).
- **Natural decay**: Score decreases smoothly with distance from the front.

---

## 📐 Mathematical Definition

### Inputs
- $ P_{\text{norm}} = \{ p_1, p_2, \ldots, p_n \} \subset \mathbb{R}^d $: Normalized Pareto front.
- $ r \in \mathbb{R}^+ $: Radius hyperparameter.

### Spherical Neighborhoods
For each point $ p_i \in P_{\text{norm}} $, define:
$$
B(p_i, r) = \{ y \in \mathbb{R}^d : \| y - p_i \|_2 \leq r \}
$$

---

## ✅ Feasibility Rule

### Binary Decision
Given target point $ y^* \in \mathbb{R}^d $:
$$
\text{Feasible} \iff y^* \in \bigcup_{i=1}^n B(p_i, r)
$$

### Soft Score Function
$$
s(y^*) = \max_i \left[ 1 - \frac{ \| y^* - p_i \|_2 }{ r } \right]_+
$$

| Score | Meaning |
|-------|---------|
| **1.0** | Perfect match with an observed Pareto point. |
| **0.5** | Midway between the center and the trusted boundary. |
| **0.0** | Infeasible region; the target is too far for reliable prediction. |

---

## 📘 Comparison of Methods

| Aspect | **LSNF (Spheres)** | **LCNF (Cubes)** | **KDE** |
|---|---|---|---|
| **Geometry** | Union of spheres | Union of cubes | Continuous density |
| **Bias** | None (Isotropic) | Axis-aligned | Bandwidth-dependent |
| **Cost** | ⚡ Low | ⚡ Low | 🐢 High in high dims |
| **Closeness** | Euclidean distance | Infinity norm | Kernel density |

---

## 🧾 Implementation Detail

The `FeasibilityChecker` returns a structured validation result:

```python
{
    "is_feasible": True,
    "score": 0.92,
    "method": "LSNF",
    "distance_to_front": 0.07,
    "nearest_point": [0.32, 0.81],
    "suggestions": [...] # Nearby feasible points if infeasible
}
```
