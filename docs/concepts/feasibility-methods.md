# Local Spherical Neighborhood Feasibility (LSNF)

## 🧠 Core Idea

Define a sphere (ball) of radius $ r $ around each normalized Pareto point. A point is feasible if it lies within **any** such sphere. This ensures:

- **Smoothness**: Continuous transition across boundaries
- **Rotation-invariance**: No axis-aligned bias (unlike cubes)
- **Natural decay**: Score decreases smoothly with distance from the front

---

## 📐 Mathematical Definition

### Inputs

- $ P_{\text{norm}} = \{ p_1, p_2, \ldots, p_n \} \subset \mathbb{R}^d $: Normalized Pareto front
- $ r \in \mathbb{R}^+ $: Radius hyperparameter

### Spherical Neighborhoods

For each point $ p_i \in P_{\text{norm}} $, define:
$$
B(p_i, r) = \{ y \in \mathbb{R}^d : \| y - p_i \|_2 \leq r \}
$$
where $ \| \cdot \|_2 $ is the Euclidean norm.

### Feasible Region

$$
B_r = \bigcup_{i=1}^n B(p_i, r)
$$

---

## ✅ Feasibility Rule

### Binary Decision

Given target point $ y^* \in \mathbb{R}^d $:
$$
\text{Feasible} \iff y^* \in B_r
$$

### Soft Score Function

$$
s(y^*) = \max_i \left[ 1 - \frac{ \| y^* - p_i \|_2 }{ r } \right]_+
$$
where $ [x]_+ = \max(0, x) $. This yields:

- $ s(y^*) = 1 $: At center of a sphere
- $ s(y^*) \approx 0 $: On boundary
- $ s(y^*) = 0 $: Outside all spheres

---

## 🔍 Interpretation

### Key Properties

| Score | Meaning |
|-------|---------|
| 1.0   | Perfect match with a Pareto point |
| 0.5   | Midway between center and boundary |
| 0.0   | Infeasible region |

---

## 📘 Comparison Table

| Aspect                | **LSNF (Spheres)**                          | **LCNF (Cubes)**                          | **KDE**                                  | **Convex Hull + Buffer**          |
|-----------------------|---------------------------------------------|-------------------------------------------|------------------------------------------|-----------------------------------|
| **Feasibility Region** | Union of spheres (smooth, isotropic)        | Union of cubes (boxy, axis-aligned)       | Continuous density field                 | Convex shape + Euclidean buffer   |
| **Mathematical Core**  | $ \| y^* - p_i \|_2 \leq r $              | $ \| y^* - p_i \|_\infty \leq \delta $  | $ \hat{f}(y^*) = \sum K((y^* - p_i)/h) $ | $ d(y^*, \text{Conv}(P)) \leq \delta $ |
| **Score Function**     | $ 1 - \frac{\| y^* - p_i \|_2}{r} $       | $ 1 - \frac{\| y^* - p_i \|_\infty}{\delta} $ | $ \hat{f}(y^*) / \max_i \hat{f}(p_i) $ | $ 1 - \frac{d(y^*)}{\delta} $   |
| **Boundary Behavior**  | Smooth (radial symmetry)                    | Sharp corners, axis-biased                | Smooth, depends on kernel bandwidth      | Sharp boundary, convexity-dependent |
| **Dimensional Robustness** | ✅ Handles high dimensions               | ✅ Yes                                    | ❌ Hard to tune kernel bandwidth         | ❌ Unstable in high $ d $        |
| **Computational Cost** | ✅ Efficient (distance check)              | ✅ Efficient (cube overlap)                | ❌ Slow in high $ d $, scaling issues  | ❌ Expensive (QP projections)     |
| **Interpretability**   | ✅ High                                    | ✅ High                                   | Medium                                   | ✅ High                           |
| **Sensitivity**        | Controlled by $ r $                      | Controlled by $ \delta $                 | Controlled by bandwidth $ h $          | Controlled by buffer $ \delta $  |

---

## 🧠 Why LSNF May Be Preferable

### Key Advantages

- **Rotation-invariant**: Spheres treat all directions equally
- **No kernel tuning**: Explicit scoring vs KDE's bandwidth sensitivity
- **Natural decay**: Score reflects proximity to the front
- **Non-convex compatibility**: Handles irregular/disjoint fronts

---

## 🧾 Feasibility Checker Output Format

```python
{
    "is_feasible": True,
    "score": 0.92,
    "method": "LSNF",
    "distance_to_front": 0.07,
    "nearest_point": [0.32, 0.81],
    "suggestions": [...]
}
```
