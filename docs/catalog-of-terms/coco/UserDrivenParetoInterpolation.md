# Mapping User Preferences to Pareto-Optimal Solutions

## Step-by-Step Explanation with Mathematics

### 1. Normalize Objectives

**Purpose:** Remove scale differences between objectives (e.g., time vs. energy).

**Mathematics:**
$$
f_{i}^{\text{norm}} = \frac{f_i - f_i^{\text{min}}}{f_i^{\text{max}} - f_i^{\text{min}}}
$$

**Example:**

Original Pareto Front:
$$
\mathbf{F} = \begin{bmatrix}
10 & 200 \\
20 & 150 \\
30 & 100 \\
\end{bmatrix}, \quad
\begin{cases}
f_1^{\text{min}} = 10, \ f_1^{\text{max}} = 30 \\
f_2^{\text{min}} = 100, \ f_2^{\text{max}} = 200 \\
\end{cases}
$$

Normalized Front:
$$
\mathbf{F}_{\text{norm}} = \begin{bmatrix}
0 & 1 \\
0.5 & 0.5 \\
1 & 0 \\
\end{bmatrix}
$$

---

### 2. Compute Preference Alignment

**Purpose:** Find Pareto solutions closest to the user's desired trade-off.

**Mathematics:**

Normalize user weights:
$$
\mathbf{w}_{\text{norm}} = \frac{\mathbf{w}}{\|\mathbf{w}\|_1} = \left[\frac{w_1}{w_1 + w_2}, \frac{w_2}{w_1 + w_2}\right]
$$

Compute alignment scores via dot product:
$$
\text{Alignment}_k = \mathbf{F}_{\text{norm}}[k] \cdot \mathbf{w}_{\text{norm}}
$$

**Example:**

For $\mathbf{w} = [0.7, 0.3]$:
$$
\text{Alignment} =
\begin{bmatrix}
0 \cdot 0.7 + 1 \cdot 0.3 \\
0.5 \cdot 0.7 + 0.5 \cdot 0.3 \\
1 \cdot 0.7 + 0 \cdot 0.3 \\
\end{bmatrix} =
\begin{bmatrix}
0.3 \\
0.5 \\
0.7 \\
\end{bmatrix}
$$

---

### 3. Calculate Interpolation Parameter ($\alpha$)

**Purpose:** Map the highest-alignment solution to a position along the Pareto set.

**Mathematics:**
$$
\alpha = \frac{\text{argmax}(\text{Alignment})}{N - 1}
$$

**Example:**

For $N = 3$ solutions and max alignment at index 2:
$$
\alpha = \frac{2}{3 - 1} = 1.0
$$

---

### 4. Interpolate in Decision Space

**Purpose:** Generate smooth transitions between Pareto-optimal solutions.

**Mathematics (Linear Interpolation):**

Given two neighboring solutions $x_i$ and $x_{i+1}$:
$$
x_{\text{interp}} = (1 - \alpha_{\text{frac}}) \cdot x_i + \alpha_{\text{frac}} \cdot x_{i+1}
$$

**Example:**

If $\alpha = 0.75$ falls between $x_2$ and $x_3$:
$$
x_{\text{interp}} = 0.25 \cdot x_2 + 0.75 \cdot x_3
$$

---

## Why We Need $\alpha$ Before Interpolation?

### Parameterization

$\alpha \in [0,1]$ uniformly parameterizes the Pareto set from the solution optimal for $f_1$ ($\alpha = 0$) to $f_2$ ($\alpha = 1$).

### Continuity

Enables smooth interpolation between neighboring solutions, even if the user’s preference doesn’t exactly match a precomputed solution.

### Efficiency

Avoids re-solving the optimization problem for every new preference.

---

## Complete Workflow Example

**Input:**

- User weights $\mathbf{w} = [0.7, 0.3]$
- Pareto front $\mathbf{F}$ and set $\mathbf{X}$

**Steps:**

1. Normalize $\mathbf{F}$ → $\mathbf{F}_{\text{norm}}$
2. Compute alignment scores → $[0.3, 0.5, 0.7]$
3. Find $\alpha = 1.0$
4. Return $x_3$ (no interpolation needed here)

**Result:** The solution optimal for $f_1$ (third solution in the Pareto set).

---

## Key Insight

The interpolation parameter $\alpha$ acts as a bridge between:

- **User Preferences** (weights in objective space)
- **Decision Variables** (parameters in design space)

By converting preferences to positions along the Pareto front ($\alpha$), we enable efficient exploration of optimal trade-offs.
