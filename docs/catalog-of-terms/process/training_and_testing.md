# Multi-Objective Optimization: Local Interpolation for Inverse Mapping

## 🧠 Overall Goal

Given:

- Pareto-optimal objective vectors:  
  $$Y = \{ y_i = (f_1^{(i)}, f_2^{(i)}) \in \mathbb{R}^2 \}_{i=1}^N$$
- Corresponding decision vectors:  
  $$X = \{ x_i \in \mathbb{R}^n \}_{i=1}^N$$
- User-provided target: $$y_{\text{target}} = (f_1^{\text{target}}, f_2^{\text{target}})$$

Estimate:  
$$x_{\text{target}} \approx \text{interpolated inverse of } y_{\text{target}}$$

## 📈 Step-by-Step Process

### 1. User Selection and Validation

- User selects arbitrary $$y_{\text{target}} = (f_1, f_2)$$
- **Validation condition**:  
  $$\min_i \| y_i - y_{\text{target}} \|_2 < \varepsilon$$
- If not satisfied: Issue warning/fallback

### 2. Neighbor Selection in Y-Space

- Find top $k$ nearest neighbors using Euclidean distance:  
  $$\mathcal{N} = \{ i \mid \text{sorted by } \| y_i - y_{\text{target}} \|_2, \text{ top } k \}$$
- Get neighboring sets:  
  $$Y_{\mathcal{N}} = \{ y_i \}_{i \in \mathcal{N}}, \quad X_{\mathcal{N}} = \{ x_i \}_{i \in \mathcal{N}}$$

### 3. Fit Local Interpolators

- Construct vector-valued interpolators:  
  $$\hat{x} = I_j(y) \in \mathbb{R}^n, \quad j = 1,2,\dots,m$$
- Common interpolators: RBF, Kriging, MLP
- Per-dimension approximation:  
  $$\hat{x}_d = I_j^{(d)}(f_1, f_2),  d = 1,\dots,n$$

### 4. Interpolation Step

- Generate candidate solutions:  
  $$\hat{x}_{\text{target}}^{(j)} = I_j(f_1^{\text{target}}, f_2^{\text{target}})$$
- Output: $$\{ \hat{x}_{\text{target}}^{(1)}, \dots, \hat{x}_{\text{target}}^{(m)} \}$$
- **Handling options**:
  - Return all candidates
  - Average solutions
  - Uncertainty-based ranking
  - Validation (next step)

### 5. Optional Refinement (Inverse Evaluation)

- If true objective function $$f(x)$$ is available:
  - Compute: $$f(\hat{x}_{\text{target}}^{(j)}) = (f_1^{(j)}, f_2^{(j)})$$
  - Select best $j^*$:  
    $$j^* = \arg\min_j \| f(\hat{x}_{\text{target}}^{(j)}) - y_{\text{target}} \|$$

## 📊 Process Flowchart

```mermaid
flowchart TD
  A["Pareto Front Data (f1, f2, x1, x2)"] --> B["Train Interpolator RBF"]
  A --> C["Train Interpolator Kriging"]
  A --> D["Train Interpolator MLP"]

  B --> B1["Train h1_RBF: (f1,f2) -> x1"]
  B --> B2["Train h2_RBF: (f1,f2) -> x2"]

  C --> C1["Train h1_Kriging: (f1,f2) -> x1"]
  C --> C2["Train h2_Kriging: (f1,f2) -> x2"]

  D --> D1["Train h1_MLP: (f1,f2) -> x1"]
  D --> D2["Train h2_MLP: (f1,f2) -> x2"]
```

```mermaid
flowchart
  E["User selects (f1*, f2*)"] --> F["Find Nearest Neighbors"]
  F --> G["Evaluate with all Interpolators"]
```

```mermaid
flowchart TD
  G["User prefernce (y1, y2)"] --> H1["Use RBF: estimate x1*, x2*"]
  G --> H2["Use Kriging: estimate x1*, x2*"]
  G --> H3["Use MLP: estimate x1*, x2*"]

  H1 --> I1["Evaluate f(x*) using true function"]
  H2 --> I2["Evaluate f(x*) using true function"]
  H3 --> I3["Evaluate f(x*) using true function"]

  I1 --> Z["Select best solution (x1*, x2*)"]
  I2 --> Z
  I3 --> Z
```
