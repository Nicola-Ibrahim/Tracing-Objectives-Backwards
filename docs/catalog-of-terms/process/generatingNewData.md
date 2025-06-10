# Interpolation-Based Inverse Design Documentation

## Overview

This process enables finding optimal design parameters $X^*$ that produce objectives $f(X^*) \approx Y^*$ matching user-specified targets. Given historical data ${(X_i, Y_i)}$ where $Y_i = f(X_i)$, we:

1. Train a forward model predicting objectives from designs
2. Perform local interpolation around the target in objective space
3. Optimize within the interpolated region to find $X^*$

## 🚀 Step-by-Step Process

### Step 1: Train Forward Model

1. Collect known solutions: $ (Xᵢ, Yᵢ) $ pairs

2. Fit regression model $ f: X → Y $

3. Validate model accuracy in objective space

### Step 2: Identify Y-space Neighbors

1. User specifies target $ Y* (e.g., [0.7, 0.3]) $

2. Find k-nearest neighbors to $ Y* $ in objective space

3. Select based on Euclidean distance: $ argmin ‖Yᵢ - Y*‖ $

### Step 3: Extract Corresponding X Neighbors

1. Retrieve decision variables X₁, X₂, ..., Xₖ

2. Associated with closest Yᵢ from Step 2

### Step 4: Build Local X-space Interpolator

Fit interpolator to X neighbors

Interpolation methods:

1. **RBF**: Radial basis function network

2. **Barycentric**: Weighted average within simplex

3. **Gaussian Process**: With local kernel

### Step 5: Optimize for X*

```math
X^*= \underset{X}{\text{argmin}}  \|f(X) - Y^*\|^2
```

1. Search within interpolated X-space

2. Use gradient-based or simplex optimization

3. Constrain search to convex hull of neighbors

### Step 6: Validation

1. Evaluate true f(X*) if available

2. Verify $ ‖f(X*) - Y*‖ < \epsilon $

3. Fallback: Compare with forward model prediction

## Key Components

### **1. Forward Model ($f: X \rightarrow Y$)**

- **Purpose**: Approximates true objective function
- **Input**: $X \in \mathbb{R}^{n \times 2}$ (design parameters)
- **Output**: $Y \in \mathbb{R}^{n \times 2}$ (predicted objectives)
- **Implementation**:
  - Regression models (Ridge, RBF, Gaussian Process)
  - Trained on historical ${(X_i, Y_i)}$ pairs
  - Provides smooth predictions for unseen $X$

### **2. Target Specification ($Y^*$)**

- User-defined point in objective space (e.g., $Y^* = [0.7, 0.3]$)
- Represents desired trade-off between objectives

### **3. Neighborhood Identification**

1. Find $k$-nearest neighbors to $Y^*$ in historical objective space
   - Distance metric: Euclidean distance $\|Y_i - Y^*\|_2$
   - Typical $k$: 3-5 (balances locality and stability)
2. Retrieve corresponding design points ${X_1, X_2, ..., X_k}$

### **4. Local Interpolator**

- **Purpose**: Models continuous design space between neighbors
- **Domain**: Convex hull of ${X_1, ..., X_k}$
- **Methods**:

  ```mermaid
  graph LR
  A[Interpolation Methods] --> B[Radial Basis Functions]
  A --> C[Gaussian Process]
  A --> D[Barycentric Coordinates]
  A --> E[Local Polynomial Regression]
  ```

### **5. Optimization Process**

**Initial Setup:**

- Interpolated X-space bounds:  
  - Vertex 1: [1.0, 2.0]  
  - Vertex 2: [1.1, 2.1]  
  - Vertex 3: [0.9, 1.8]  
- Forward model: RBF with cubic kernel  
- Loss function: $L(X) = \sqrt{(f_1(X)-0.7)^2 + (f_2(X)-0.3)^2}$  

**Optimization Steps:**

1. **Initial Guess**: Barycenter of triangle  
   $X_0 = \frac{[1.0,2.0] + [1.1,2.1] + [0.9,1.8]}{3} \approx [1.0, 1.97]$  
   Predicted $f(X_0) = [0.65, 0.35]$  
   Loss: 0.071  

2. **Gradient Descent**:  
   - Iteration 1: Step toward [1.1, 2.1]  
     $X_1 = [1.03, 2.01]$  
     $f(X_1) = [0.68, 0.32]$  
     Loss: 0.028  
   - Iteration 2: Refine direction  
     $X_2 = [1.06, 2.03]$  
     $f(X_2) = [0.695, 0.305]$  
     Loss: 0.007  

3. **Convergence**:  
   Final $X^* = [1.075, 2.045]$  
   Predicted $f(X^*) = [0.702, 0.301]$  
   Loss: 0.002  

---

### **6. Validation**

**Forward Model Check**:  

- Predicted: $[0.702, 0.301]$ vs Target $[0.7, 0.3]$  
- Relative error: $\frac{\|Y^* - f(X^*)\|}{\|Y^*\|} = 0.3\%$  

**True Function Verification** (if available):  

- Simulate $X^*$ → Actual $f(X^*) = [0.71, 0.295]$  
- Practical error: 1.4% (acceptable for interpolation)  

**Edge Cases**:  

- If $X^*$ violates constraints (e.g., exits convex hull):  
  - Project back to nearest valid $X$  
  - Re-evaluate with stricter bounds  
