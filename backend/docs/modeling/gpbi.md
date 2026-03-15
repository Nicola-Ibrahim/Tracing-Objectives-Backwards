# 🧬 GPBI: Global Pareto-Based Inverse

## 🌟 Overview

The **Global Pareto-Based Inverse (GPBI)** is our proposed algorithm for high-fidelity inverse design. Unlike local interpolators, GPBI leverages the global geometric structure of the Pareto front to provide more stable and physically consistent design proposals.

## 🚀 How it Works

GPBI operates on the principle of **Global Geometric Alignment**. Instead of fitting a local surrogate model, it partitions the objective space based on the density of known Pareto-optimal solutions and uses a global mapping function to estimate the design variables.

### Key Steps:
1. **Global Normalization**: All Pareto data is scaled to a global unit hypercube.
2. **Structural Partitioning**: The objective space is decomposed into regions of similar geometric curvature.
3. **Inverse Mapping**: A global non-linear estimator maps $Y \rightarrow X$ across the entire front.
4. **Feasibility Projection**: Proposals are projected onto the nearest manifold boundary to ensure they remain within the design space.

## 🎯 Advantages

- **Stability**: Less sensitive to local noise in the Pareto data compared to RBF or Kriging.
- **Global Awareness**: Better handles disjoint Pareto fronts or regions with high non-linearity.
- **Efficiency**: Requires fewer data points to achieve comparable accuracy in high-dimensional decision spaces.

## 🛠 Usage in the Framework

To use the GPBI estimator, specify it in your training configuration:

```python
estimator = GPBIEstimator(
    latent_dim=10,
    reg_weight=0.01,
    n_partitions=5
)
```

## 📊 Performance Comparison

| Metric | RBF/Kriging | CVAE/INN | **GPBI** |
| :--- | :--- | :--- | :--- |
| **Stability** | Medium | Low | **High** |
| **Data Efficiency** | High | Low | **High** |
| **Multimodality** | Low | High | **Medium** |
