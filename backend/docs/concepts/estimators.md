# Estimators Documentation

This document provides an overview of the **ml_mappers** module, detailing its deterministic and probabilistic models, their objectives, loss functions, and associated mathematical formulations. The goal is to enable users to understand and implement these models effectively for inverse mapping tasks.

---

## Table of Contents

1. [Deterministic Models](#deterministic-models)
2. [Probabilistic Models](#probabilistic-models)
3. [Conclusion](#conclusion)

---

## Deterministic Models

Deterministic models produce a single output for a given input without considering uncertainty. These models are implemented in the `deterministic` subfolder.

### Model Overview

| Model Name          | Objective                          | Loss Function       | Formula                                                                 |
|---------------------|------------------------------------|---------------------|-------------------------------------------------------------------------|
| Clough-Tocher       | Interpolate scattered data         | MSE                 | $\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$                        |
| Gaussian Process    | Model complex patterns with kernels| Log-likelihood      | $\log p(\mathbf{y} \| \mathbf{X}, \theta)$                             |
| Kriging             | Spatial prediction with covariance | Log-likelihood      | $\log p(\mathbf{y} \| \mathbf{X}, \theta)$                             |
| Linear              | Fit linear relationships           | MSE                 | $\frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{w}^\top \mathbf{x}_i)^2$     |
| Nearest Neighbors   | Predict based on closest samples   | MAE                 | $\frac{1}{N} \sum_{i=1}^N \|y_i - \hat{y}_i\|$                        |
| Neural Network (NN) | Learn nonlinear mappings           | MSE or Custom       | $\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \ell(y_i, f(\mathbf{x}_i))$   |
| RBF                 | Radial basis function interpolation| MSE                 | $\frac{1}{N} \sum_{i=1}^N (y_i - \sum_j w_j \phi(\|\mathbf{x}_i - c_j\|))^2$ |
| Spline              | Smooth piecewise polynomial fitting| MSE                 | $\frac{1}{N} \sum_{i=1}^N (y_i - S(\mathbf{x}_i))^2$                   |
| SVR                 | Robust regression with margins     | ε-insensitive       | $\sum_{i=1}^N \max(0, \|y_i - f(\mathbf{x}_i)\| - \epsilon)^2$        |

### Key Formulations

- **MSE (Mean Squared Error)**:  
  $$
  \mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
  $$
- **MAE (Mean Absolute Error)**:  
  $$
  \mathcal{L}_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^N \|y_i - \hat{y}_i\|
  $$
- **ε-insensitive Loss (SVR)**:  
  $$
  \mathcal{L}_{\epsilon} = \sum_{i=1}^N \max(0, \|y_i - f(\mathbf{x}_i)\| - \epsilon)^2
  $$

---

## Probabilistic Models

Probabilistic models capture uncertainty in predictions by modeling outputs as probability distributions. These models are in the `probabilistic` subfolder.

### Model Overview

| Model Name | Objective                          | Loss Function               | Formula                                                                 |
|------------|------------------------------------|-----------------------------|-------------------------------------------------------------------------|
| CVAE       | Generative modeling with priors    | ELBO (Evidence Lower Bound) | $\mathcal{L}_{\text{CVAE}} = \mathbb{E}_{q_\phi(z\|x)}[\log p_\theta(x\|z)] - \text{KL}(q_\phi(z\|x) \| p(z))$ |
| MDN        | Mixture density estimation         | Negative Log-Likelihood     | $-\log \left( \sum_{k=1}^K \pi_k \mathcal{N}(y \| \mu_k, \sigma_k^2) \right)$ |

### Key Formulations

- **ELBO (CVAE)**:  
  $$
  \mathcal{L}_{\text{CVAE}} = \mathbb{E}_{q_\phi(z\|x)}[\log p_\theta(x\|z)] - \beta \cdot \text{KL}(q_\phi(z\|x) \| p(z))
  $$
  where $\beta$ balances reconstruction and regularization.

- **Mixture Density Loss (MDN)**:  
  $$
  \mathcal{L}_{\text{MDN}} = -\sum_{i=1}^N \log \left( \sum_{k=1}^K \pi_k^{(i)} \mathcal{N}(y_i \| \mu_k^{(i)}, \sigma_k^{(i)}) \right)
  $$

---

## 🔁 Comparison Table

| Model     | Type         | Multimodal? | Uncertainty? | Sample Efficient? | Expressive? |
|-----------|--------------|-------------|--------------|-------------------|-------------|
| Linear    | Deterministic| ❌          | ❌           | ✅                | ❌          |
| SVR       | Deterministic| ❌          | ✅ (margin)  | ✅                | ⚠️          |
| RBF       | Deterministic| ❌          | ❌           | ✅                | ✅          |
| GPR       | Deterministic| ❌          | ✅           | ✅                | ✅          |
| KNN       | Deterministic| ❌          | ❌           | ✅                | ❌          |
| NN        | Deterministic| ❌          | ❌           | ⚠️                | ✅✅         |
| MDN       | Probabilistic| ✅          | ✅           | ⚠️                | ✅✅✅        |
| CVAE      | Probabilistic| ✅          | ✅           | ❌

---

## Conclusion

The `ml_mappers` module offers a comprehensive suite of models for inverse mapping tasks, ranging from simple interpolators (e.g., Nearest Neighbors) to complex probabilistic frameworks (e.g., CVAE). Users should choose models based on their specific requirements for accuracy, interpretability, and uncertainty quantification. For implementation details, refer to the source code in the respective Python files.
