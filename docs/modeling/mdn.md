# Mixture Density Network (MDN) Estimator

## Purpose

An MDN models p(x|y) as a mixture of Gaussians. It outputs mixture weights, means, and variances, which enables sampling multiple candidate decisions for a single target objective.

## Training objective

Given pairs (x, y), the MDN learns mixture parameters and minimizes the negative log-likelihood:

$$
L = -log\left(\sum_{k=1}^K pi_k(y) * N(x | mu_k(y), sigma_k(y))\right)
$$

## Inference

Given a target y*, the MDN produces mixture parameters and samples K candidate decisions x_hat.

```mermaid
%%{init: {
  "theme": "default",
  "themeVariables": {
    "primaryColor": "#E3F2FD",
    "primaryBorderColor": "#1E88E5",
    "primaryTextColor": "#1565C0",
    "secondaryColor": "#FFF3E0",
    "secondaryBorderColor": "#FB8C00",
    "secondaryTextColor": "#EF6C00",
    "tertiaryColor": "#E8F5E9",
    "tertiaryBorderColor": "#43A047",
    "tertiaryTextColor": "#2E7D32",
    "lineColor": "#607D8B",
    "fontFamily": "Segoe UI, sans-serif",
    "fontSize": "16px"
  }
}}%%
flowchart LR
  subgraph TRAIN["Training"]
    direction LR
    y_in["Objective y"] --> mdn["MDN network"]
    mdn --> mix["Mixture params: pi_k, mu_k, sigma_k"]
    x_in["Decision x"] --> nll["NLL loss"]
    mix --> nll
  end

  subgraph INFER["Inference"]
    direction LR
    y_star["Target y*"] --> mdn_inf["MDN network"]
    mdn_inf --> mix_inf["Mixture params: pi_k, mu_k, sigma_k"]
    mix_inf --> sample["Sample K candidates x_hat"]
    sample --> rank["Filter and rank by forward check"]
  end
```

## When to use

- Inverse mapping is multi-valued and you want diverse candidates.
- You want a compact probabilistic output without a latent variable model.
