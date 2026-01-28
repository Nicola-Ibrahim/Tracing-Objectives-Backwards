# Invertible Neural Network (INN) Estimator

## Purpose

An INN learns a bijection between decisions x and outputs (y, z). The latent z captures information lost in the forward mapping, allowing inversion from (y*, z) to x_hat.

## Training signals

- Forward consistency: f_inn(x) -> (y_hat, z_hat)
- Inverse consistency: f_inn^{-1}(y, z) -> x_hat
- Distribution matching on z to encourage a simple prior (e.g., standard normal)

```mermaid
%%{init: {
  "theme": "default",
  "themeVariables": {
    "primaryColor": "#E3F2FD",
    "primaryBorderColor": "#1E88E5",
    "primaryTextColor": "#1565C0",
    "secondaryColor": "#FFEBEE",
    "secondaryBorderColor": "#E53935",
    "secondaryTextColor": "#C62828",
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
    x_in["Decision x"] --> inn_fwd["INN forward"]
    inn_fwd --> y_hat["y_hat"]
    inn_fwd --> z_hat["z_hat"]
    y_true["Objective y"] --> y_loss["Forward loss"]
    y_hat --> y_loss
    z_hat --> z_loss["Latent matching"]
  end

  subgraph INFER["Inference"]
    direction LR
    y_star["Target y*"] --> z_prior["Sample z ~ N(0, I)"]
    z_prior --> inn_inv["INN inverse"]
    y_star --> inn_inv
    inn_inv --> x_star["Candidates x_hat"]
  end
```

## When to use

- You want exact invertibility in the architecture.
- You need fast inverse sampling conditioned on a target objective.
