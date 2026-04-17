# Conditional VAE (CVAE) Estimator

## Purpose

Model the inverse mapping as a conditional distribution p(x|y). A CVAE captures one-to-many relationships by sampling multiple plausible decisions x given a target objective y.

## Inputs and outputs

- Condition: objective vector y
- Target: decision vector x
- Latent: z (captures multi-modality)

## Training objective (ELBO)

The CVAE maximizes the evidence lower bound:

$$
L = E_{q_phi(z|x,y)}[log p_theta(x|z,y)] - beta * KL(q_phi(z|x,y) || p(z))
$$

## Inference

At query time, sample z from the prior p(z), then decode with the target y to generate candidate decisions x_hat.

```mermaid
%%{init: {
  "theme": "default",
  "themeVariables": {
    "primaryColor": "#E3F2FD",
    "primaryBorderColor": "#1E88E5",
    "primaryTextColor": "#1565C0",
    "secondaryColor": "#E8F5E9",
    "secondaryBorderColor": "#43A047",
    "secondaryTextColor": "#2E7D32",
    "tertiaryColor": "#FFF8E1",
    "tertiaryBorderColor": "#FB8C00",
    "tertiaryTextColor": "#EF6C00",
    "lineColor": "#607D8B",
    "fontFamily": "Segoe UI, sans-serif",
    "fontSize": "16px"
  }
}}%%
flowchart LR
  subgraph TRAIN["Training"]
    direction LR
    x_in["Decision x"] --> enc["Encoder q_phi(z|x,y)"]
    y_in["Condition y"] --> enc
    enc --> mu["mu"]
    enc --> logvar["logvar"]
    mu --> reparam["z = mu + eps * exp(0.5 * logvar)"]
    logvar --> reparam
    reparam --> dec["Decoder p_theta(x|z,y)"]
    y_in --> dec
    dec --> x_hat["Reconstruction x_hat"]
    x_in -.-> recon_loss["Reconstruction loss"]
    x_hat -.-> recon_loss
    mu -.-> kl_loss["KL loss"]
    logvar -.-> kl_loss
  end

  subgraph INFER["Inference"]
    direction LR
    y_star["Target y*"] --> prior["Sample z ~ p(z)"]
    prior --> dec_inf["Decoder p_theta(x|z,y*)"]
    y_star --> dec_inf
    dec_inf --> x_star["Candidates x_hat"]
  end
```

## When to use

- Multi-modal inverse mappings where multiple decisions satisfy the same objective.
- Interactive exploration that benefits from diverse candidate proposals.
