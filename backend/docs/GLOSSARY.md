# 📖 Technical Glossary

This glossary defines key terms and project-specific concepts used across the **Tracing Objectives Backwards** ecosystem.

## 🔄 Inverse Mapping Concepts

### **Inverse Design**
The process of determining the input parameters (**X**) that yield a desired performance output (**Y**). Unlike traditional optimization, it often deals with "one-to-many" mappings where multiple inputs can result in the same output.

### **Decision Space (X)**
The multi-dimensional space comprising all possible design parameters or variables that can be modified (e.g., geometry, material properties).

### **Objective Space (Y)**
The multi-dimensional space comprising the performance metrics or outcomes we want to achieve (e.g., efficiency, drag, cost).

### **Pareto Front**
A set of non-dominated designs where no single objective can be improved without degrading at least one other objective.

## 🧠 AI & Modeling

### **GPBI (Global Pareto-Based Inverse)**
A custom algorithm (implemented in this project) that leverages the Pareto front structure to guide the synthesis of design candidates.

### **MDN (Mixture Density Network)**
A neural network architecture that predicts the parameters of a probability distribution (usually a Gaussian Mixture) rather than a single point value. Crucial for handling multi-modal inverse mappings.

### **CVAE (Conditional Variational Autoencoder)**
A generative model that learns to map high-dimensional data to a latent space conditioned on specific labels (objectives), allowing for controlled generation of new designs.

### **PIT (Probability Integral Transform)**
A diagnostic tool used to assess if a predictive distribution is well-calibrated.

### **MACE (Mean Absolute Calibration Error)**
A metric used to quantify the calibration error of a probabilistic model.

## 🏗️ Architecture

### **Bounded Context**
A central pattern in **Domain-Driven Design (DDD)**. It defines a boundary within which a particular domain model is defined and applicable. In our backend, `dataset`, `modeling`, and `evaluation` are distinct bounded contexts.

### **Modular Monolith**
An architectural style where the system is physically a single deployment unit but logically divided into highly decoupled, independent modules.

---
*Missing a term? Open a PR or an issue!*
