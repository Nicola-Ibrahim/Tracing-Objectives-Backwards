# ⚛️ Frontend: Inverse Exploration Dashboard

[![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat&logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Docker](https://img.shields.io/badge/Infrastructure-Docker-blue.svg)](Dockerfile)
[![ESLint](https://img.shields.io/badge/ESLint-4B32C3?style=flat&logo=eslint&logoColor=white)](https://eslint.org/)

The interactive command center for **Tracing Objectives Backwards**. This React-based dashboard provides a high-fidelity interface for exploring Pareto-optimal manifolds, training AI models via the web, and interactively proposing design candidates.

---

## 🎨 Key Technical Features

- **📊 High-Dimensional Visualization**: Leverages **Plotly** and **Recharts** to render complex decision and objective spaces with interactive filtering and selection. We use high-concurrency data fetching to ensure smooth exploration of large Pareto fronts.
- **🔄 Candidate Manifold Explorer**: A specialized interface for traversing the latent space of generative models (GPBI, MDN), allowing researchers to "trace" performance targets back to design parameters.
- **🌑 Premium Theme Engine**: A robust dark-mode UI built with **Tailwind CSS** and **Framer Motion**, prioritizing clarity and visual excellence for engineering data. Its adaptive design ensures productivity across all device sizes.
- **🎣 Type-Safe API Interaction**: Full **TypeScript** integration with the FastAPI backend, ensuring reliable data flow and runtime safety for all inverse mapping operations.

---

## 🏗️ Architecture Overview

The frontend follows a **Feature-Based Module** structure, isolating domain-specific views from shared UI logic.

- **`features/`**: Contains self-contained modules like `TrainEngine`, `DatasetExplorer`, and `CandidateGeneration`.
- **`components/`**: Atomic UI library (cards, buttons, charts) used project-wide.
- **`hooks/lib/`**: Interaction layer for API communication and global state management, optimized for real-time AI status updates.

---

## 🚦 Getting Started (Local)

### Prerequisites
- [Node.js](https://nodejs.org/) (v18+)
- [npm](https://www.npmjs.com/) or [yarn](https://yarnpkg.com/)

### Installation
```bash
npm install
```

### Development Server
```bash
npm run dev
```
Access the dashboard at `http://localhost:3000`.

### 🐳 Dockerization

To run the frontend as a standalone container:

**1. Build the image:**
```bash
docker build -t tob-frontend .
```

**2. Run the container:**
```bash
docker run -p 3000:3000 tob-frontend
```

---

## 📖 Extended Knowledge

For a deeper look into the frontend patterns and architectural vision:
- 🏛️ **[Central Architecture Blueprint](../docs/architecture/README.md)**
- 🧭 **[Developer Knowledge Portal](../docs/README.md)**

---
Related: [Root README](../README.md) | [Backend README](../backend/README.md)
