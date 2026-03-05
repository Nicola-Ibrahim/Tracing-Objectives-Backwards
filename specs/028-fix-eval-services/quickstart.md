# Quickstart: Evaluation Service Fixes

## Overview
This feature branch (`028-fix-eval-services`) rectifies the data structural mismatches between the backend diagnostic services and the frontend chart expectations.

## Running the Changes

1. Ensure the backend and frontend are running:
   ```bash
   poe serve
   npm run dev
   ```

2. Generate/Train at least two candidate engines on the same dataset (e.g., `cocoex_f2`).
3. Navigate to **Evaluation** in the sidebar.
4. Select the dataset and multiple engines.
5. Click **Run Comparative Diagnosis**.
6. The charts should render completely without frontend `undefined` iteration errors or backend 500 server errors.
