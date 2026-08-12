# Train-Delay Forecasting: Trustworthy AI Decision-Making

## Project Overview
This repository contains the technical foundation for a project-based learning module designed for undergraduate civil-engineering students. The objective is to move beyond teaching basic model training and focus on evaluating whether an AI prediction is sufficiently reliable for real-world infrastructure decisions.

The pipeline is built on the public [RIDE Gold Lite](https://huggingface.co/datasets/orailix/ride-gold-lite) dataset (Elliker et al., 2026): Belgian passenger railway operations with a fixed forward-in-time benchmark design - training snapshots from 2023-2024, test snapshots from 2025 - so temporal drift is inherent to the task.

## Directory Architecture
*   `data/raw/`: RIDE Gold Lite download (core evaluation table + tabular arrays, ~7.3 GB).
*   `data/prepared/`: Partially prepared student datasets (interpretable features in physical units, chronological train / calibration / test splits) plus a `column_manifest.csv` documenting every column.
*   `data/models/`, `data/predictions/`: Fitted models and test-set predictions with intervals.
*   `notebooks/`: `reliability_analysis.ipynb` - visual walkthrough of reliability diagrams, drift, and decision scenarios.
*   `src/`: Core Python scripts (see pipeline below).
*   `reports/`: Figures and the mined decision-scenario table.
*   `requirements.txt`: Pinned project dependencies.

## Running the Pipeline
Inside the project virtual environment:

```bash
python src/download_data.py        # fetch RIDE Gold Lite (core + tabular)
python src/prepare_data.py         # build train / calibration / test parquets
python src/module1_accuracy.py     # Module 1: XGBoost vs naive baseline, MAE breakdowns
python src/module2_uncertainty.py  # Module 2: MAPIE split conformal + CQR intervals
python src/module3_drift.py        # Module 3: conditional coverage, temporal drift
python src/module4_scenarios.py    # Module 4: mined engineering decision scenarios
```

Every script prints its metrics (MAE, coverage, interval width) to the console and saves its artifacts under `data/` and `reports/`.

## Technical Modules
*   **Module 1: Prediction Accuracy.** Scaffolded XGBoost workflow predicting the change in a train's delay at its next 15 scheduled events. Evaluated by MAE against a translation baseline ("delay stays the same"), overall and by horizon / delay-change bins.
*   **Module 2: Prediction Uncertainty.** Two contrasting conformal methods via MAPIE: split conformal prediction (near-constant width, marginal coverage guarantee) and conformalized quantile regression (adaptive width). Coverage, width, and a reliability diagram (nominal vs empirical coverage).
*   **Module 3: Temporal Drift & Conditional Reliability.** Chronological evaluation (calibrate on late-2024 data, test on 2025) sliced by prediction horizon, delay regime, congestion, and test month.
*   **Module 4: Engineering Decision-Making.** Real test instances where predictions are accurate but uncertain, narrow but miscalibrated, or degrade with horizon and disruption - the raw material for decision-making exercises.
