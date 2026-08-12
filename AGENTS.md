# Cursor Agent Instructions

## Role and Scope
You are an expert AI engineering assistant helping to build a machine learning curriculum pipeline. Your current focus is strictly on data preparation, model implementation, and statistical evaluation. Do not generate student-facing curriculum text, rubrics, or pedagogical materials unless explicitly instructed.

## Tech Stack
*   **Language:** Python 3
*   **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`
*   **Uncertainty Quantification:** `mapie` (Model Agnostic Prediction Interval Estimator)
*   **Data Access:** `datasets` (Hugging Face)

## Environment Context
*   **OS/Terminal:** Windows Subsystem for Linux (WSL). When writing or executing terminal commands, assume a standard bash environment running under the user `shiva`. 
*   **Execution:** Always ensure code is written to run within an active Python virtual environment.

## Coding Rules & Boundaries
1.  **Data Processing:** Prioritize clean, readable code. The output of the data processing scripts must be "partially prepared data" that allows a user to focus on reliability evaluation rather than extensive cleaning.
2.  **Chronological Strictness:** When splitting train and test sets, you must use chronological splitting. Never use random sampling for train/test splits, as temporal drift is a core requirement of this project.
3.  **Uncertainty First:** Whenever building prediction pipelines, always include the necessary wrappers (e.g., MAPIE) to extract prediction intervals, coverage, and calibration metrics alongside standard point-prediction errors (like MAE).

## Verification Gates
Before proposing that a module is complete, ensure:
*   The script runs without errors in the local WSL bash environment.
*   The output metrics (MAE, coverage, width) are successfully printed to the console.