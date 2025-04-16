# Heuristic Function Classifier (A* 8-Puzzle)

This project classifies which heuristic function (Manhattan, Euclidean, Hamming, or Linear Conflict) was used in an A* search trace on 8-puzzle problems using a trained ML model.

---

## Files Overview

| File                             | Purpose                                               |
|----------------------------------|-------------------------------------------------------|
| `generate_eval_dataset.py`       | Generates synthetic A* episodes and features         |
| `model_v2.py`                    | Trains classifiers and saves final ensemble model    |
| `evaluate_model_on_eval_data.py` | Evaluates model predictions on test data             |
| `make_model_input.py`            | Defines heuristic functions and state generators     |
| `a_star.py`                      | A* implementation used internally                    |
| `data_gen.py`                    | Legacy generation script (optional)                  |

---

## How to Use

### Step 1: Generate Synthetic Data
```bash
python generate_eval_dataset.py
```
This creates 10,000 episodes, each using a random heuristic, and saves them in `eval_data/` folder.

---

### Step 2: Train the Model
```bash
python model_v2.py <data_file name after using data_gen.py>
```

- Trains 4 classifiers: RandomForest, MLP, XGBoost, LightGBM
- Uses `GridSearchCV` for tuning
- Applies class reweighting and SHAP pruning
- Uses `StackingClassifier` to combine base models
- Saves trained model + encoder + feature indices in `best_model.pkl`

---

### Step 3: Evaluate the Model
```bash
python evaluate_model_on_eval_data.py
```
- Predicts heuristic used per episode
- Compares predictions to ground-truth
- Prints accuracy and confusion matrix

---

## Features Used

Each episode is described by 28 features including:
- Path cost, node expansions, branching factor
- Heuristic value statistics (mean, variance, min/max)
- Clustering, directionality, heuristic delta smoothness
- Goal state visits, forward transitions, and more

These capture how different heuristics behave during search.

---

## Results

- Final model: StackingClassifier with logistic regression
- Accuracy: **88.00%** on validation set
- SHAP and permutation importance used for explainability

---

## Output

- `best_model.pkl` includes:
  - Trained model
  - LabelEncoder
  - Important feature indices (SHAP-based)

