# Coffee Churn Prediction

This model predicts the probability of customer churn for the Happy Beans Coffee delivery service.

## Project Task

Build an interpretable binary classification model to predict customer churn for a coffee delivery service.

The project required:

- exploratory data analysis and data quality checks;
- handling missing values, invalid values, outliers and categorical features;
- preventing data leakage through pipeline-based preprocessing;
- building a `DummyClassifier` baseline and a `LogisticRegression` model;
- feature engineering and feature selection experiments;
- cross-validation and hyperparameter tuning;
- evaluation primarily with PR-AUC, alongside Precision, Recall, F1 and Log Loss;
- selection of an appropriate classification threshold;
- final evaluation on a held-out test set;
- saving the complete preprocessing + model pipeline for further use.

## Loading and Using the Model

The model is saved together with the complete preprocessing pipeline and metadata.

```python
import joblib

# Load the model
artifact = joblib.load('models/coffee_churn_model.joblib')

pipeline = artifact['pipeline']
threshold = artifact['metadata']['threshold']

# X_new contains raw data in the same format as the training set
y_proba = pipeline.predict_proba(X_new)[:, 1]

# Apply the saved classification threshold
y_pred = (y_proba >= threshold).astype(int)
```

The saved `threshold = 0.3` is used, so the final class is determined from the probabilities returned by `predict_proba()`.

The pipeline automatically performs feature engineering, missing-value handling, categorical-feature encoding, and scaling.

## Final Model

Parameters:

- `C = 0.01`
- `max_iter = 100`
- `threshold = 0.3`

Held-out test-set results:

- Precision: **0.636**
- Recall: **0.651**
- F1: **0.643**
- PR-AUC: **0.714**
- Log Loss: **0.113**

The model identifies approximately **65% of customers who actually churn**, while maintaining Precision of approximately **64%**.
