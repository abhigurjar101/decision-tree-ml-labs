from pathlib import Path
"""
=============================================================================
DECISION TREE REGRESSION MASTER LAB
End-to-End Practical Implementation with Scikit-Learn
Case Study: House Price Prediction (Continuous Target)
=============================================================================
This script demonstrates:
1. Continuous target data generation & EDA
2. DecisionTreeRegressor instantiation & fitting
3. Variance Reduction / MSE splitting criteria mechanics
4. Prediction generation: piecewise-constant / step-function nature
5. Regression metrics: MAE, MSE, RMSE, R² score
6. Tree visualization (Text & Graphic)
7. Hyperparameter tuning (max_depth, min_samples_leaf)
8. Residual analysis and variance comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_regression_pipeline():
    print("=" * 70)
    print("STEP 1: LOAD & EXPLORE REAL ESTATE DATASET")
    print("=" * 70)
    df = pd.read_csv(Path(__file__).resolve().parent / "data" / "house_prices_data.csv")
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nSummary Statistics of Target (Price_k$):")
    print(df['Price_k$'].describe().round(2))

    features = ['Area_sqft', 'Bedrooms', 'Distance_KM', 'Property_Age']
    X = df[features]
    y = df['Price_k$']

    print("\n" + "=" * 70)
    print("STEP 2: TRAIN / TEST SPLIT")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")

    print("\n" + "=" * 70)
    print("STEP 3: TRAIN DECISION TREE REGRESSOR (MSE / SQUARED_ERROR)")
    print("=" * 70)
    reg = DecisionTreeRegressor(criterion='squared_error', max_depth=3, random_state=42)
    reg.fit(X_train, y_train)

    print("\n" + "=" * 70)
    print("STEP 4: MODEL EVALUATION & REGRESSION METRICS")
    print("=" * 70)
    y_pred = reg.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE):     ${mae*1000:,.2f}  (in $k: {mae:.2f})")
    print(f"Mean Squared Error (MSE):      {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse*1000:,.2f}  (in $k: {rmse:.2f})")
    print(f"R-squared Score (R²):          {r2:.4f}  ({r2*100:.1f}% variance explained)")

    print("\n" + "=" * 70)
    print("STEP 5: REGRESSION TREE RULES IN PLAIN TEXT")
    print("=" * 70)
    rules = export_text(reg, feature_names=features)
    print(rules)

    print("\n" + "=" * 70)
    print("STEP 6: FEATURE IMPORTANCES (VARIANCE REDUCTION)")
    print("=" * 70)
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': reg.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importances.to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 7: 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(reg, X_train, y_train, cv=kf, scoring='r2')
    print(f"5-Fold CV R²: {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
    print("Fold Scores:", np.round(cv_r2, 4))

    print("\n" + "=" * 70)
    print("STEP 8: HYPERPARAMETER TUNING VIA GRIDSEARCHCV")
    print("=" * 70)
    param_grid = {
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': [2, 3, 4, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10, 20]
    }
    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        n_jobs=1
    )
    grid.fit(X_train, y_train)

    print("Best Hyperparameters:")
    for k, v in grid.best_params_.items():
        print(f"  {k}: {v}")
    best_test_rmse = np.sqrt(mean_squared_error(y_test, grid.best_estimator_.predict(X_test)))
    print(f"Tuned Test RMSE: ${best_test_rmse*1000:,.2f}  (in $k: {best_test_rmse:.2f})")
    print(f"Tuned Test R²:   {r2_score(y_test, grid.best_estimator_.predict(X_test)):.4f}")

    print("\n" + "=" * 70)
    print("STEP 9: KEY OBSERVATION: STEP-FUNCTION NATURE")
    print("=" * 70)
    print("Unique predictions on test set:", len(np.unique(y_pred)))
    print("Max depth was 3, yielding at most 2^3 = 8 leaf nodes.")
    print("Each leaf predicts the constant arithmetic mean of its training samples.")
    print("Decision Trees approximate continuous functions with discrete piecewise steps!")

if __name__ == "__main__":
    run_regression_pipeline()
