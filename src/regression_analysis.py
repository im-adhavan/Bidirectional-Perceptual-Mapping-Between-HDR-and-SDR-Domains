import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


def run_multivariate_regression(
    df,
    feature_columns,
    target_column,
    polynomial_degree=2,
    use_ridge=True,
    ridge_alpha=1.0,
    k_folds=5,
):

    X = df[feature_columns].values
    y = df[target_column].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    if use_ridge:
        model = Ridge(alpha=ridge_alpha)
    else:
        model = Ridge(alpha=0.0)

    model.fit(X_poly, y)

    r2 = model.score(X_poly, y)
    cv_r2 = cross_val_score(model, X_poly, y, cv=k_folds).mean()

    return {
        "r2": r2,
        "cv_r2": cv_r2,
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "feature_names": poly.get_feature_names_out(feature_columns)
    }