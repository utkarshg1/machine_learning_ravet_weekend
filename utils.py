from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator

import pandas as pd


def preprocess_data(X: pd.DataFrame) -> tuple[pd.DataFrame, ColumnTransformer]:
    # Extract categorical and continuous features
    cat_cols = list(X.columns[X.dtypes == "object"])
    con_cols = list(X.columns[X.dtypes != "object"])

    # Create a numerical pipeline on continuous data
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    # Create a categorical pipeline for categorical data
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
    )

    # Combine both pipelines with column transformer
    pre = ColumnTransformer(
        [("num", num_pipe, con_cols), ("cat", cat_pipe, cat_cols)]
    ).set_output(transform="pandas")

    # Fit transform data on X
    X_pre = pre.fit_transform(X)
    return X_pre, pre


def evaluate_single_model(
    model: BaseEstimator,
    xtrain: pd.DataFrame,
    ytrain: pd.Series,
    xtest: pd.DataFrame,
    ytest: pd.Series,
) -> dict:
    # Cross validation on xtrain
    scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring="f1_macro")
    f1_cv = scores.mean()

    # Fit the model on train data
    model.fit(xtrain, ytrain)

    # Predict the results for train and test
    ypred_train = model.predict(xtrain)
    ypred_test = model.predict(xtest)

    # Calculate f1 score on train and test
    f1_train = f1_score(ytrain, ypred_train, average="macro")
    f1_test = f1_score(ytest, ypred_test, average="macro")

    # Extract model name
    name = type(model).__name__

    # Create a dictionary showing above results
    res = {
        "model_name": name,
        "model": model,
        "f1_train": round(f1_train, 4),
        "f1_test": round(f1_test, 4),
        "f1_cv": f1_cv.round(4),
    }
    return res


def algorithm_evaluation(
    models: list[BaseEstimator],
    xtrain: pd.DataFrame,
    ytrain: pd.Series,
    xtest: pd.DataFrame,
    ytest: pd.Series,
) -> tuple[BaseEstimator, pd.DataFrame]:
    # Create blank results list
    model_res = []

    # Apply for loop on the models
    for model in models:
        r = evaluate_single_model(model, xtrain, ytrain, xtest, ytest)
        print(r)
        model_res.append(r)
        print("\n=================================================\n")

    # Convert model_res to dataframe
    res_df = pd.DataFrame(model_res)

    # Choose the best model
    sorted_df = res_df.sort_values(by="f1_cv", ascending=False).reset_index(drop=True)

    # Show the best model
    print("Best model selected : ")
    print(sorted_df.head(1))

    # Select best model
    best_model = sorted_df.loc[0, "model"]

    return best_model, sorted_df
