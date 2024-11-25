import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def training(X_train, y_train, model):
    train_dict = X_train[cat_columns + num_columns].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dict)

    model = model
    model.fit(X_train, y_train)

    return dv, model


def predict(df, dv, model):
    dicts = df[cat_columns + num_columns].to_dict(orient="records")

    X = dv.transform(dicts)
    y_pred = model.predict(X)

    return y_pred

def accu_class(y_train, y_pred_train, y_test, y_pred_test):
    print("accuracy score - Train : ", accuracy_score(y_train, y_pred_train))
    print("accuracy score - Test  : ", accuracy_score(y_test, y_pred_test))
    print(
        "----------------------------------------------------------------------------"
    )
    print(
        "Classification Report - Train : \n",
        classification_report(y_train, y_pred_train),
        "\n",
    )
    print(
        "----------------------------------------------------------------------------"
    )
    print(
        "Classification Report - Test: \n",
        classification_report(y_test, y_pred_test),
        "\n",
    )

if __name__ == "__main__":
    df = pd.read_csv("bodyPerformance.csv")

    print(df.head())

    # Split Data
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1, stratify=df["class"]
    )

    df_train, df_val = train_test_split(
        df_train, test_size=0.25, random_state=1, stratify=df_train["class"]
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # Split between X (features) and y (target)
    X_train = df_train.drop("class", axis=1)
    y_train = df_train["class"]

    X_val = df_val.drop("class", axis=1)
    y_val = df_val["class"]

    # Training
    print("Training...")
    # Get Numerical Columns and Categorical Columns (original)
    num_columns = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_columns = X_train.select_dtypes(include=["object"]).columns.tolist()

    dv, model = training(
        X_train,
        y_train,
        RandomForestClassifier(
            n_estimators=200,
            min_samples_split=5,
            min_samples_leaf=4,
            max_samples=None,
            max_features="log2",
            max_depth=20,
            class_weight=None,
            bootstrap=False,
        ),
    )

    print("Evaluating...")
    y_pred_train = predict(X_train, dv, model)
    y_pred_val = predict(X_val, dv, model)

    accu_class(y_train, y_pred_train, y_val, y_pred_val)

    print("Saving...")
    with open('body_performance_model.bin', 'wb') as f_out:
        pickle.dump((dv, model), f_out)

    print("Done")
    

