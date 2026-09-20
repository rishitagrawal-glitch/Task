import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    mean_squared_error, r2_score,
)

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]

def task1_eda(data: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("TASK 1: DATA PREPROCESSING & EDA")
    print("=" * 60)

    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    print("\nShape:", data.shape)
    print("\nFirst 5 rows:\n", data.head())
    print("\nInfo:")
    data.info()
    print("\nMissing values per column:\n", data.isnull().sum())
    n_dupes = data.duplicated().sum()
    print(f"\nDuplicate rows: {n_dupes}")
    data = data.drop_duplicates().reset_index(drop=True)

    
    print("\nOutliers per feature (IQR method):")
    for col in features:
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((data[col] < low) | (data[col] > high)).sum()
        print(f"  {col}: {n_out}")
        # Cap (winsorize) instead of dropping, to keep the dataset small-but-intact
        data[col] = data[col].clip(lower=low, upper=high)


    print("\nSummary statistics:\n", data[features].describe())
    print("\nClass distribution:\n", data["species"].value_counts())

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.ravel(), features):
        sns.histplot(data=data, x=col, hue="species", kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig("eda_histograms.png", dpi=120)

    plt.figure(figsize=(6, 5))
    sns.heatmap(data[features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("eda_correlation.png", dpi=120)

    sns.pairplot(data, vars=features, hue="species")
    plt.savefig("eda_pairplot.png", dpi=120)

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data.melt(id_vars="species", value_vars=features),
                x="variable", y="value", hue="species")
    plt.title("Feature Boxplots by Species")
    plt.tight_layout()
    plt.savefig("eda_boxplots.png", dpi=120)

    print("\nKey findings:")
    print("- No missing values; a few duplicate rows were removed.")
    print("- Petal length and petal width are strongly correlated (~0.96).")
    print("- Setosa is clearly separable using petal measurements.")
    print("- Versicolor and virginica overlap slightly.")
    return data

def task2_classification(data: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("TASK 2: CLASSIFICATION MODEL")
    print("=" * 60)

    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = data[features]
    y = data["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro")
        rec = recall_score(y_test, preds, average="macro")

        print(f"\n--- {name} ---")
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f} (macro)")
        print(f"Recall   : {rec:.3f} (macro)")
        print("\nClassification report:\n",
              classification_report(y_test, preds))

        cm = confusion_matrix(y_test, preds, labels=model.classes_)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"confusion_{name.replace(' ', '_').lower()}.png", dpi=120)

    print("\nExplanation of results:")
    print("Both models perform very well on Iris. Setosa is classified almost")
    print("perfectly; any errors occur between versicolor and virginica, whose")
    print("petal measurements overlap. Precision and recall are both reported")
    print("as macro averages across the three classes.")


def task3_regression(data: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("TASK 3: REGRESSION MODEL")
    print("=" * 60)

    X = data[["petal_length"]]
    y = data["petal_width"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lin = LinearRegression().fit(X_train, y_train)
    lin_pred = lin.predict(X_test)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_p = poly.fit_transform(X_train)
    X_test_p = poly.transform(X_test)
    poly_model = LinearRegression().fit(X_train_p, y_train)
    poly_pred = poly_model.predict(X_test_p)

    for name, pred in [("Linear", lin_pred), ("Polynomial (deg 2)", poly_pred)]:
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        print(f"{name:20s} RMSE = {rmse:.3f} | R2 = {r2_score(y_test, pred):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X, y, alpha=0.5, label="Data")
    grid = np.linspace(X.min().iloc[0], X.max().iloc[0], 200).reshape(-1, 1)
    grid_df = pd.DataFrame(grid, columns=["petal_length"])
    axes[0].plot(grid, lin.predict(grid_df), color="red", label="Linear fit")
    axes[0].plot(grid, poly_model.predict(poly.transform(grid_df)),
                 color="green", label="Polynomial fit")
    axes[0].set_xlabel("Petal length (cm)")
    axes[0].set_ylabel("Petal width (cm)")
    axes[0].set_title("Regression Fits")
    axes[0].legend()

    axes[1].scatter(y_test, poly_pred, alpha=0.7)
    lims = [y_test.min(), y_test.max()]
    axes[1].plot(lims, lims, "r--", label="Perfect prediction")
    axes[1].set_xlabel("Actual petal width")
    axes[1].set_ylabel("Predicted petal width")
    axes[1].set_title("Predicted vs Actual (Polynomial)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("regression_results.png", dpi=120)

    print("\nInsights:")
    print("Petal length is a strong predictor of petal width (R2 ~ 0.9+).")
    print("The polynomial model gives only a small improvement over the linear")
    print("one, suggesting the relationship is mostly linear.")


if __name__ == "__main__":
    clean_df = task1_eda(df)
    task2_classification(clean_df)
    task3_regression(clean_df)
    plt.show()