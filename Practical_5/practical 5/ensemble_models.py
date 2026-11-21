"""
Practical 5: Ensemble Learning - Stacking, AdaBoost and Voting
Dataset : Breast Cancer Wisconsin (Diagnostic) - 569 samples, 30 features, binary target
Base models      : SVM, Decision Tree, KNN, Gaussian Naive Bayes
Ensemble models  : Voting (hard / majority), Voting (soft), Stacking (Logistic Regression
                   meta-classifier), AdaBoost (decision stumps), AdaBoost (depth-3 decision trees)
Metrics          : Accuracy, Precision, Recall, F1-Score, ROC-AUC, Log Loss, 5-fold CV accuracy
Run              : python ensemble_models.py
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # works without a display; remove if running interactively
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (VotingClassifier, StackingClassifier, AdaBoostClassifier)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss)

# Newer scikit-learn versions warn that SVC(probability=True) will be deprecated; the
# warning is harmless here, so it is hidden to keep the console output readable.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*probability.*", category=FutureWarning)

SEED = 42

# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target          # target: 0 = malignant, 1 = benign
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, "
      f"Train: {len(X_train)}, Test: {len(X_test)}\n")

# ---------------------------------------------------------------------------
# 2. Base classifiers (scaling is done inside a pipeline so that it is fitted
#    only on the training data of each fit / cross-validation fold)
# ---------------------------------------------------------------------------
def base_models():
    return [
        ("svm", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True,
                                                    random_state=SEED))),
        ("dt",  DecisionTreeClassifier(max_depth=5, random_state=SEED)),
        ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
        ("nb",  make_pipeline(StandardScaler(), GaussianNB())),
    ]

models = {
    "SVM": base_models()[0][1],
    "Decision Tree": base_models()[1][1],
    "KNN": base_models()[2][1],
    "Naive Bayes": base_models()[3][1],
}

# ---------------------------------------------------------------------------
# 3. Ensemble models
# ---------------------------------------------------------------------------
# (a) Voting - hard voting = majority vote of the predicted class labels
models["Voting (Hard)"] = VotingClassifier(estimators=base_models(), voting="hard")
# (a2) Voting - soft voting = average of predicted probabilities (extra, for comparison)
models["Voting (Soft)"] = VotingClassifier(estimators=base_models(), voting="soft")

# (b) Stacking - a Logistic Regression meta-classifier learns how to combine the base
#     models' predicted probabilities (generated with 5-fold cross-validation)
models["Stacking"] = StackingClassifier(
    estimators=base_models(),
    final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
    cv=5)

# (c) AdaBoost - sequentially trains weak learners, re-weighting misclassified samples
models["AdaBoost (Stumps)"] = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
    n_estimators=100, learning_rate=0.5, random_state=SEED)
models["AdaBoost (Depth-3 Tree)"] = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3, random_state=SEED),
    n_estimators=100, learning_rate=0.5, random_state=SEED)

# ---------------------------------------------------------------------------
# 4. Training and evaluation
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    row = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        # error counts on the test set (target 0 = malignant, 1 = benign)
        "Malignant Missed": int(((y_test == 0) & (y_pred == 1)).sum()),   # malignant predicted benign
        "Benign Flagged": int(((y_test == 1) & (y_pred == 0)).sum()),     # benign predicted malignant
    }
    if hasattr(model, "predict_proba") and not (
            isinstance(model, VotingClassifier) and model.voting == "hard"):
        proba = model.predict_proba(X_test)
        row["ROC-AUC"] = roc_auc_score(y_test, proba[:, 1])
        row["Log Loss"] = log_loss(y_test, proba)
    else:
        row["ROC-AUC"], row["Log Loss"] = np.nan, np.nan   # hard voting gives no probabilities
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    row["CV Acc (mean)"] = scores.mean()
    row["CV Acc (std)"] = scores.std()
    results[name] = row
    print(f"--- {name} ---")
    for k, v in row.items():
        print(f"{k:<16}: {v:.4f}" if isinstance(v, float) else f"{k:<16}: {v}")
    print()

# ---------------------------------------------------------------------------
# 5. Comparative analysis
# ---------------------------------------------------------------------------
df = pd.DataFrame(results).T
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print("=" * 120)
print("                                   ENSEMBLE vs BASE MODEL COMPARISON")
print("=" * 120)
print(df.round(4))
print("=" * 120)
df.round(4).to_csv("results.csv")

colors = ["#9ecae1"] * 4 + ["#fdae6b"] * 2 + ["#a1d99b"] + ["#fc9272"] * 2

# Figure 1: test-set metrics
df[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind="bar", figsize=(12, 6), width=0.8)
plt.title("Test-set Performance: Base vs Ensemble Models")
plt.ylabel("Score")
plt.ylim(0.85, 1.0)
plt.xticks(rotation=20)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=150)
plt.close()

# Figure 2: 5-fold cross-validated accuracy with standard deviation
plt.figure(figsize=(10, 5))
plt.bar(df.index, df["CV Acc (mean)"], yerr=df["CV Acc (std)"], capsize=4, color=colors)
plt.title("5-Fold Cross-Validated Accuracy (mean ± std)")
plt.ylabel("Accuracy")
plt.ylim(0.88, 1.0)
plt.xticks(rotation=20)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("cv_accuracy.png", dpi=150)
plt.close()

# Figure 3: AdaBoost test accuracy as the number of boosting rounds grows
ada = models["AdaBoost (Stumps)"]
staged = [accuracy_score(y_test, p) for p in ada.staged_predict(X_test)]
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(staged) + 1), staged, color="green", linewidth=2)
plt.title("AdaBoost (Decision Stumps): Test Accuracy vs Boosting Rounds")
plt.xlabel("Number of boosting rounds")
plt.ylabel("Test accuracy")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("adaboost_staged.png", dpi=150)
plt.close()
