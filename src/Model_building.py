# """
# train_pipeline.py
#
# Train, evaluate, and save a spam classification pipeline (TF-IDF + best model)
# as a single joblib artifact for FastAPI deployment.
# Generates confusion matrices, metric comparison bar charts, ROC & PR curves.
#
# Steps:
# 1. Load preprocessed dataset
# 2. Encode labels (ham=0, spam=1)
# 3. Check class imbalance
# 4. Train/test split
# 5. Baseline + tuned models comparison
# 6. Save results (CSV + plots)
# 7. Save best tuned model as a Pipeline (TF-IDF + model)
#
# Author: Your Name
# """
#
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
#
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     classification_report, confusion_matrix,
#     roc_curve, auc, precision_recall_curve
# )
# from prettytable import PrettyTable
#
#
# # ---------------------------------------------------------
# # Utility Functions
# # ---------------------------------------------------------
# def evaluate_model(y_true, y_pred, model_name):
#     """Return metrics dict for a given model prediction."""
#     return {
#         "Model": model_name,
#         "Accuracy": accuracy_score(y_true, y_pred),
#         "Precision": precision_score(y_true, y_pred),
#         "Recall": recall_score(y_true, y_pred),
#         "F1 Score": f1_score(y_true, y_pred),
#     }
#
#
# def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
#     """Plot confusion matrix heatmap for a model."""
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#                 xticklabels=["Ham", "Spam"],
#                 yticklabels=["Ham", "Spam"])
#     plt.title(f"{model_name} Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight", dpi=150)
#         print(f"📊 Saved: {save_path}")
#     plt.close()
#
#
# def plot_metrics_bar(results, save_path=None):
#     """Plot bar chart comparison of models (Accuracy, Precision, Recall, F1)."""
#     df = pd.DataFrame(results)
#     metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
#
#     for metric in metrics:
#         plt.figure(figsize=(10, 5))
#         sns.barplot(x="Model", y=metric, data=df,
#                     dodge=False, palette="rocket")
#         plt.xticks(rotation=45, ha="right")
#         plt.title(f"Model Comparison - {metric}")
#         plt.ylabel(metric)
#         if save_path:
#             fname = os.path.join(save_path, f"comparison_{metric.lower().replace(' ', '_')}.png")
#             plt.savefig(fname, bbox_inches="tight", dpi=150)
#             print(f"📊 Saved: {fname}")
#         plt.close()
#
#
# def plot_roc_curve(y_true, y_prob, model_name, save_path=None):
#     """Plot ROC curve with AUC score."""
#     fpr, tpr, _ = roc_curve(y_true, y_prob)
#     roc_auc = auc(fpr, tpr)
#
#     plt.figure(figsize=(6, 5))
#     plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
#     plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title(f"ROC Curve - {model_name}")
#     plt.legend(loc="lower right")
#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight", dpi=150)
#         print(f"📊 Saved: {save_path}")
#     plt.close()
#
#
# def plot_pr_curve(y_true, y_prob, model_name, save_path=None):
#     """Plot Precision-Recall curve with AUC score."""
#     precision, recall, _ = precision_recall_curve(y_true, y_prob)
#     pr_auc = auc(recall, precision)
#
#     plt.figure(figsize=(6, 5))
#     plt.plot(recall, precision, color="purple", lw=2, label=f"AUC = {pr_auc:.3f}")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title(f"Precision-Recall Curve - {model_name}")
#     plt.legend(loc="lower left")
#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight", dpi=150)
#         print(f"📊 Saved: {save_path}")
#     plt.close()
#
#
# # ---------------------------------------------------------
# # Main Script
# # ---------------------------------------------------------
# def main():
#     # Directories
#     os.makedirs("models", exist_ok=True)
#     os.makedirs("outputs/figures", exist_ok=True)
#
#     # 1. Load dataset
#     df = pd.read_csv(r"C:\Users\chinm\Downloads\outputs\processed_spam.csv")
#
#     # 2. Encode labels
#     le = LabelEncoder()
#     y = le.fit_transform(df["Sender"])  # ham=0, spam=1
#
#     # 3. Check class imbalance
#     class_counts = pd.Series(y).value_counts(normalize=True)
#     print("\nClass distribution:")
#     print(class_counts)
#
#     if class_counts.min() < 0.4:  # heuristic: <40% minority → imbalanced
#         print("⚠️ Dataset appears imbalanced. Consider techniques like:")
#         print("- Class weights (e.g. in LogisticRegression, SVM, RF)")
#         print("- Oversampling (SMOTE) or undersampling")
#         print("- Using Precision-Recall AUC instead of accuracy for model selection")
#
#     # 4. Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         df["Messages"], y, test_size=0.2, random_state=42, stratify=y
#     )
#
#     # 5. Define models and hyperparams
#     model_configs = {
#         "Logistic Regression": {
#             "model": LogisticRegression(),
#             "params": {"clf__C": [0.01, 0.1, 1, 10], "clf__solver": ["liblinear"]}
#         },
#         "SVM": {
#             "model": SVC(probability=True),  # enable predict_proba
#             "params": {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]}
#         },
#         "Random Forest": {
#             "model": RandomForestClassifier(),
#             "params": {"clf__n_estimators": [50, 100, 200],
#                        "clf__max_depth": [None, 10, 20],
#                        "clf__min_samples_split": [2, 5]}
#         },
#         "KNN": {
#             "model": KNeighborsClassifier(),
#             "params": {"clf__n_neighbors": [3, 5, 7],
#                        "clf__weights": ["uniform", "distance"]}
#         },
#         "XGBoost": {
#             "model": XGBClassifier(eval_metric="logloss", use_label_encoder=False),
#             "params": {"clf__n_estimators": [50, 100, 200],
#                        "clf__learning_rate": [0.01, 0.1, 0.2]}
#         }
#     }
#
#     results = []
#     best_model = None
#     best_model_name = None
#     best_acc = 0
#
#     for model_name, cfg in model_configs.items():
#         print(f"\n==== {model_name} ====")
#
#         # Build pipeline: TF-IDF + model
#         pipeline = Pipeline([
#             ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
#             ("clf", cfg["model"])
#         ])
#
#         # Baseline training
#         pipeline.fit(X_train, y_train)
#         y_pred_base = pipeline.predict(X_test)
#         base_metrics = evaluate_model(y_test, y_pred_base, f"{model_name} (Baseline)")
#         results.append(base_metrics)
#
#         plot_confusion_matrix(
#             y_test, y_pred_base,
#             f"{model_name} (Baseline)",
#             save_path=f"outputs/figures/{model_name.replace(' ', '_')}_baseline_cm.png"
#         )
#
#         # Hyperparameter tuning
#         grid = GridSearchCV(
#             pipeline, cfg["params"], cv=3,
#             scoring="accuracy", n_jobs=-1
#         )
#         grid.fit(X_train, y_train)
#
#         tuned_model = grid.best_estimator_
#         y_pred_tuned = tuned_model.predict(X_test)
#         tuned_metrics = evaluate_model(y_test, y_pred_tuned, f"{model_name} (Tuned)")
#         tuned_metrics["Best Hyperparameters"] = grid.best_params_
#         results.append(tuned_metrics)
#
#         plot_confusion_matrix(
#             y_test, y_pred_tuned,
#             f"{model_name} (Tuned)",
#             save_path=f"outputs/figures/{model_name.replace(' ', '_')}_tuned_cm.png"
#         )
#
#         # ROC & PR curves (if probas available)
#         if hasattr(tuned_model, "predict_proba"):
#             y_prob = tuned_model.predict_proba(X_test)[:, 1]
#         elif hasattr(tuned_model, "decision_function"):
#             y_prob = tuned_model.decision_function(X_test)
#         else:
#             y_prob = None
#
#         if y_prob is not None:
#             roc_path = f"outputs/figures/{model_name.replace(' ', '_')}_tuned_roc.png"
#             pr_path = f"outputs/figures/{model_name.replace(' ', '_')}_tuned_pr.png"
#             plot_roc_curve(y_test, y_prob, f"{model_name} (Tuned)", save_path=roc_path)
#             plot_pr_curve(y_test, y_prob, f"{model_name} (Tuned)", save_path=pr_path)
#
#         # Track best
#         if tuned_metrics["Accuracy"] > best_acc:
#             best_acc = tuned_metrics["Accuracy"]
#             best_model = tuned_model
#             best_model_name = model_name
#
#     # Results summary
#     pd.DataFrame(results).to_csv("outputs/model_results.csv", index=False)
#     plot_metrics_bar(results, save_path="outputs/figures")
#
#     # Save best pipeline + label encoder
#     if best_model:
#         joblib.dump(best_model, "models/spam_classifier.pkl")
#         joblib.dump(le, "models/label_encoder.pkl")
#         print(f"\n✅ Saved best pipeline: {best_model_name} "
#               f"with accuracy {best_acc:.4f} "
#               f"-> models/spam_classifier.pkl")
#
#
# if __name__ == "__main__":
#     main()


"""
train_pipeline.py

Train, evaluate, and save SMS spam classification pipelines
under both imbalanced and SMOTE-balanced setups.

Outputs:
- Confusion matrices (baseline + tuned)
- ROC & PR curves (tuned)
- Metric comparison bar charts
- CSV results for both setups
- Best pipelines saved for deployment (FastAPI ready)

Author: Your Name
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from prettytable import PrettyTable


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------
def evaluate_model(y_true, y_pred, model_name):
    """Return metrics dict for a given model prediction."""
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot confusion matrix heatmap for a model."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"📊 Saved: {save_path}")
    plt.close()


def plot_metrics_bar(results, save_path=None, tag=""):
    """Plot bar chart comparison of models (Accuracy, Precision, Recall, F1)."""
    df = pd.DataFrame(results)
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

    for metric in metrics:
        plt.figure(figsize=(10, 5))
        sns.barplot(x="Model", y=metric, data=df,
                    dodge=False, palette="rocket")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Model Comparison - {metric} ({tag})")
        plt.ylabel(metric)
        if save_path:
            fname = os.path.join(save_path, f"comparison_{metric.lower().replace(' ', '_')}_{tag}.png")
            plt.savefig(fname, bbox_inches="tight", dpi=150)
            print(f"📊 Saved: {fname}")
        plt.close()


def plot_roc_curve(y_true, y_prob, model_name, save_path=None):
    """Plot ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"📊 Saved: {save_path}")
    plt.close()


def plot_pr_curve(y_true, y_prob, model_name, save_path=None):
    """Plot Precision-Recall curve with AUC score."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="purple", lw=2, label=f"AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="lower left")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"📊 Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------
# Training Function (Reusable for imbalanced & SMOTE)
# ---------------------------------------------------------
def run_experiment(X_train, X_test, y_train, y_test, le, tag="imbalanced"):
    """
    Run baseline + tuned models on given dataset.
    Saves metrics, plots, and best pipeline.
    """
    print(f"\n🔎 Running experiment: {tag}")

    # Models + params
    model_configs = {
        "Logistic Regression": {
            "model": LogisticRegression(),
            "params": {"clf__C": [0.01, 0.1, 1, 10], "clf__solver": ["liblinear"]}
        },
        "SVM": {
            "model": SVC(probability=True),
            "params": {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]}
        },
        "Random Forest": {
            "model": RandomForestClassifier(),
            "params": {"clf__n_estimators": [50, 100, 200],
                       "clf__max_depth": [None, 10, 20],
                       "clf__min_samples_split": [2, 5]}
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {"clf__n_neighbors": [3, 5, 7],
                       "clf__weights": ["uniform", "distance"]}
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric="logloss", use_label_encoder=False),
            "params": {"clf__n_estimators": [50, 100, 200],
                       "clf__learning_rate": [0.01, 0.1, 0.2]}
        }
    }

    results = []
    best_model = None
    best_name = None
    best_acc = 0

    for model_name, cfg in model_configs.items():
        print(f"\n==== {model_name} ({tag}) ====")

        # Choose pipeline type
        if tag == "smote":
            pipeline = ImbPipeline([
                ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
                ("smote", SMOTE(random_state=42)),
                ("clf", cfg["model"])
            ])
        else:
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
                ("clf", cfg["model"])
            ])

        # Baseline
        pipeline.fit(X_train, y_train)
        y_pred_base = pipeline.predict(X_test)
        base_metrics = evaluate_model(y_test, y_pred_base, f"{model_name} (Baseline, {tag})")
        results.append(base_metrics)

        plot_confusion_matrix(
            y_test, y_pred_base,
            f"{model_name} (Baseline, {tag})",
            save_path=f"outputs/figures/{model_name.replace(' ', '_')}_baseline_cm_{tag}.png"
        )

        # Hyperparameter tuning
        grid = GridSearchCV(pipeline, cfg["params"], cv=3, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)

        tuned_model = grid.best_estimator_
        y_pred_tuned = tuned_model.predict(X_test)
        tuned_metrics = evaluate_model(y_test, y_pred_tuned, f"{model_name} (Tuned, {tag})")
        tuned_metrics["Best Hyperparameters"] = grid.best_params_
        results.append(tuned_metrics)

        plot_confusion_matrix(
            y_test, y_pred_tuned,
            f"{model_name} (Tuned, {tag})",
            save_path=f"outputs/figures/{model_name.replace(' ', '_')}_tuned_cm_{tag}.png"
        )

        # ROC & PR curves
        if hasattr(tuned_model, "predict_proba"):
            y_prob = tuned_model.predict_proba(X_test)[:, 1]
        elif hasattr(tuned_model, "decision_function"):
            y_prob = tuned_model.decision_function(X_test)
        else:
            y_prob = None

        if y_prob is not None:
            roc_path = f"outputs/figures/{model_name.replace(' ', '_')}_tuned_roc_{tag}.png"
            pr_path = f"outputs/figures/{model_name.replace(' ', '_')}_tuned_pr_{tag}.png"
            plot_roc_curve(y_test, y_prob, f"{model_name} (Tuned, {tag})", save_path=roc_path)
            plot_pr_curve(y_test, y_prob, f"{model_name} (Tuned, {tag})", save_path=pr_path)

        # Track best
        if tuned_metrics["F1 Score"] > best_acc:  # use F1 for imbalance
            best_acc = tuned_metrics["F1 Score"]
            best_model = tuned_model
            best_name = model_name

    # Save results
    results_path = f"outputs/model_results_{tag}.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"\n📄 Saved results: {results_path}")
    plot_metrics_bar(results, save_path="outputs/figures", tag=tag)

    # Save best pipeline + label encoder
    if best_model:
        joblib.dump(best_model, f"models/spam_classifier_{tag}.pkl")
        joblib.dump(le, f"models/label_encoder_{tag}.pkl")
        print(f"\n✅ Saved best {tag} pipeline: {best_name} "
              f"with F1 {best_acc:.4f} "
              f"-> models/spam_classifier_{tag}.pkl")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    os.makedirs("models/smote", exist_ok=True)
    os.makedirs("outputs/figure", exist_ok=True)

    # Load data
    df = pd.read_csv(r"C:\Users\chinm\Downloads\outputs\processed_spam.csv")
    le = LabelEncoder()
    y = le.fit_transform(df["Sender"])  # ham=0, spam=1

    # Class distribution
    print("\nClass distribution (original):")
    print(pd.Series(y).value_counts(normalize=True))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["Messages"], y, test_size=0.2, random_state=42, stratify=y
    )

    # Run both experiments
    run_experiment(X_train, X_test, y_train, y_test, le, tag="imbalanced")
    run_experiment(X_train, X_test, y_train, y_test, le, tag="smote")


if __name__ == "__main__":
    main()
