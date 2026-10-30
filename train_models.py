"""
train_models.py
---------------
2. MODEL BUILDING & TRAINING (classical models on TF-IDF features)

Task: token-level language identification  ->  KN | HI | EN   (romanized tokens)

Features : TF-IDF over character 1-4-grams (char_wb)   [same idea as your
           original pipeline, which used char n-gram counts]
Algorithms (3):
    1. Logistic Regression
    2. Linear SVM  (probability-calibrated so ROC curves can be drawn)
    3. Random Forest
(+ BERT is trained separately in bert_model.py and merged into the
   comparison automatically if its results file exists.)

Outputs go to ./results :
    metrics_<model>.json, report_<model>.txt,
    confusion_matrix_<model>.png, roc_curve_<model>.png,
    model_comparison.csv/.png, roc_all_models.png, cv_results.csv

Run:
    python data_builder.py            # once (or pass --data your.csv)
    python train_models.py
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

from evaluation import CLASSES, evaluate, plot_comparison, plot_roc_overlay

SEED = 42


def tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4),
                           lowercase=True, sublinear_tf=True)


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(max_iter=2000, C=5.0, random_state=SEED)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", tfidf()),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=SEED), cv=5)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", tfidf()),
            ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)),
        ]),
    }


def load_data(path):
    df = pd.read_csv(path)
    if "split" not in df.columns:  # user-supplied CSV: make a grouped 80/20 split
        rng = np.random.RandomState(SEED)
        if "group" not in df.columns:
            df["group"] = df["token"].str.lower()
        groups = df["group"].unique()
        test_groups = set(rng.choice(groups, int(0.2 * len(groups)), replace=False))
        df["split"] = np.where(df["group"].isin(test_groups), "test", "train")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/token_lid_dataset.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        from data_builder import main as build
        import sys; sys.argv = ["data_builder.py", "--out", args.data]
        build()

    os.makedirs(args.out, exist_ok=True)
    df = load_data(args.data)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    print(f"Train: {len(tr)}  Test: {len(te)}  (split by base-word group)")
    print(tr.label.value_counts().to_dict(), "\n")

    all_metrics, curves, cv_rows = [], {}, []
    gkf = GroupKFold(n_splits=5)

    for name, model in build_models().items():
        print(f"=== {name} ===")
        # 5-fold grouped cross-validation on the TRAIN set (stability check)
        cv = cross_validate(model, tr.token, tr.label, groups=tr.group, cv=gkf,
                            scoring=["accuracy", "f1_macro"], n_jobs=1)
        cv_rows.append({"model": name,
                        "cv_accuracy_mean": cv["test_accuracy"].mean(),
                        "cv_accuracy_std": cv["test_accuracy"].std(),
                        "cv_f1_macro_mean": cv["test_f1_macro"].mean(),
                        "cv_f1_macro_std": cv["test_f1_macro"].std()})

        model.fit(tr.token, tr.label)
        proba_raw = model.predict_proba(te.token)
        order = [list(model.classes_).index(c) for c in CLASSES]
        proba = proba_raw[:, order]
        pred = np.array(CLASSES)[proba.argmax(axis=1)]

        m, report, cm = evaluate(name, te.label.values, pred, proba, args.out)
        all_metrics.append(m)
        curves[name] = (label_binarize(te.label, classes=CLASSES), proba)
        print(report)
        print("Confusion matrix (rows=true, cols=pred, order KN/HI/EN):\n", cm, "\n")
        joblib.dump(model, os.path.join(args.out, f"model_{name.lower().replace(' ', '_')}.joblib"))

        # save predictions for error analysis
        out = te[["token", "label"]].copy()
        out["pred"] = pred
        out[[f"p_{c}" for c in CLASSES]] = proba
        out[out.label != out.pred].to_csv(
            os.path.join(args.out, f"errors_{name.lower().replace(' ', '_')}.csv"), index=False)

    # merge BERT results if bert_model.py has been run
    bert_json = os.path.join(args.out, "metrics_bert.json")
    bert_npz = os.path.join(args.out, "bert_test_probs.npz")
    if os.path.exists(bert_json) and os.path.exists(bert_npz):
        all_metrics.append(json.load(open(bert_json)))
        z = np.load(bert_npz, allow_pickle=True)
        curves["BERT"] = (label_binarize(z["y_true"], classes=CLASSES), z["proba"])
        print("(merged BERT results into the comparison)")

    pd.DataFrame(cv_rows).to_csv(os.path.join(args.out, "cv_results.csv"), index=False)
    comp = pd.DataFrame([{k: v for k, v in m.items() if k != "roc_auc_per_class"}
                         for m in all_metrics])
    comp.to_csv(os.path.join(args.out, "model_comparison.csv"), index=False)
    plot_comparison(all_metrics, os.path.join(args.out, "model_comparison.png"))
    plot_roc_overlay(curves, CLASSES, os.path.join(args.out, "roc_all_models.png"))

    print("\n" + "=" * 70 + "\nFINAL COMPARISON (held-out test set)\n" + "=" * 70)
    print(comp.round(4).to_string(index=False))
    print("\n5-fold grouped CV on train set:")
    print(pd.DataFrame(cv_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
