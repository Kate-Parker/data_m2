"""Train and evaluate a first branded-food category classifier."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder


TARGET = "branded_food_category"
TEXT_COLUMNS = ["product_description", "ingredients"]
CATEGORICAL_COLUMNS = ["brand_owner", "brand_name", "serving_size_unit"]
NUMERIC_COLUMNS = [
    "serving_size",
    "energy_kcal",
    "protein_g",
    "total_fat_g",
    "carbohydrate_g",
    "total_sugars_g",
    "sodium_mg",
    "fiber_g",
    "saturated_fat_g",
]
REQUIRED_COLUMNS = [
    *TEXT_COLUMNS,
    *CATEGORICAL_COLUMNS,
    *NUMERIC_COLUMNS,
    TARGET,
]


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path, usecols=REQUIRED_COLUMNS)
    if data[TARGET].isna().any():
        raise ValueError(f"{path} contains rows without a target category")

    data["texte_produit"] = (
        data["product_description"].fillna("").astype(str)
        + " "
        + data["ingredients"].fillna("").astype(str)
    )
    labels = data.pop(TARGET).astype(str)
    return data, labels


def build_model() -> Pipeline:
    preparation = ColumnTransformer(
        transformers=[
            (
                "texte",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
                "texte_produit",
            ),
            (
                "categoriel",
                Pipeline(
                    steps=[
                        ("imputation", SimpleImputer(strategy="most_frequent")),
                        (
                            "encodage",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=2,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
            (
                "numerique",
                Pipeline(
                    steps=[
                        (
                            "imputation",
                            SimpleImputer(
                                strategy="median", keep_empty_features=True
                            ),
                        ),
                        ("mise_a_echelle", MaxAbsScaler()),
                    ]
                ),
                NUMERIC_COLUMNS,
            ),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline(
        steps=[
            ("preparation", preparation),
            ("classification", ComplementNB(alpha=0.1)),
        ]
    )


def evaluate_model(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    train_x, train_y = load_split(train_path)
    test_x, test_y = load_split(test_path)

    train_classes = set(train_y.unique())
    test_classes = set(test_y.unique())
    if not test_classes.issubset(train_classes):
        raise ValueError("The test split contains categories absent from training")

    majority_class = train_y.value_counts().idxmax()
    majority_prediction = np.full(len(test_y), majority_class, dtype=object)
    majority_metrics = {
        "accuracy": float(accuracy_score(test_y, majority_prediction)),
        "f1_macro": float(
            f1_score(test_y, majority_prediction, average="macro", zero_division=0)
        ),
    }

    model = build_model()
    fit_started = time.perf_counter()
    model.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - fit_started

    prediction_started = time.perf_counter()
    predictions = model.predict(test_x)
    prediction_seconds = time.perf_counter() - prediction_started

    metrics: dict[str, Any] = {
        "modele": "ComplementNB avec TF-IDF, variables catégorielles et numériques",
        "fichier_train": str(train_path),
        "fichier_test": str(test_path),
        "lignes_train": int(len(train_y)),
        "lignes_test": int(len(test_y)),
        "categories_train": int(len(train_classes)),
        "categories_test": int(len(test_classes)),
        "categories_test_absentes_train": 0,
        "baseline_classe_majoritaire": majority_metrics,
        "accuracy": float(accuracy_score(test_y, predictions)),
        "f1_macro": float(
            f1_score(test_y, predictions, average="macro", zero_division=0)
        ),
        "f1_pondere": float(
            f1_score(test_y, predictions, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
        "temps_entrainement_secondes": round(fit_seconds, 2),
        "temps_prediction_test_secondes": round(prediction_seconds, 2),
        "imputation_apprise_sur_train_uniquement": True,
        "vectoriseur_appris_sur_train_uniquement": True,
        "hyperparametres_principaux": {
            "tfidf_max_features": 50_000,
            "tfidf_ngram_range": [1, 2],
            "tfidf_min_df": 2,
            "complement_nb_alpha": 0.1,
        },
    }
    per_class_report = classification_report(
        test_y,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "modele_classification.joblib"
    metrics_path = output_dir / "classification_metrics.json"
    report_path = output_dir / "classification_par_categorie.json"
    dump(model, model_path)
    metrics["modele_sauvegarde"] = str(model_path)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(per_class_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=project_dir / "data" / "processed" / "train.csv",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=project_dir / "data" / "processed" / "test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "data" / "processed",
    )
    args = parser.parse_args()

    metrics = evaluate_model(args.train, args.test, args.output_dir)
    print("Évaluation sur le jeu de test (jamais utilisé pour l'entraînement) :")
    print(f"  Exactitude (accuracy) : {metrics['accuracy']:.4f}")
    print(f"  F1 macro              : {metrics['f1_macro']:.4f}")
    print(f"  F1 pondéré            : {metrics['f1_pondere']:.4f}")
    print(f"  Baseline majorité     : {metrics['baseline_classe_majoritaire']['accuracy']:.4f}")
    print(f"  Modèle enregistré     : {metrics['modele_sauvegarde']}")


if __name__ == "__main__":
    main()
