"""Build customer profiles and cluster Instacart customers with MiniBatchKMeans."""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler


RANDOM_STATE = 42
TARGET_EVAL_SET = "prior"
CHUNK_SIZE = 750_000
SILHOUETTE_SAMPLE_SIZE = 2_500


def _column_slug(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return re.sub(r"[^a-z0-9]+", "_", ascii_value).strip("_")


def _read_orders(orders_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    orders = pd.read_csv(
        orders_path,
        usecols=[
            "order_id",
            "user_id",
            "eval_set",
            "order_number",
            "order_dow",
            "order_hour_of_day",
            "days_since_prior_order",
        ],
        dtype={
            "order_id": "uint32",
            "user_id": "uint32",
            "eval_set": "category",
            "order_number": "uint8",
            "order_dow": "uint8",
            "order_hour_of_day": "uint8",
            "days_since_prior_order": "float32",
        },
    )
    if orders["order_id"].duplicated().any():
        raise ValueError("orders.csv contains duplicate order_id values")

    counts = {
        "orders_total": int(len(orders)),
        "users_total": int(orders["user_id"].nunique()),
        "orders_prior": int(orders["eval_set"].eq(TARGET_EVAL_SET).sum()),
        "orders_train_eval": int(orders["eval_set"].eq("train").sum()),
        "orders_test_eval": int(orders["eval_set"].eq("test").sum()),
    }
    prior = orders.loc[orders["eval_set"].eq(TARGET_EVAL_SET)].drop(
        columns="eval_set"
    )
    if prior.empty:
        raise ValueError("No eval_set='prior' orders found")
    return prior, counts


def _build_order_features(prior_orders: pd.DataFrame) -> pd.DataFrame:
    order_features = prior_orders.copy()
    dow_angle = 2 * np.pi * order_features["order_dow"].astype("float32") / 7
    hour_angle = (
        2 * np.pi * order_features["order_hour_of_day"].astype("float32") / 24
    )
    order_features["dow_sin"] = np.sin(dow_angle)
    order_features["dow_cos"] = np.cos(dow_angle)
    order_features["hour_sin"] = np.sin(hour_angle)
    order_features["hour_cos"] = np.cos(hour_angle)
    order_features["weekend_order"] = order_features["order_dow"].isin([0, 6]).astype(
        "uint8"
    )
    order_features["evening_order"] = order_features[
        "order_hour_of_day"
    ].between(17, 22).astype("uint8")

    features = order_features.groupby("user_id").agg(
        prior_order_count=("order_id", "nunique"),
        mean_days_between_orders=("days_since_prior_order", "mean"),
        median_days_between_orders=("days_since_prior_order", "median"),
        std_days_between_orders=("days_since_prior_order", "std"),
        mean_dow_sin=("dow_sin", "mean"),
        mean_dow_cos=("dow_cos", "mean"),
        mean_hour_sin=("hour_sin", "mean"),
        mean_hour_cos=("hour_cos", "mean"),
        weekend_order_share=("weekend_order", "mean"),
        evening_order_share=("evening_order", "mean"),
    )
    features["mean_order_hour"] = (
        np.arctan2(features["mean_hour_sin"], features["mean_hour_cos"])
        .mod(2 * np.pi)
        .div(2 * np.pi)
        .mul(24)
    )
    features["mean_order_dow"] = (
        np.arctan2(features["mean_dow_sin"], features["mean_dow_cos"])
        .mod(2 * np.pi)
        .div(2 * np.pi)
        .mul(7)
    )
    return features


def _aggregate_prior_items(
    input_dir: Path,
    prior_orders: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    products = pd.read_csv(
        input_dir / "products.csv",
        usecols=["product_id", "department_id"],
        dtype={"product_id": "uint32", "department_id": "uint8"},
    )
    departments = pd.read_csv(
        input_dir / "departments.csv",
        dtype={"department_id": "uint8", "department": str},
    )
    if products["product_id"].duplicated().any():
        raise ValueError("products.csv contains duplicate product_id values")

    order_to_user = pd.Series(
        prior_orders["user_id"].to_numpy(),
        index=prior_orders["order_id"].to_numpy(),
    )
    product_to_department = pd.Series(
        products["department_id"].to_numpy(),
        index=products["product_id"].to_numpy(),
    )

    user_item_blocks: list[pd.DataFrame] = []
    basket_blocks: list[pd.DataFrame] = []
    department_blocks: list[pd.DataFrame] = []
    item_rows = 0

    for chunk in pd.read_csv(
        input_dir / "order_products__prior.csv",
        usecols=["order_id", "product_id", "reordered"],
        dtype={"order_id": "uint32", "product_id": "uint32", "reordered": "uint8"},
        chunksize=CHUNK_SIZE,
    ):
        item_rows += len(chunk)
        chunk["user_id"] = chunk["order_id"].map(order_to_user)
        chunk["department_id"] = chunk["product_id"].map(product_to_department)
        if chunk["user_id"].isna().any():
            raise ValueError("Some prior order products have no matching prior order")
        if chunk["department_id"].isna().any():
            raise ValueError("Some prior products have no matching department")

        user_item_blocks.append(
            chunk.groupby("user_id", as_index=False).agg(
                item_count=("product_id", "size"),
                reordered_count=("reordered", "sum"),
            )
        )
        basket_blocks.append(
            chunk.groupby("order_id", as_index=False)
            .size()
            .rename(columns={"size": "basket_size"})
        )
        department_blocks.append(
            chunk.groupby(["user_id", "department_id"], as_index=False)
            .size()
            .rename(columns={"size": "item_count"})
        )

    user_item_counts = (
        pd.concat(user_item_blocks, ignore_index=True)
        .groupby("user_id")
        .sum()
    )
    user_item_counts["reorder_rate"] = (
        user_item_counts["reordered_count"] / user_item_counts["item_count"]
    )

    basket_sizes = (
        pd.concat(basket_blocks, ignore_index=True)
        .groupby("order_id", as_index=False)["basket_size"]
        .sum()
    )
    basket_sizes = basket_sizes.merge(
        prior_orders[["order_id", "user_id"]],
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    if basket_sizes["user_id"].isna().any():
        raise ValueError("Some prior baskets have no matching prior order")
    basket_features = basket_sizes.groupby("user_id")["basket_size"].agg(
        mean_basket_size="mean",
        median_basket_size="median",
        std_basket_size="std",
    )

    department_counts = (
        pd.concat(department_blocks, ignore_index=True)
        .groupby(["user_id", "department_id"], as_index=False)["item_count"]
        .sum()
    )
    department_matrix = department_counts.pivot(
        index="user_id", columns="department_id", values="item_count"
    ).fillna(0)
    department_shares = department_matrix.div(
        department_matrix.sum(axis=1), axis=0
    )
    department_names = departments.set_index("department_id")["department"].to_dict()
    department_shares.columns = [
        f"share_department_{_column_slug(department_names[int(department_id)])}"
        for department_id in department_shares.columns
    ]
    department_shares["departments_covered"] = department_matrix.gt(0).sum(axis=1)

    item_report = {
        "prior_item_rows": int(item_rows),
        "prior_orders_with_products": int(basket_sizes["order_id"].nunique()),
        "products_in_catalog": int(products["product_id"].nunique()),
        "departments_in_catalog": int(departments["department_id"].nunique()),
    }
    return (
        user_item_counts,
        basket_features.join(department_shares, how="outer"),
        item_report,
    )


def _fit_clusters(
    customer_features: pd.DataFrame,
    output_dir: Path,
    min_k: int = 2,
    max_k: int = 10,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    department_columns = [
        column
        for column in customer_features.columns
        if column.startswith("share_department_")
    ]
    cluster_columns = [
        "log_prior_order_count",
        "log_mean_basket_size",
        "log_median_basket_size",
        "log_mean_days_between_orders",
        "std_days_between_orders",
        "reorder_rate",
        "weekend_order_share",
        "evening_order_share",
        "mean_dow_sin",
        "mean_dow_cos",
        "mean_hour_sin",
        "mean_hour_cos",
        *department_columns,
    ]
    matrix = customer_features[cluster_columns].replace([np.inf, -np.inf], np.nan)
    matrix = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(
        matrix
    )
    matrix = RobustScaler().fit_transform(matrix).astype(np.float32, copy=False)

    if len(matrix) < max_k:
        raise ValueError("Not enough customers to evaluate the requested k range")
    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = rng.choice(
        len(matrix),
        size=min(SILHOUETTE_SAMPLE_SIZE, len(matrix)),
        replace=False,
    )

    results: list[dict[str, Any]] = []
    best_model: MiniBatchKMeans | None = None
    best_labels: np.ndarray | None = None
    best_silhouette = -1.0
    for k in range(min_k, max_k + 1):
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            batch_size=4096,
            n_init=5,
            max_iter=100,
        )
        labels = model.fit_predict(matrix)
        silhouette = float(
            silhouette_score(matrix[sample_indices], labels[sample_indices])
        )
        results.append(
            {
                "k": k,
                "inertie": float(model.inertia_),
                "silhouette_echantillon": silhouette,
            }
        )
        if silhouette > best_silhouette:
            best_model = model
            best_labels = labels
            best_silhouette = silhouette

    if best_model is None or best_labels is None:
        raise RuntimeError("No clustering model was fitted")

    scores = pd.DataFrame(results)
    selected_k = int(scores.loc[scores["silhouette_echantillon"].idxmax(), "k"])
    labels = pd.Series(
        best_labels,
        index=customer_features.index,
        name="cluster",
        dtype="int16",
    )
    report = {
        "algorithm": "MiniBatchKMeans",
        "k_selected_by_highest_sample_silhouette": selected_k,
        "silhouette_sample_size": int(len(sample_indices)),
        "sample_seed": RANDOM_STATE,
        "n_customers_clustered": int(len(customer_features)),
        "features_used_for_clustering": cluster_columns,
        "scaling": "SimpleImputer(median) + RobustScaler",
    }
    return labels, scores, report


def segment_customers(
    input_dir: Path,
    output_dir: Path,
    min_k: int = 2,
    max_k: int = 10,
) -> dict[str, Any]:
    required_files = [
        "orders.csv",
        "order_products__prior.csv",
        "products.csv",
        "departments.csv",
    ]
    missing = [name for name in required_files if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Instacart files: {', '.join(missing)}")
    if min_k < 2 or max_k < min_k:
        raise ValueError("Require 2 <= min_k <= max_k")

    started = time.perf_counter()
    prior_orders, order_report = _read_orders(input_dir / "orders.csv")
    order_features = _build_order_features(prior_orders)
    user_item_counts, item_features, item_report = _aggregate_prior_items(
        input_dir, prior_orders
    )

    customers = order_features.join(user_item_counts, how="inner").join(
        item_features, how="inner"
    )
    if customers.empty:
        raise ValueError("No customers have both prior orders and prior items")
    if customers.index.duplicated().any():
        raise ValueError("Customer profile contains duplicate user_id values")
    if customers["reorder_rate"].lt(0).any() or customers["reorder_rate"].gt(1).any():
        raise ValueError("Computed reorder_rate is outside [0, 1]")

    department_columns = [
        column
        for column in customers.columns
        if column.startswith("share_department_")
    ]
    department_share_sum = customers[department_columns].sum(axis=1)
    if not np.allclose(department_share_sum.to_numpy(), 1.0, atol=1e-5):
        raise ValueError("Department shares do not sum to 1 for every customer")

    customers["log_prior_order_count"] = np.log1p(customers["prior_order_count"])
    customers["log_mean_basket_size"] = np.log1p(customers["mean_basket_size"])
    customers["log_median_basket_size"] = np.log1p(customers["median_basket_size"])
    customers["log_mean_days_between_orders"] = np.log1p(
        customers["mean_days_between_orders"]
    )

    labels, scores, cluster_report = _fit_clusters(
        customers, output_dir, min_k=min_k, max_k=max_k
    )
    customers["cluster"] = labels
    department_share_columns = [
        column
        for column in customers.columns
        if column.startswith("share_department_")
    ]
    profile_columns = [
        "prior_order_count",
        "mean_days_between_orders",
        "mean_basket_size",
        "median_basket_size",
        "reorder_rate",
        "mean_order_hour",
        "mean_order_dow",
        "weekend_order_share",
        "evening_order_share",
        "departments_covered",
        *department_share_columns,
    ]
    profiles = customers.groupby("cluster")[profile_columns].mean()
    profiles.insert(0, "n_clients", customers.groupby("cluster").size())
    profiles.insert(
        1,
        "part_clients",
        (profiles["n_clients"] / len(customers)).round(4),
    )
    profiles["departement_dominant"] = (
        profiles[department_share_columns].idxmax(axis=1).str.replace(
            "share_department_", "", regex=False
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    segments_path = output_dir / "instacart_segments_clients.csv"
    profiles_path = output_dir / "instacart_profils_segments.csv"
    scores_path = output_dir / "instacart_scores_k.csv"
    plot_path = output_dir / "instacart_coude_silhouette.png"

    customer_output = customers.copy()
    customer_output.insert(0, "user_id", customer_output.index.astype("uint32"))
    customer_output.to_csv(segments_path, index=False, encoding="utf-8")
    profiles.to_csv(profiles_path, encoding="utf-8")
    scores.to_csv(scores_path, index=False, encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(scores["k"], scores["inertie"], marker="o")
    axes[0].set(title="Méthode du coude", xlabel="Nombre de clusters k", ylabel="Inertie")
    axes[1].plot(
        scores["k"], scores["silhouette_echantillon"], marker="o", color="darkorange"
    )
    axes[1].set(
        title="Silhouette sur échantillon",
        xlabel="Nombre de clusters k",
        ylabel="Score silhouette",
    )
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    report: dict[str, Any] = {
        **order_report,
        **item_report,
        **cluster_report,
        "n_customers_with_profile": int(len(customers)),
        "n_clusters": int(labels.nunique()),
        "cluster_sizes": {
            str(int(cluster)): int(size)
            for cluster, size in labels.value_counts().sort_index().items()
        },
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "outputs": {
            "customer_segments": str(segments_path),
            "cluster_profiles": str(profiles_path),
            "k_scores": str(scores_path),
            "elbow_silhouette_plot": str(plot_path),
        },
        "data_scope": (
            "Only eval_set='prior' orders and order_products__prior.csv were used; "
            "train/test evaluation orders were not used to build customer features."
        ),
    }
    report_path = output_dir / "instacart_segmentation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Segmentation terminée : {len(customers):,} clients, "
        f"k={report['n_clusters']}, silhouette "
        f"{scores.loc[scores['k'].eq(report['n_clusters']), 'silhouette_echantillon'].iloc[0]:.4f}."
    )
    print(f"Rapport : {report_path}")
    return report


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_dir.parent / "archive (1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "data" / "processed" / "instacart_segmentation",
    )
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=10)
    args = parser.parse_args()

    segment_customers(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_k=args.min_k,
        max_k=args.max_k,
    )


if __name__ == "__main__":
    main()
