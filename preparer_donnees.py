"""Prepare a branded-food classification dataset and a leakage-safe split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


NUTRIENT_FEATURES = {
    "1008": ("energy_kcal", "Energy", "KCAL"),
    "1003": ("protein_g", "Protein", "G"),
    "1004": ("total_fat_g", "Total lipid (fat)", "G"),
    "1005": ("carbohydrate_g", "Carbohydrate, by difference", "G"),
    "2000": ("total_sugars_g", "Total Sugars", "G"),
    "1093": ("sodium_mg", "Sodium, Na", "MG"),
    "1079": ("fiber_g", "Fiber, total dietary", "G"),
    "1258": ("saturated_fat_g", "Fatty acids, total saturated", "G"),
}

BRANDED_COLUMNS = [
    "fdc_id",
    "brand_owner",
    "brand_name",
    "gtin_upc",
    "ingredients",
    "serving_size",
    "serving_size_unit",
    "branded_food_category",
    "modified_date",
    "short_description",
]


def clean_text(values: pd.Series) -> pd.Series:
    """Normalize whitespace and common encoding artifacts without guessing labels."""
    cleaned = values.fillna("").astype("string").str.normalize("NFKC")
    cleaned = cleaned.str.replace("\ufffd", " ", regex=False)
    cleaned = cleaned.str.replace(r"[\u2010-\u2015\u2212]", "-", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned.mask(cleaned.eq(""), pd.NA)


def load_nutrient_names(raw_dir: Path) -> dict[str, dict[str, str]]:
    nutrients = pd.read_csv(
        raw_dir / "nutrient.csv",
        usecols=["id", "name", "unit_name"],
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    names = nutrients.set_index("id")[["name", "unit_name"]].to_dict("index")
    for nutrient_id, (_, expected_name, expected_unit) in NUTRIENT_FEATURES.items():
        actual = names.get(nutrient_id)
        if actual is None:
            raise ValueError(f"Nutrient {nutrient_id} is missing from nutrient.csv")
        if actual["name"] != expected_name or actual["unit_name"].upper() != expected_unit:
            raise ValueError(
                f"Unexpected definition for nutrient {nutrient_id}: {actual}"
            )
    return names


def load_latest_products(
    raw_dir: Path, chunk_size: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    winners = pd.DataFrame()
    raw_rows = 0
    missing_upc_rows = 0
    invalid_id_rows = 0
    invalid_serving_sizes = 0

    for chunk in pd.read_csv(
        raw_dir / "branded_food.csv",
        usecols=BRANDED_COLUMNS,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
        chunksize=chunk_size,
    ):
        raw_rows += len(chunk)
        for column in (
            "brand_owner",
            "brand_name",
            "ingredients",
            "serving_size_unit",
            "branded_food_category",
            "short_description",
        ):
            chunk[column] = clean_text(chunk[column])

        chunk["gtin_upc"] = chunk["gtin_upc"].astype("string").str.strip()
        missing_upc_rows += int(
            (chunk["gtin_upc"].isna() | chunk["gtin_upc"].eq("")).sum()
        )
        chunk = chunk.loc[chunk["gtin_upc"].notna() & chunk["gtin_upc"].ne("")].copy()

        chunk["_fdc_numeric"] = pd.to_numeric(chunk["fdc_id"], errors="coerce")
        invalid_id_rows += int(chunk["_fdc_numeric"].isna().sum())
        chunk = chunk.loc[chunk["_fdc_numeric"].notna()].copy()

        chunk["_modified_at"] = pd.to_datetime(
            chunk["modified_date"], errors="coerce", format="mixed"
        )
        serving_size = pd.to_numeric(chunk["serving_size"], errors="coerce")
        valid_serving_size = serving_size.notna() & np.isfinite(serving_size) & serving_size.ge(0)
        invalid_serving_sizes += int((chunk["serving_size"].ne("") & ~valid_serving_size).sum())
        chunk["serving_size"] = serving_size.where(valid_serving_size)
        chunk["serving_size_unit"] = chunk["serving_size_unit"].str.casefold()

        chunk = chunk.sort_values(
            ["gtin_upc", "_modified_at", "_fdc_numeric"],
            ascending=[True, False, False],
            kind="stable",
            na_position="last",
        ).drop_duplicates("gtin_upc", keep="first")

        if not winners.empty:
            chunk = pd.concat([winners, chunk], ignore_index=True)
        winners = chunk.sort_values(
            ["gtin_upc", "_modified_at", "_fdc_numeric"],
            ascending=[True, False, False],
            kind="stable",
            na_position="last",
        ).drop_duplicates("gtin_upc", keep="first")

    if invalid_id_rows:
        raise ValueError(f"{invalid_id_rows} rows have an invalid fdc_id")

    winners = winners.drop(columns=["_fdc_numeric", "_modified_at"]).reset_index(drop=True)
    counts = {
        "source_branded_food_rows": raw_rows,
        "latest_unique_upc_rows": len(winners),
        "rows_missing_upc": missing_upc_rows,
        "rows_invalid_fdc_id": invalid_id_rows,
        "invalid_or_negative_serving_sizes_set_missing": invalid_serving_sizes,
        "latest_rows_missing_category": int(winners["branded_food_category"].isna().sum()),
    }
    winners = winners.loc[winners["branded_food_category"].notna()].copy()
    return winners.reset_index(drop=True), counts


def load_food_descriptions(
    raw_dir: Path, fdc_ids: set[str], chunk_size: int
) -> pd.DataFrame:
    matches: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        raw_dir / "food.csv",
        usecols=["fdc_id", "description", "data_type"],
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
        chunksize=chunk_size,
    ):
        chunk = chunk.loc[
            chunk["fdc_id"].isin(fdc_ids) & chunk["data_type"].eq("branded_food"),
            ["fdc_id", "description"],
        ].copy()
        if not chunk.empty:
            chunk["description"] = clean_text(chunk["description"])
            matches.append(chunk)

    if not matches:
        raise ValueError("No matching branded-food rows found in food.csv")
    descriptions = pd.concat(matches, ignore_index=True)
    if descriptions["fdc_id"].duplicated().any():
        raise ValueError("food.csv contains duplicate fdc_id values for selected products")
    if set(descriptions["fdc_id"]) != fdc_ids:
        missing = len(fdc_ids - set(descriptions["fdc_id"]))
        raise ValueError(f"food.csv is missing {missing} selected fdc_id values")
    return descriptions


def load_nutrient_values(
    raw_dir: Path,
    fdc_ids: set[str],
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    nutrient_ids = set(NUTRIENT_FEATURES)
    matches: list[pd.DataFrame] = []
    matched_rows = 0
    invalid_amounts = 0

    for chunk in pd.read_csv(
        raw_dir / "food_nutrient.csv",
        usecols=["fdc_id", "nutrient_id", "amount"],
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False,
        chunksize=chunk_size,
    ):
        chunk = chunk.loc[
            chunk["fdc_id"].isin(fdc_ids) & chunk["nutrient_id"].isin(nutrient_ids),
            ["fdc_id", "nutrient_id", "amount"],
        ].copy()
        if chunk.empty:
            continue

        matched_rows += len(chunk)
        amounts = pd.to_numeric(chunk["amount"], errors="coerce")
        valid_amount = amounts.notna() & np.isfinite(amounts) & amounts.ge(0)
        invalid_amounts += int((chunk["amount"].ne("") & ~valid_amount).sum())
        chunk["amount"] = amounts.where(valid_amount)
        matches.append(chunk)

    if matches:
        selected = pd.concat(matches, ignore_index=True)
    else:
        selected = pd.DataFrame(columns=["fdc_id", "nutrient_id", "amount"])

    if selected.duplicated(["fdc_id", "nutrient_id"]).any():
        raise ValueError(
            "Multiple values found for a selected fdc_id/nutrient_id pair; "
            "aggregation must be chosen before continuing"
        )

    values = selected.pivot(index="fdc_id", columns="nutrient_id", values="amount")
    values = values.rename(
        columns={nutrient_id: feature[0] for nutrient_id, feature in NUTRIENT_FEATURES.items()}
    )
    values = values.reindex(columns=[feature[0] for feature in NUTRIENT_FEATURES.values()])
    values = values.reset_index()
    return values, {
        "selected_nutrient_rows_joined": matched_rows,
        "invalid_or_negative_nutrient_amounts_set_missing": invalid_amounts,
    }


def prepare_data(
    raw_dir: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    chunk_size: int = 100_000,
) -> dict[str, Any]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    nutrient_names = load_nutrient_names(raw_dir)
    products, report = load_latest_products(raw_dir, chunk_size)
    fdc_ids = set(products["fdc_id"])

    descriptions = load_food_descriptions(raw_dir, fdc_ids, chunk_size)
    products = products.merge(
        descriptions, on="fdc_id", how="left", validate="one_to_one"
    )
    products["product_description"] = products["description"].combine_first(
        products["short_description"]
    )

    nutrient_values, nutrient_report = load_nutrient_values(
        raw_dir, fdc_ids, chunk_size
    )
    products = products.merge(
        nutrient_values, on="fdc_id", how="left", validate="one_to_one"
    )

    feature_columns = [
        "product_description",
        "brand_owner",
        "brand_name",
        "ingredients",
        "serving_size",
        "serving_size_unit",
        *[feature[0] for feature in NUTRIENT_FEATURES.values()],
    ]
    target_column = "branded_food_category"
    model_data = products[feature_columns + [target_column]].copy()

    labels = model_data[target_column]
    class_counts = labels.value_counts()
    singleton_mask = labels.map(class_counts).eq(1)
    singleton_indices = np.flatnonzero(singleton_mask.to_numpy())
    stratifiable_indices = np.flatnonzero(~singleton_mask.to_numpy())
    if not len(stratifiable_indices):
        raise ValueError("No category has enough rows for a stratified train/test split")

    train_indices, test_indices = train_test_split(
        stratifiable_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels.iloc[stratifiable_indices],
    )
    test_categories = set(labels.iloc[test_indices])
    categories_added_to_test = class_counts.index[
        class_counts.gt(1) & ~class_counts.index.isin(test_categories)
    ]
    for category in categories_added_to_test:
        category_rows = train_indices[
            labels.iloc[train_indices].eq(category).to_numpy()
        ]
        if len(category_rows) < 2:
            raise ValueError(f"Cannot place category {category!r} in both splits")
        moved_index = category_rows[-1]
        train_indices = train_indices[train_indices != moved_index]
        test_indices = np.append(test_indices, moved_index)

    train_indices = np.concatenate([train_indices, singleton_indices])
    train_data = model_data.iloc[train_indices].reset_index(drop=True)
    test_data = model_data.iloc[test_indices].reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_data.to_csv(train_path, index=False, encoding="utf-8")
    test_data.to_csv(test_path, index=False, encoding="utf-8")

    report.update(nutrient_report)
    report.update(
        {
            "rows_after_latest_version_and_category_filter": len(model_data),
            "train_rows": len(train_data),
            "test_rows": len(test_data),
            "test_fraction_actual": round(len(test_data) / len(model_data), 6),
            "target_column": target_column,
            "target_class_count": int(labels.nunique()),
            "test_target_class_count": int(test_data[target_column].nunique()),
            "categories_added_to_test_for_coverage": int(len(categories_added_to_test)),
            "singleton_categories_kept_in_train": int(len(singleton_indices)),
            "random_state": random_state,
            "test_size_requested": test_size,
            "feature_columns": feature_columns,
            "identifiers_excluded_from_model_files": ["fdc_id", "gtin_upc"],
            "nutrient_features": {
                nutrient_names[nutrient_id]["name"]: {
                    "column": feature[0],
                    "nutrient_id": nutrient_id,
                    "unit": nutrient_names[nutrient_id]["unit_name"],
                    "non_missing_rows": int(model_data[feature[0]].notna().sum()),
                }
                for nutrient_id, feature in NUTRIENT_FEATURES.items()
            },
            "train_target_counts": {
                str(key): int(value) for key, value in train_data[target_column].value_counts().items()
            },
            "test_target_counts": {
                str(key): int(value) for key, value in test_data[target_column].value_counts().items()
            },
            "joins": [
                "branded_food.fdc_id = food.fdc_id (one-to-one)",
                "food_nutrient.fdc_id = branded_food.fdc_id, filtered to selected nutrient IDs",
                "nutrient.id = food_nutrient.nutrient_id (nutrient labels/units verified)",
            ],
            "excluded_as_not_applicable_or_not_needed_for_this_branded_food_model": [
                "food_attribute and food_attribute_type",
                "food_portion and measure_unit",
                "food_component",
                "food_nutrient_conversion_factor and food_calorie_conversion_factor",
                "input_food",
                "food_update_log_entry",
                "microbe",
                "sample_food and sub_sample_food",
            ],
            "cleaning_notes": [
                "Text normalized with Unicode NFKC, replacement characters removed, and whitespace collapsed.",
                "The latest modified_date row per non-empty GTIN/UPC was retained; ties use the largest numeric fdc_id.",
                "Latest rows with no category were excluded; singleton categories stay in training only.",
                "Invalid or negative serving sizes and selected nutrient amounts were set missing, not imputed.",
                "Nutrient amounts are kept in source-reported units; they are not rescaled.",
                "No ID or GTIN/UPC column is written to model train/test files.",
            ],
            "output_files": [str(train_path), str(test_path)],
        }
    )
    report_path = output_dir / "quality_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "raw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "processed",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    report = prepare_data(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        chunk_size=args.chunk_size,
    )
    print(
        f"Terminé : {report['train_rows']:,} lignes train, "
        f"{report['test_rows']:,} lignes test, "
        f"{report['target_class_count']} catégories."
    )
    print(f"Rapport qualité : {args.output_dir / 'quality_report.json'}")


if __name__ == "__main__":
    main()
