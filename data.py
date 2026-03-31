import os
import sys

import pandas as pd

from preprocessing_modules import (
    DataCleaner,
    add_anonymous_principal_flag,
    add_datetime_epoch_seconds,
    add_time_features_and_finalize_numeric,
    add_selective_label_encoding,
    drop_columns_over_missing_fraction,
    impute_values,
    remove_outliers_iqr,
    save_label_mappings_txt,
    save_ml_csv,
)


def preprocess(input_csv: str, output_csv: str, missing_threshold: float = 0.7) -> str:
    df = pd.read_csv(input_csv, low_memory=False)

    cleaner = DataCleaner(df)
    cleaner.remove_empty_rows_cols()
    cleaner.fix_datetime_columns()
    cleaner.normalize_string_columns()
    cleaner.clean_boolean_columns()
    cleaner.remove_duplicates()
    cleaner.feature_selection()
    
    df = cleaner.df

    df, _ = drop_columns_over_missing_fraction(df, missing_threshold=missing_threshold)
    df = impute_values(df)
    df, outlier_report = remove_outliers_iqr(df)
    df = add_anonymous_principal_flag(df)
    df = add_datetime_epoch_seconds(df)
    df, mappings = add_selective_label_encoding(df)
    df, final_drop_report = add_time_features_and_finalize_numeric(df)
    save_ml_csv(df, output_csv)
    mapping_path = os.path.join(os.path.dirname(output_csv), "label_mappings.txt")
    save_label_mappings_txt(mappings, mapping_path)
    print("Outlier report:", outlier_report)
    print("Final numeric drop report:", final_drop_report)
    print("Label mappings:", mapping_path)
    return output_csv


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(root, "dataset.csv")
    output_path = os.path.join(root, "processedfiles", "ml_ready.csv")

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    if not os.path.isfile(input_path):
        print("Missing input CSV:", input_path)
        sys.exit(1)

    preprocess(input_path, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
