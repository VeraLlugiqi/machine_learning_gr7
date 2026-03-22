import os
import sys

import pandas as pd

from preprocessing_modules import (
    DataCleaner,
    add_anonymous_principal_flag,
    add_caller_ip_first_octet,
    add_datetime_epoch_seconds,
    add_selective_label_encoding,
    drop_columns_over_missing_fraction,
    impute_values,
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
    df = add_anonymous_principal_flag(df)
    df = add_datetime_epoch_seconds(df)
    df = add_caller_ip_first_octet(df)
    df = add_selective_label_encoding(df)
    save_ml_csv(df, output_csv)
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
