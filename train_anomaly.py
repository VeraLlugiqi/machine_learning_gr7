"""
Hyrje e shkurtër për kompatibilitet.

Përdor:
- python train_anomaly.py --method isolation_forest
- python train_anomaly.py --method local_outlier_factor
- python train_anomaly.py --method svm
- python train_anomaly.py --method one_class_svm
- python train_anomaly.py --method elliptic_envelope
"""
import argparse
import sys

from anomaly_models.isolation_forest import main as isolation_forest_main
from anomaly_models.local_outlier_factor import main as local_outlier_factor_main
from anomaly_models.one_class_svm import main as one_class_svm_main
from anomaly_models.elliptic_envelope import main as elliptic_envelope_main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zgjidh modelin e anomaly detection për trajnim."
    )
    parser.add_argument(
        "--method",
        choices=[
            "isolation_forest",
            "local_outlier_factor",
            "one_class_svm",
            "svm",
            "elliptic_envelope",
        ],
        default="isolation_forest",
        help="Algoritmi që do të ekzekutohet.",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    if args.method == "local_outlier_factor":
        local_outlier_factor_main()
        return

    if args.method in {"one_class_svm", "svm"}:
        one_class_svm_main()
        return

    if args.method == "elliptic_envelope":
        elliptic_envelope_main()
        return

    isolation_forest_main()


if __name__ == "__main__":
    main()
