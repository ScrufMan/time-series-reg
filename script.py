import argparse
from pathlib import Path

from energy_model import EnergyModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to the input CSV file."
    )
    parser.add_argument(
        "--quantity",
        type=str,
        required=True,
        choices=[
            "Consumption",
            "Grid consumption",
            "Grid backflow",
            "PV generation",
            "Battery charging",
            "Battery discharging",
        ],
        help="Name of the quantity to model.",
    )

    args = parser.parse_args()

    print(f"Starting analysis for input: {args.input}, quantity: {args.quantity}")

    try:
        model = EnergyModel(csv_filepath=args.input)

        # Fit the model on the entire dataset to get the primary daily profile
        print(f"\n--- Fitting model for '{args.quantity}' on the full dataset ---")
        daily_model_profile = model.fit(args.quantity)
        print("Daily Model Profile:")
        print(daily_model_profile)

        # Generate predictions for the entire dataset using this profile
        all_data_predictions = model.predict_from_profile(
            daily_model_profile, model.data.index
        )

        # Evaluate the model on the entire dataset
        actual_values_full = model.data[args.quantity]
        rmse, accuracy_percent = model.evaluate(
            actual_values_full, all_data_predictions
        )
        print(f"\n--- Model Evaluation on Full Dataset for '{args.quantity}' ---")
        print(f"Overall RMSE: {rmse:.2f}")
        print(
            f"The model is {accuracy_percent:.1f}% accurate (based on 1 - NRMSE_range)."
        )

        # Perform time-series cross-validation
        print(
            f"\n--- Performing Time Series Cross-Validation for '{args.quantity}' ---"
        )
        avg_cv_rmse = model.perform_timeseries_cv(args.quantity)
        print(f"\nAverage RMSE across folds: {avg_cv_rmse:.2f}\n")

        # Plot data and the fitted model (on the full dataset fit)
        print("\n--- Plotting Data and Fitted Model ---")
        model.plot_data_and_model(args.quantity, daily_model_profile)

        print("\nAnalysis complete.")

    except Exception as e:
        print(f"An error occurred during the main execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
