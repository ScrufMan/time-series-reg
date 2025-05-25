from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


class EnergyModel:
    """
    Model energy patterns assuming an identical repeating daily pattern.
    """

    def __init__(self, csv_filepath: Path):
        self.csv_filepath = csv_filepath
        self.data = None
        self._load_and_preprocess_data()
        # Using LinearRegression without intercept, coefficients will be the means for each slot
        self.lr = LinearRegression(fit_intercept=False)

    def _load_and_preprocess_data(self):
        """
        Loads data from the CSV file and performs initial preprocessing:
        - Converts 'Time' column to datetime and sets as UTC index.
        - Drops missing values.
        - Creates a 'TimeSlot' column for modeling repeating patterns.
        """
        try:
            df = pd.read_csv(self.csv_filepath, sep=";")
        except FileNotFoundError:
            print(f"ERROR: File '{self.csv_filepath}' not found.")
            raise
        except Exception as e:
            print(f"ERROR: Could not read CSV file: {e}")
            raise

        df["Time"] = pd.to_datetime(df["Time"], errors="raise", utc=True)
        df.set_index("Time", inplace=True)
        df.sort_index(inplace=True)
        df.dropna(inplace=True)  # Drop rows with any NaN values in measurement columns

        # Create a 'TimeSlot' string ('00:00', '00:30', ..)
        df["TimeSlot"] = df.index.strftime("%H:%M")
        self.data = df
        self.all_possible_timeslots = sorted(df["TimeSlot"].unique().tolist())

    def _prepare_features_target(
        self, data_subset: pd.DataFrame, quantity_name: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepares features (X - one-hot encoded TimeSlot) and target (y).

        Args:
            data_subset (pd.DataFrame): The subset of data to use.
            quantity_name (str): The name of the target variable column.

        Returns:
            (X_dummies, y_target)
                X_dummies: DataFrame of one-hot encoded all possible time slots.
                y_target: Series for the target quantity.
        """
        assert (
            quantity_name in data_subset.columns
        ), f"Quantity '{quantity_name}' not found in the data columns."

        y_target = data_subset[quantity_name]

        # Create one-hot encoded features for 'TimeSlot'
        X_dummies = pd.get_dummies(data_subset["TimeSlot"], prefix="slot")

        # Ensure all possible timeslot columns are present, fill with 0 if missing
        for slot in self.all_possible_timeslots:
            col_name = f"slot_{slot}"
            if col_name not in X_dummies.columns:
                X_dummies[col_name] = 0

        # Make sure the columns are in the correct order (all_possible_timeslots)
        X_dummies = X_dummies[map(lambda x: f"slot_{x}", self.all_possible_timeslots)]

        return X_dummies, y_target

    def fit(self, quantity_name: str) -> pd.Series:
        """
        Fits a linear regression model to establish the typical daily pattern
        for the specified quantity.
        Since the model assumes a repeating daily pattern, LR should learn only the means
        of each time slot.

        Args:
            quantity_name (str): The name of the quantity to model.

        Returns:
            pd.Series: The daily model profile (index: TimeSlot str, values: predicted means).
        """
        assert (
            self.data is not None
        ), "Data not loaded. Call _load_and_preprocess_data() first."

        X_full, y_full = self._prepare_features_target(self.data, quantity_name)

        self.lr.fit(X_full, y_full)

        # The daily model profile are the coefficients of the regression model
        daily_model_profile_np = self.lr.coef_

        # Create a Series for the profile with 'HH:MM' strings as index
        daily_model_profile = pd.Series(
            daily_model_profile_np, index=self.all_possible_timeslots
        )
        daily_model_profile.name = f"{quantity_name}_profile"

        return daily_model_profile

    def predict_from_profile(
        self, daily_profile: pd.Series, target_index: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Generates predictions for a given time index using the daily profile.

        Args:
            daily_profile (pd.Series): A Series representing the daily pattern
                                       (index: 'HH:MM' TimeSlot, values: predicted means).
            target_index (pd.DatetimeIndex): The time index for which to generate predictions.

        Returns:
            pd.Series: Predicted values aligned with the target_index.
        """
        # Map the time part of the target_index to the profile
        time_slots_for_prediction = target_index.strftime("%H:%M")
        predicted_values = pd.Series(index=target_index, dtype=float)

        for slot_str, value in daily_profile.items():
            predicted_values[time_slots_for_prediction == slot_str] = value

        return predicted_values

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
        """
        Evaluates the model fit using RMSE and a NRMSE based accuracy metric.

        Args:
            y_true (pd.Series): True values.
            y_pred (pd.Series): Predicted values from the model.

        Returns:
            tuple: (rmse, accuracy)
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # https://lifesight.io/glossary/normalized-root-mean-square-error/
        y_true_range = y_true.max() - y_true.min()

        if (
            y_true_range == 0
        ):  # Handles cases like 'Grid backflow' where variation is zero
            if rmse == 0:
                accuracy_percent = 100.0
            else:
                accuracy_percent = 0.0
        else:
            nrmse = rmse / y_true_range
            accuracy_percent = (1 - nrmse) * 100
            accuracy_percent = max(0, accuracy_percent)

        return rmse, accuracy_percent

    def plot_data_and_model(self, quantity_name: str, daily_model_profile: pd.Series):
        """
        Plots the original data for the quantity along with the fitted model.

        Args:
            quantity_name (str): The name of the quantity to plot.
            daily_model_profile (pd.Series): The fitted daily pattern.
        """
        actual_data = self.data[quantity_name]
        predicted_data = self.predict_from_profile(daily_model_profile, self.data.index)

        plt.figure(figsize=(15, 8))
        plt.plot(
            actual_data.index,
            actual_data,
            label=f"Actual {quantity_name}",
            alpha=0.5,
            color="blue",
        )
        plt.plot(
            predicted_data.index,
            predicted_data,
            label=f"Fitted Model (Daily Pattern for {quantity_name})",
            linestyle="--",
            color="red",
        )

        plt.title(f"Actual Data vs. Fitted Daily Pattern for {quantity_name}")
        plt.xlabel("Time (UTC)")
        plt.ylabel("Power (W)")

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))  # Every 3 days
        plt.gcf().autofmt_xdate()

        plt.legend()
        plt.grid(True, which="major", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    def perform_timeseries_cv(self, quantity_name: str, n_splits: int = 10) -> float:
        """
        Performs time series cross-validation for the given quantity.
        The model is retrained on each training fold.

        Args:
            quantity_name (str): The quantity to evaluate.
            n_splits (int): Number of splits for TimeSeriesSplit.

        Returns:
            average_rmse
        """
        print(f"Time Series CV for '{quantity_name}' (k={n_splits}):\n")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        rmse_scores = []

        for fold, (train_index, test_index) in enumerate(tscv.split(self.data)):

            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]

            # Prepare features and target for this training fold
            X_train, y_train = self._prepare_features_target(train_data, quantity_name)

            # Train a new model on this training fold
            self.lr.fit(X_train, y_train)

            # Create the daily profile from this training fold coefs
            cv_daily_profile_values = self.lr.coef_
            cv_daily_profile = pd.Series(
                cv_daily_profile_values, index=self.all_possible_timeslots
            )

            # Generate predictions for the test fold using the profile
            y_pred_cv = self.predict_from_profile(cv_daily_profile, test_data.index)
            y_true_cv = test_data[quantity_name]

            fold_rmse = np.sqrt(mean_squared_error(y_true_cv, y_pred_cv))
            rmse_scores.append(fold_rmse)
            print(
                f"CV Fold {fold+1}/{n_splits} (Train: {len(train_index)} Test: {len(test_index) }): RMSE = {fold_rmse:.2f}"
            )

        average_rmse = np.mean(rmse_scores)
        return average_rmse
