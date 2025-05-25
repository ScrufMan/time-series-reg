from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import argparse


def analyze_and_plot_data(csv_path: Path):
    """
    Loads a CSV file, processes the data, and plots the time series of energy measurements.
    """
    # 1. Load CSV file
    print(f"Loading: {csv_path}\n")
    try:
        df = pd.read_csv(csv_path, sep=";")
        print(f"Rows: {len(df)}, Cols: {len(df.columns)}")
    except FileNotFoundError:
        print(f"ERROR: File '{csv_path}' not found.")
        return
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # 2. Descriptive statistics on raw data
    print("\n--- First 5 rows of the DataFrame ---")
    print(df.head())

    print("\n--- DataFrame info ---")
    print(df.info())

    print("\n--- Descriptive statistics ---")
    print(df.describe())

    print("\n--- Checking for missing values ---")
    print(df.isnull().sum())

    print("\n--- Checking for duplicate rows ---")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")

    # 3. Remove rows with NaN
    print("\n--- Removing rows with NaN values ---")
    initial_row_count = len(df)
    df.dropna(inplace=True)
    final_row_count = len(df)
    print(f"Removed {initial_row_count - final_row_count} rows with NaN values.")

    # 4. Process 'Time' column
    print("\n--- Processing 'Time' column ---")
    try:
        df["Time"] = pd.to_datetime(df["Time"], errors="raise", utc=True)
        df.set_index("Time", inplace=True)
        df.sort_index(inplace=True)
    except Exception as e:
        print(f"ERROR: Cannot convert 'Time' column: {e}")
        return

    print(f"Measurments interval: from {df.index.min()} to {df.index.max()}")

    # 5. Plot the data
    plt.figure(figsize=(15, 8))
    for column in df.columns:
        if df[column].dtype in [
            "float64",
            "int64",
        ]:  # Plot only numeric columns
            plt.plot(df.index, df[column], label=column, alpha=0.8)

    plt.title("Timespan (30 days)")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")

    # Format x-axis for better readability
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))  # Every 3 days
    plt.gcf().autofmt_xdate()  # Rotate date labels for better visibility

    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True, which="major", linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    # 6. Show correlation matrix
    # Correlation matrix only for numeric columns
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to the input CSV file."
    )

    args = parser.parse_args()

    try:
        analyze_and_plot_data(args.input)
    except Exception as e:
        print(f"An error occurred during the main execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
