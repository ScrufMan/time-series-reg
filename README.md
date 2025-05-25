# Energy Pattern Analysis Model

This project provides a Python script to model and analyze energy consumption/generation patterns based on the assumption of an identical repeating daily cycle.

## Features

* Loads energy data from a CSV file.
* Models a chosen energy quantity (e.g., Consumption, PV generation) by calculating a typical daily profile using Linear Regression.
* The model assumes the daily pattern is repeating and identical.
* Evaluates model fit using RMSE and a normalized accuracy metric.
* Performs time-series cross-validation.
* Generates a plot comparing actual data with the fitted daily pattern model.

## Requirements

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* pytest (for running tests)

## Installation

1. **Clone the repository:**

    ```bash
    git clone git@github.com:ScrufMan/time-series-reg.git
    cd time-series-reg
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Statistics

The `stats.py` module provides functions to calculate various statistics for the energy data and plot graph and correlation heatmap.

**Command-line arguments:**

* `--input`: Path to the input CSV file (e.g., `SG.csv`).

**Example:**

```bash
python stats.py --input SG.csv
```

### Running the Model

The main script `script.py` is used to run the analysis. It requires the path to the input CSV file and the name of the quantity to analyze.

**Command-line arguments:**

* `--input`: Path to the input CSV file (e.g., `SG.csv`).
* `--quantity`: Name of the quantity to model.
    Choices: `Consumption`, `Grid consumption`, `Grid backflow`, `PV generation`, `Battery charging`, `Battery discharging`.

**Example:**

```bash
python script.py --input SG.csv --quantity Consumption
```

### Running Tests

To run the tests, use the following command:

```bash
pytest -v
```
