import pytest
import pandas as pd
from pathlib import Path
import numpy as np

from energy_model import EnergyModel

# --- Test Data ---


TEST_CSV_DATA = (
    "Time;Consumption;Grid consumption;Grid backflow;PV generation;Battery charging;Battery discharging\n"
    "2025-01-01T00:00:00Z;100;80;0;0;20;0\n"
    "2025-01-01T00:30:00Z;120;90;0;0;30;0\n"
    "2025-01-01T01:00:00Z;110;110;0;0;0;0\n"
    "2025-01-02T00:00:00Z;150;120;0;0;30;0\n"
    "2025-01-02T00:30:00Z;180;150;0;0;30;0\n"
    "2025-01-02T01:00:00Z;160;160;0;0;0;0\n"
    "2025-01-03T00:00:00Z;125;100;0;0;25;0\n"
    # Adding a row with NaN to test dropna
    "2025-01-03T00:30:00Z;;160;0;0;30;0\n"  # Missing Consumption
    "2025-01-03T01:00:00Z;135;135;0;0;0;0\n"
)


# --- Fixtures ---


@pytest.fixture
def temp_csv_file(tmp_path: Path) -> Path:
    """
    Creates a temporary CSV file with test data and returns its path.
    tmp_path is a built-in pytest fixture.
    """
    filepath = tmp_path / "test_sg_data.csv"
    filepath.write_text(TEST_CSV_DATA)
    return filepath


@pytest.fixture
def energy_model_instance(temp_csv_file: Path) -> EnergyModel:
    """
    Provides an instance of EnergyModel initialized with the temporary test CSV.
    """
    return EnergyModel(csv_filepath=temp_csv_file)


# --- Unit Tests ---


def test_load_and_preprocess_data(energy_model_instance: EnergyModel):
    """
    Test 1: Verifies correct data loading, preprocessing, and attribute initialization.
    """
    model = energy_model_instance

    # Check that data is loaded and is a DataFrame
    assert isinstance(model.data, pd.DataFrame)

    # Check that the index is a DatetimeIndex and UTC timezone is set
    assert isinstance(model.data.index, pd.DatetimeIndex)
    assert model.data.index.tzinfo == pd.Timestamp("2025-01-01", tz="UTC").tzinfo

    # Check that rows with NaN were dropped
    # Original data has 10 lines (1 header + 9 data). One data row has NaN in Consumption.
    # So, after dropna there should be 8 data rows.
    assert len(model.data) == 8
    assert not model.data["Consumption"].isnull().any()  # Ensure no NaNs in key column

    # Check for 'TimeSlot' column and its content for a sample
    assert "TimeSlot" in model.data.columns
    assert model.data.iloc[0]["TimeSlot"] == "00:00"
    assert model.data.iloc[-1]["TimeSlot"] == "01:00"

    # Check 'all_possible_timeslots'
    # Unique timeslots are 00:00, 00:30, 01:00
    expected_slots = sorted(["00:00", "00:30", "01:00"])
    assert model.all_possible_timeslots == expected_slots


def test_fit_calculates_correct_daily_profile(energy_model_instance: EnergyModel):
    """
    Test 2: Verifies that the 'fit' method correctly calculates the daily profile.
    Since LR is used, the profile values should be the mean of the quantity for each time slot.
    """
    model = energy_model_instance
    quantity_to_test = "Consumption"

    slot1_mean = (100 + 150 + 125) / 3
    slot2_mean = (120 + 180) / 2
    slot3_mean = (110 + 160 + 135) / 3

    expected_profile_values = {
        "00:00": slot1_mean,
        "00:30": slot2_mean,
        "01:00": slot3_mean,
    }

    daily_profile = model.fit(quantity_to_test)

    assert isinstance(daily_profile, pd.Series)
    assert daily_profile.name == f"{quantity_to_test}_profile"
    assert len(daily_profile) == len(model.all_possible_timeslots)

    for slot, expected_mean in expected_profile_values.items():
        assert daily_profile.loc[slot] == pytest.approx(expected_mean)


def test_predict_from_profile_generates_correct_predictions(
    energy_model_instance: EnergyModel,
):
    """
    Test 3: Verifies that 'predict_from_profile' correctly uses the daily profile
    to generate predictions for a given DatetimeIndex.
    """
    model = energy_model_instance
    quantity_to_test = "Consumption"

    # Previous test asserts that the profile is calculated correctly.
    fitted_profile = model.fit(quantity_to_test)

    # Create a target DatetimeIndex for prediction
    test_times = [
        "2025-01-04T00:00:00Z",
        "2025-01-04T00:30:00Z",
        "2025-01-04T01:00:00Z",
        "2025-01-05T00:00:00Z",
        "2025-01-05T00:30:00Z",
    ]
    target_index = pd.to_datetime(test_times, utc=True)

    predictions = model.predict_from_profile(fitted_profile, target_index)

    assert isinstance(predictions, pd.Series)
    assert predictions.index.equals(target_index)

    # Check values based on the fitted_profile from the previous test's expected values
    # Slot 00:00: 125.0
    # Slot 00:30: 150.0
    # Slot 01:00: 135.0
    assert predictions.loc[
        pd.Timestamp("2025-01-04T00:00:00Z", tz="UTC")
    ] == pytest.approx(125.0)
    assert predictions.loc[
        pd.Timestamp("2025-01-04T00:30:00Z", tz="UTC")
    ] == pytest.approx(150.0)
    assert predictions.loc[
        pd.Timestamp("2025-01-04T01:00:00Z", tz="UTC")
    ] == pytest.approx(135.0)
    assert predictions.loc[
        pd.Timestamp("2025-01-05T00:00:00Z", tz="UTC")
    ] == pytest.approx(125.0)
    assert predictions.loc[
        pd.Timestamp("2025-01-05T00:30:00Z", tz="UTC")
    ] == pytest.approx(150.0)
