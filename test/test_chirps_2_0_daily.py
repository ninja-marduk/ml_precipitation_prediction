import os
import pytest
import sys
import xarray as xr
import pandas as pd
from unittest.mock import patch, MagicMock
from data.load.test import (
    list_nc_files,
    load_and_crop_file,
    save_cropped_file,
    process_and_crop_files,
    combine_cropped_files,
    aggregate_monthly_coordinates_precipitation,
    save_to_netcdf
)

# Constants for testing
TEST_PATH = "/tmp/test_chirps/"
TEST_OUTPUT_PATH = "/tmp/test_output/"
LON_MIN, LON_MAX = -74.8, -71.9
LAT_MIN, LAT_MAX = 4.5, 7.3

@pytest.fixture
def mock_nc_file(tmp_path):
    """Create a mock NetCDF file for testing."""
    file_path = tmp_path / "test.nc"
    data = xr.Dataset(
        {
            "precip": (("time", "latitude", "longitude"), [[[1.0, 2.0], [3.0, 4.0]]])
        },
        coords={
            "time": pd.date_range("2000-01-01", periods=1),
            "latitude": [LAT_MIN, LAT_MAX],
            "longitude": [LON_MIN, LON_MAX],
        },
    )
    data.to_netcdf(file_path)
    return file_path

def test_list_nc_files(tmp_path):
    """Test listing NetCDF files in a directory."""
    # Create mock files
    (tmp_path / "file1.nc").touch()
    (tmp_path / "file2.nc").touch()
    (tmp_path / "not_a_nc.txt").touch()

    # Test function
    nc_files = list_nc_files(tmp_path)
    assert len(nc_files) == 2
    assert "file1.nc" in nc_files
    assert "file2.nc" in nc_files

def test_load_and_crop_file(mock_nc_file):
    """Test loading and cropping a NetCDF file."""
    cropped_ds = load_and_crop_file(mock_nc_file, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    assert "precip" in cropped_ds
    assert cropped_ds.dims["latitude"] == 2
    assert cropped_ds.dims["longitude"] == 2

def test_save_cropped_file(mock_nc_file, tmp_path):
    """Test saving a cropped NetCDF file."""
    cropped_ds = load_and_crop_file(mock_nc_file, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    save_cropped_file(cropped_ds, tmp_path, "cropped_test.nc")
    assert os.path.exists(tmp_path / "cropped_test.nc")

@patch("os.listdir", return_value=["file1.nc", "file2.nc"])
@patch("..data.load.chirps_2_0_daily.load_and_crop_file", return_value=MagicMock())
@patch("..data.load.chirps_2_0_daily.save_cropped_file")
def test_process_and_crop_files(mock_save, mock_load, mock_list, tmp_path):
    """Test processing and cropping multiple NetCDF files."""
    process_and_crop_files(TEST_PATH, TEST_OUTPUT_PATH, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    assert mock_load.call_count == 2
    assert mock_save.call_count == 2

def test_combine_cropped_files(mock_nc_file, tmp_path):
    """Test combining multiple cropped NetCDF files."""
    # Create mock cropped files
    cropped_file1 = tmp_path / "cropped1.nc"
    cropped_file2 = tmp_path / "cropped2.nc"
    ds = load_and_crop_file(mock_nc_file, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    ds.to_netcdf(cropped_file1)
    ds.to_netcdf(cropped_file2)

    # Test function
    combined_ds = combine_cropped_files(tmp_path)
    assert combined_ds.dims["time"] == 2

def test_aggregate_monthly_coordinates_precipitation(mock_nc_file):
    """Test aggregating precipitation data by month and coordinates."""
    ds = load_and_crop_file(mock_nc_file, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    aggregated_df = aggregate_monthly_coordinates_precipitation(ds)
    assert "mean_precipitation" in aggregated_df
    assert "max_precipitation" in aggregated_df
    assert "min_precipitation" in aggregated_df

def test_save_to_netcdf(mock_nc_file, tmp_path):
    """Test saving a DataFrame to a NetCDF file."""
    ds = load_and_crop_file(mock_nc_file, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    df = ds.to_dataframe().reset_index()
    save_to_netcdf(df, tmp_path, "test_output.nc")
    assert os.path.exists(tmp_path / "test_output.nc")
