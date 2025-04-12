import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import numpy as np

# Constants
DEM_PATH_90 = "/Users/riperez/Conda/anaconda3/doc/precipitation/qgis_output/dem_boyaca_90.tif"
SHAPEFILE_BOYACA = "/Users/riperez/Conda/anaconda3/doc/precipitation/shapes/MGN_Departamento.shp"
OUTPUT_NETCDF_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/dem_boyaca_90.nc"

def load_dem(dem_path):
    """
    Load a DEM GeoTIFF file and return its data, metadata, and coordinates.

    Parameters:
        dem_path (str): Path to the DEM GeoTIFF file.

    Returns:
        tuple: A tuple containing the DEM data (numpy array), metadata (dict), and coordinates (longitude, latitude).
    """
    with rasterio.open(dem_path) as dem_dataset:
        dem_data = dem_dataset.read(1)  # Read the first band
        dem_meta = dem_dataset.meta  # Get metadata

        # Extract coordinates
        transform = dem_dataset.transform
        width = dem_dataset.width
        height = dem_dataset.height
        lon = np.arange(transform[2], transform[2] + width * transform[0], transform[0])
        lat = np.arange(transform[5], transform[5] + height * transform[4], transform[4])

    return dem_data, dem_meta, lon, lat


def plot_dem_with_boundary(dem_data, dem_meta, shapefile_path, title, output_path=None):
    """
    Plot a DEM with a shapefile boundary overlay.

    Parameters:
        dem_data (numpy array): DEM data.
        dem_meta (dict): Metadata of the DEM.
        shapefile_path (str): Path to the shapefile for boundary overlay.
        title (str): Title of the plot.
        output_path (str, optional): Path to save the plot. If None, the plot is shown.
    """
    # Load the shapefile
    gdf_boundary = gpd.read_file(shapefile_path)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    elevation_plot = ax.imshow(
        dem_data,
        cmap='terrain',
        extent=[
            dem_meta['transform'][2],
            dem_meta['transform'][2] + dem_meta['transform'][0] * dem_meta['width'],
            dem_meta['transform'][5] + dem_meta['transform'][4] * dem_meta['height'],
            dem_meta['transform'][5]
        ]
    )
    plt.colorbar(elevation_plot, label='Elevation (meters)', ax=ax)

    # Add the boundary overlay
    gdf_boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)

    # Configure title and labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def save_dem_to_netcdf(dem_data, lon, lat, output_path):
    """
    Save DEM data to a NetCDF file.

    Parameters:
        dem_data (numpy array): DEM data.
        lon (numpy array): Longitude values.
        lat (numpy array): Latitude values.
        output_path (str): Path to save the NetCDF file.
    """
    # Create an xarray DataArray
    dem_da = xr.DataArray(
        dem_data,
        coords={"latitude": lat, "longitude": lon},
        dims=["latitude", "longitude"],
        name="DEM"
    )

    # Add metadata
    dem_da.attrs["units"] = "meters"
    dem_da.attrs["description"] = "Digital Elevation Model (90m resolution)"

    # Save to NetCDF
    dem_da.to_netcdf(output_path)
    print(f"DEM data saved to NetCDF file: {output_path}")


def main():
    """
    Main function to load, plot, and export the DEM with the Boyacá boundary.
    """
    print("Loading DEM data...")
    dem_data, dem_meta, lon, lat = load_dem(DEM_PATH_90)
    print("DEM data loaded successfully!")
    print(f"DEM Metadata: {dem_meta}")
    print(f"DEM Shape: {dem_data.shape}")

    print("Plotting DEM with Boyacá boundary...")
    plot_dem_with_boundary(
        dem_data,
        dem_meta,
        SHAPEFILE_BOYACA,
        title="Elevation Map of Boyacá (90m Resolution)"
    )
    print("Plotting completed!")

    print("Saving DEM data to NetCDF...")
    save_dem_to_netcdf(dem_data, lon, lat, OUTPUT_NETCDF_PATH)
    print("DEM data saved successfully!")


if __name__ == "__main__":
    main()
