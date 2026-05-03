import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import os

# Rutas de los datasets clusterizados
LOW_ELEVATION_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/ds_low_elevation.nc"
MEDIUM_ELEVATION_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/ds_medium_elevation.nc"
HIGH_ELEVATION_PATH = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/ds_high_elevation.nc"

# Ruta del shapefile de Boyacá
SHAPEFILE_BOYACA = "/Users/riperez/Conda/anaconda3/doc/precipitation/shapes/MGN_Departamento.shp"

# Ruta para guardar la imagen
OUTPUT_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "elevation_clusters_boyaca.png"
)

def plot_elevation_clusters(low_ds, medium_ds, high_ds, shapefile_path, output_image_path):
    """
    Plot the elevation clusters (low, medium, high) on a map of Boyacá and save the image.

    Parameters:
        low_ds (xarray.Dataset): Dataset for low elevation.
        medium_ds (xarray.Dataset): Dataset for medium elevation.
        high_ds (xarray.Dataset): Dataset for high elevation.
        shapefile_path (str): Path to the shapefile of Boyacá.
        output_image_path (str): Path to save the output image.
    """
    # Load the shapefile
    boyaca_boundary = gpd.read_file(shapefile_path)

    # Extract elevation data for plotting
    low_elevation = low_ds["elevation"]
    medium_elevation = medium_ds["elevation"]
    high_elevation = high_ds["elevation"]

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot the shapefile boundary
    boyaca_boundary.plot(ax=ax, color="none", edgecolor="black", linewidth=1)

    # Plot low elevation
    low_elevation.plot(ax=ax, cmap="Blues", alpha=0.6, label="Low Elevation")

    # Plot medium elevation
    medium_elevation.plot(ax=ax, cmap="Greens", alpha=0.6, label="Medium Elevation")

    # Plot high elevation
    high_elevation.plot(ax=ax, cmap="Reds", alpha=0.6, label="High Elevation")

    # Add legend and title
    plt.legend(["Boyacá Boundary", "Low Elevation", "Medium Elevation", "High Elevation"])
    plt.title("Elevation Clusters in Boyacá")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid()

    # Save the plot as an image
    plt.savefig(output_image_path, dpi=300)
    print(f"Image saved to: {output_image_path}")

    # Show the plot
    plt.show()

def main():
    """
    Main function to load datasets and plot elevation clusters.
    """
    print("Loading datasets...")
    low_ds = xr.open_dataset(LOW_ELEVATION_PATH)
    medium_ds = xr.open_dataset(MEDIUM_ELEVATION_PATH)
    high_ds = xr.open_dataset(HIGH_ELEVATION_PATH)
    print("Datasets loaded successfully!")

    print("Plotting elevation clusters...")
    plot_elevation_clusters(low_ds, medium_ds, high_ds, SHAPEFILE_BOYACA, OUTPUT_IMAGE_PATH)
    print("Plot completed!")

if __name__ == "__main__":
    main()
