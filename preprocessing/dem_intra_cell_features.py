"""
DEM Intra-Cell Feature Engineering (Posibilidad 2 + Posibilidad 3)

Computes sub-grid topographic features from a high-resolution DEM (90m)
mapped to the coarser CHIRPS grid (0.05 deg, ~5.5 km). Each CHIRPS cell
contains approximately 3000-4000 DEM pixels.

Posibilidad 2 (Deciles):
    Elevation deciles (p10, p20, ..., p100) per CHIRPS cell, capturing
    intra-cell heterogeneity. A cell in a valley has concentrated deciles;
    a cell spanning a ridge has dispersed deciles.

Posibilidad 3 (PCA / UMAP):
    Dimensionality reduction on the raw DEM pixel vectors per cell.
    PCA captures linear structure; UMAP captures nonlinear topology.

Both produce STATIC features (time-invariant) that are broadcast across
all time steps when integrated into the dataset.

Usage:
    python preprocessing/dem_intra_cell_features.py --dem data/input/dem/boyaca_dem_90m.tif
    python preprocessing/dem_intra_cell_features.py --download-srtm
    python preprocessing/dem_intra_cell_features.py --dem-nc data/output/dem_boyaca_90.nc
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CHIRPS grid definition (Boyaca region)
CHIRPS_LAT_MIN = 4.375
CHIRPS_LAT_MAX = 7.375
CHIRPS_LON_MIN = -74.925
CHIRPS_LON_MAX = -71.725
CHIRPS_STEP = 0.05
CHIRPS_NLAT = 61
CHIRPS_NLON = 65


def load_dem_geotiff(dem_path):
    """Load a DEM GeoTIFF and return data array with lat/lon coordinates.

    Returns:
        dem_data: 2D numpy array (lat, lon)
        dem_lat: 1D array of latitude values (descending)
        dem_lon: 1D array of longitude values (ascending)
    """
    try:
        import rasterio
    except ImportError:
        logger.error("rasterio not installed. Install with: pip install rasterio")
        sys.exit(1)

    logger.info(f"Loading DEM GeoTIFF: {dem_path}")
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1).astype(np.float64)
        transform = src.transform
        width = src.width
        height = src.height

    dem_lon = np.array([transform[2] + i * transform[0] for i in range(width)])
    dem_lat = np.array([transform[5] + i * transform[4] for i in range(height)])

    # Replace nodata
    dem_data[dem_data < -1000] = np.nan
    dem_data[dem_data > 9000] = np.nan

    logger.info(f"DEM shape: {dem_data.shape}, lat: [{dem_lat[0]:.4f}, {dem_lat[-1]:.4f}], "
                f"lon: [{dem_lon[0]:.4f}, {dem_lon[-1]:.4f}]")
    logger.info(f"DEM range: [{np.nanmin(dem_data):.0f}, {np.nanmax(dem_data):.0f}] m")
    return dem_data, dem_lat, dem_lon


def load_dem_netcdf(dem_path):
    """Load a DEM NetCDF file (as produced by data/load/dem90m.py)."""
    logger.info(f"Loading DEM NetCDF: {dem_path}")
    ds = xr.open_dataset(dem_path)

    # Find the DEM variable
    dem_var = None
    for v in ds.data_vars:
        if 'dem' in v.lower() or 'elevation' in v.lower():
            dem_var = v
            break
    if dem_var is None:
        dem_var = list(ds.data_vars)[0]

    dem_data = ds[dem_var].values.astype(np.float64)
    dem_lat = ds['latitude'].values
    dem_lon = ds['longitude'].values

    dem_data[dem_data < -1000] = np.nan
    dem_data[dem_data > 9000] = np.nan

    logger.info(f"DEM shape: {dem_data.shape}, lat: [{dem_lat[0]:.4f}, {dem_lat[-1]:.4f}], "
                f"lon: [{dem_lon[0]:.4f}, {dem_lon[-1]:.4f}]")
    ds.close()
    return dem_data, dem_lat, dem_lon


def download_srtm_dem(output_path, lat_min=4.0, lat_max=7.5, lon_min=-75.0, lon_max=-71.5):
    """Download SRTM 90m DEM tiles for the Boyaca region.

    Uses SRTM GL3 (90m) tiles from OpenTopography or CGIAR-CSI.
    Merges tiles and clips to the region of interest.

    Returns path to the downloaded/merged GeoTIFF.
    """
    try:
        import rasterio
        from rasterio.merge import merge as rasterio_merge
        from rasterio.mask import mask as rasterio_mask
    except ImportError:
        logger.error("rasterio required for SRTM download. pip install rasterio")
        sys.exit(1)

    import urllib.request
    import zipfile
    import tempfile
    import math

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"SRTM DEM already exists: {output_path}")
        return str(output_path)

    # SRTM tiles needed (1x1 degree tiles, format: srtm_XX_YY)
    # CGIAR SRTM v4.1 tile numbering: col = (lon+180)/5 + 1, row = (60-lat)/5 + 1
    # For Boyaca (lat 4-8, lon -75 to -71):
    # Alternatively, use direct SRTM 1-degree HGT naming: N04W075, N05W075, etc.

    logger.info("Downloading SRTM 90m DEM tiles for Boyaca region...")
    logger.info(f"Region: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]")

    # SRTM HGT tile names (1-degree tiles)
    tiles_needed = []
    for lat_tile in range(math.floor(lat_min), math.ceil(lat_max)):
        for lon_tile in range(math.floor(lon_min), math.ceil(lon_max)):
            ns = 'N' if lat_tile >= 0 else 'S'
            ew = 'E' if lon_tile >= 0 else 'W'
            tile_name = f"{ns}{abs(lat_tile):02d}{ew}{abs(lon_tile):03d}"
            tiles_needed.append(tile_name)

    logger.info(f"SRTM tiles needed: {len(tiles_needed)} tiles")
    logger.info(f"Tiles: {', '.join(tiles_needed[:10])}...")

    # Download from NASA SRTM server (requires Earthdata login)
    # Fallback: CGIAR-CSI mirror or direct HGT files
    base_url = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/"

    temp_dir = Path(tempfile.mkdtemp(prefix='srtm_'))
    downloaded_files = []

    for tile in tiles_needed:
        # Try CGIAR 5x5 degree tiles first
        hgt_url = f"https://elevation-tiles-prod.s3.amazonaws.com/skadi/{tile[:3]}/{tile}.hgt.gz"

        hgt_path = temp_dir / f"{tile}.hgt.gz"
        try:
            logger.info(f"  Downloading {tile}...")
            urllib.request.urlretrieve(hgt_url, str(hgt_path))
            downloaded_files.append(hgt_path)
        except Exception as e:
            logger.warning(f"  Could not download {tile}: {e}")

    if not downloaded_files:
        logger.error("No SRTM tiles downloaded. Please provide DEM manually.")
        logger.error("Options:")
        logger.error("  1. Download from https://dwtkns.com/srtm30m/")
        logger.error("  2. Use --dem <path_to_geotiff>")
        logger.error("  3. Use --dem-nc <path_to_netcdf>")
        sys.exit(1)

    logger.info(f"Downloaded {len(downloaded_files)} tiles. Merging...")

    # Decompress and merge
    import gzip
    tif_files = []
    for gz_path in downloaded_files:
        hgt_path = gz_path.with_suffix('')
        with gzip.open(gz_path, 'rb') as f_in:
            with open(hgt_path, 'wb') as f_out:
                f_out.write(f_in.read())
        tif_files.append(hgt_path)

    # Merge tiles with rasterio
    src_files = [rasterio.open(f) for f in tif_files]
    merged, merged_transform = rasterio_merge(src_files)
    for s in src_files:
        s.close()

    # Save merged DEM
    profile = {
        'driver': 'GTiff',
        'dtype': merged.dtype,
        'width': merged.shape[2],
        'height': merged.shape[1],
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': merged_transform,
    }
    with rasterio.open(str(output_path), 'w', **profile) as dst:
        dst.write(merged)

    logger.info(f"SRTM DEM saved: {output_path} ({merged.shape[1]}x{merged.shape[2]})")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    return str(output_path)


def get_chirps_grid():
    """Return the CHIRPS grid coordinates."""
    chirps_lat = np.linspace(CHIRPS_LAT_MIN, CHIRPS_LAT_MAX, CHIRPS_NLAT)
    chirps_lon = np.linspace(CHIRPS_LON_MIN, CHIRPS_LON_MAX, CHIRPS_NLON)
    return chirps_lat, chirps_lon


def map_dem_to_chirps_cells(dem_data, dem_lat, dem_lon, chirps_lat, chirps_lon, chirps_step=0.05):
    """Map high-resolution DEM pixels to CHIRPS grid cells.

    For each CHIRPS cell (i, j), find all DEM pixels whose center falls
    within the cell boundaries [lat - step/2, lat + step/2) x [lon - step/2, lon + step/2).

    Returns:
        cell_pixels: dict mapping (i, j) -> 1D array of elevation values
    """
    logger.info("Mapping DEM pixels to CHIRPS cells...")
    half_step = chirps_step / 2.0

    # Ensure dem_lat is monotonic
    if dem_lat[0] > dem_lat[-1]:
        # Descending latitude (typical for rasterio)
        lat_ascending = False
    else:
        lat_ascending = True

    cell_pixels = {}
    total_mapped = 0
    total_cells = len(chirps_lat) * len(chirps_lon)

    for i, clat in enumerate(chirps_lat):
        lat_lo = clat - half_step
        lat_hi = clat + half_step

        # Find DEM rows within this latitude band
        if lat_ascending:
            lat_mask = (dem_lat >= lat_lo) & (dem_lat < lat_hi)
        else:
            lat_mask = (dem_lat > lat_lo) & (dem_lat <= lat_hi)

        lat_indices = np.where(lat_mask)[0]
        if len(lat_indices) == 0:
            for j in range(len(chirps_lon)):
                cell_pixels[(i, j)] = np.array([])
            continue

        for j, clon in enumerate(chirps_lon):
            lon_lo = clon - half_step
            lon_hi = clon + half_step

            lon_mask = (dem_lon >= lon_lo) & (dem_lon < lon_hi)
            lon_indices = np.where(lon_mask)[0]

            if len(lon_indices) == 0:
                cell_pixels[(i, j)] = np.array([])
                continue

            # Extract DEM values for this cell
            dem_block = dem_data[np.ix_(lat_indices, lon_indices)]
            values = dem_block.flatten()
            values = values[~np.isnan(values)]
            cell_pixels[(i, j)] = values
            total_mapped += len(values)

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i+1}/{len(chirps_lat)} latitude rows...")

    # Statistics
    pixel_counts = [len(v) for v in cell_pixels.values()]
    non_empty = sum(1 for c in pixel_counts if c > 0)
    logger.info(f"Mapping complete: {total_mapped:,} DEM pixels -> {non_empty}/{total_cells} cells")
    if pixel_counts:
        logger.info(f"Pixels per cell: min={min(c for c in pixel_counts if c > 0)}, "
                     f"max={max(pixel_counts)}, "
                     f"mean={np.mean([c for c in pixel_counts if c > 0]):.0f}")

    return cell_pixels


# ============================================================================
# POSIBILIDAD 2: Elevation deciles and statistics
# ============================================================================

def compute_elevation_deciles(cell_pixels, chirps_lat, chirps_lon, min_pixels=10):
    """Compute elevation deciles per CHIRPS cell (Posibilidad 2).

    For each cell, compute the 10th, 20th, ..., 100th percentiles of the
    DEM pixel elevations within that cell.

    Args:
        cell_pixels: dict (i,j) -> array of elevation values
        chirps_lat: 1D array of CHIRPS latitudes
        chirps_lon: 1D array of CHIRPS longitudes
        min_pixels: minimum DEM pixels required for valid statistics

    Returns:
        deciles: array (61, 65, 10) with percentiles p10-p100
    """
    logger.info("Computing elevation deciles (Posibilidad 2)...")
    nlat = len(chirps_lat)
    nlon = len(chirps_lon)
    deciles = np.full((nlat, nlon, 10), np.nan)
    percentile_levels = np.arange(10, 101, 10)  # [10, 20, ..., 100]

    for i in range(nlat):
        for j in range(nlon):
            values = cell_pixels.get((i, j), np.array([]))
            if len(values) >= min_pixels:
                deciles[i, j, :] = np.percentile(values, percentile_levels)

    valid_cells = np.sum(~np.isnan(deciles[:, :, 0]))
    logger.info(f"Deciles computed: {valid_cells}/{nlat*nlon} cells with valid data")
    return deciles, percentile_levels


def compute_elevation_statistics(cell_pixels, chirps_lat, chirps_lon, min_pixels=10):
    """Compute elevation summary statistics per CHIRPS cell.

    Statistics: mean, std, skewness, kurtosis, range (max - min)

    Returns:
        stats: array (61, 65, 5)
        stat_names: list of statistic names
    """
    from scipy import stats as scipy_stats

    logger.info("Computing elevation statistics...")
    nlat = len(chirps_lat)
    nlon = len(chirps_lon)
    result = np.full((nlat, nlon, 5), np.nan)
    stat_names = ['mean', 'std', 'skewness', 'kurtosis', 'range']

    for i in range(nlat):
        for j in range(nlon):
            values = cell_pixels.get((i, j), np.array([]))
            if len(values) >= min_pixels:
                result[i, j, 0] = np.mean(values)
                result[i, j, 1] = np.std(values)
                result[i, j, 2] = scipy_stats.skew(values)
                result[i, j, 3] = scipy_stats.kurtosis(values)
                result[i, j, 4] = np.max(values) - np.min(values)

    valid_cells = np.sum(~np.isnan(result[:, :, 0]))
    logger.info(f"Statistics computed: {valid_cells} valid cells")
    logger.info(f"  Mean elevation: [{np.nanmin(result[:,:,0]):.0f}, {np.nanmax(result[:,:,0]):.0f}] m")
    logger.info(f"  Std elevation: [{np.nanmin(result[:,:,1]):.0f}, {np.nanmax(result[:,:,1]):.0f}] m")
    logger.info(f"  Range: [{np.nanmin(result[:,:,4]):.0f}, {np.nanmax(result[:,:,4]):.0f}] m")
    return result, stat_names


# ============================================================================
# POSIBILIDAD 3: PCA / UMAP dimensionality reduction
# ============================================================================

def build_pixel_matrix(cell_pixels, chirps_lat, chirps_lon, target_npixels=None, min_pixels=10):
    """Build a standardized matrix of DEM pixels per cell for PCA/UMAP.

    Since cells may have different numbers of DEM pixels, we pad/truncate
    to a uniform length (target_npixels). Missing values are filled with
    the cell's mean elevation.

    Args:
        cell_pixels: dict (i,j) -> array of elevation values
        target_npixels: number of pixels per cell (default: median count)
        min_pixels: minimum pixels for a cell to be included

    Returns:
        matrix: (n_valid_cells, target_npixels) array
        cell_indices: list of (i, j) tuples for valid cells
    """
    logger.info("Building pixel matrix for dimensionality reduction...")

    # Determine target size
    counts = [len(v) for v in cell_pixels.values() if len(v) >= min_pixels]
    if target_npixels is None:
        target_npixels = int(np.median(counts))
    logger.info(f"Target pixels per cell: {target_npixels}")

    nlat = len(chirps_lat)
    nlon = len(chirps_lon)
    rows = []
    cell_indices = []

    for i in range(nlat):
        for j in range(nlon):
            values = cell_pixels.get((i, j), np.array([]))
            if len(values) < min_pixels:
                continue

            # Sort values (makes the representation order-invariant in elevation space)
            sorted_vals = np.sort(values)

            # Resample to target length using linear interpolation
            if len(sorted_vals) == target_npixels:
                row = sorted_vals
            else:
                x_orig = np.linspace(0, 1, len(sorted_vals))
                x_target = np.linspace(0, 1, target_npixels)
                row = np.interp(x_target, x_orig, sorted_vals)

            rows.append(row)
            cell_indices.append((i, j))

    matrix = np.array(rows)
    logger.info(f"Pixel matrix: {matrix.shape} ({len(cell_indices)} cells x {target_npixels} features)")
    return matrix, cell_indices


def compute_pca_features(matrix, cell_indices, chirps_lat, chirps_lon, n_components=6):
    """Apply PCA to the pixel matrix (Posibilidad 3 - linear).

    Returns:
        pca_features: array (61, 65, n_components)
        explained_variance: array of explained variance ratios
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    logger.info(f"Computing PCA with {n_components} components...")

    # Standardize
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)

    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(matrix_scaled)

    # Map back to grid
    nlat = len(chirps_lat)
    nlon = len(chirps_lon)
    pca_features = np.full((nlat, nlon, n_components), np.nan)
    for idx, (i, j) in enumerate(cell_indices):
        pca_features[i, j, :] = transformed[idx]

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    logger.info(f"PCA explained variance: {explained}")
    logger.info(f"PCA cumulative variance: {cumulative}")
    logger.info(f"Total variance explained by {n_components} components: {cumulative[-1]:.3f}")

    return pca_features, explained


def compute_umap_features(matrix, cell_indices, chirps_lat, chirps_lon, n_components=6):
    """Apply UMAP to the pixel matrix (Posibilidad 3 - nonlinear).

    Returns:
        umap_features: array (61, 65, n_components)
    """
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed. Skipping UMAP. Install with: pip install umap-learn")
        return None

    from sklearn.preprocessing import StandardScaler

    logger.info(f"Computing UMAP with {n_components} components...")

    # Standardize
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)

    # UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42,
    )
    transformed = reducer.fit_transform(matrix_scaled)

    # Map back to grid
    nlat = len(chirps_lat)
    nlon = len(chirps_lon)
    umap_features = np.full((nlat, nlon, n_components), np.nan)
    for idx, (i, j) in enumerate(cell_indices):
        umap_features[i, j, :] = transformed[idx]

    logger.info(f"UMAP features computed: {umap_features.shape}")
    return umap_features


# ============================================================================
# Integration: save features to NetCDF
# ============================================================================

def save_features_netcdf(output_path, chirps_lat, chirps_lon,
                         deciles=None, percentile_levels=None,
                         stats=None, stat_names=None,
                         pca_features=None, pca_variance=None,
                         umap_features=None):
    """Save all computed features to a single NetCDF file."""
    logger.info(f"Saving features to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(coords={
        'latitude': chirps_lat,
        'longitude': chirps_lon,
    })

    # Posibilidad 2: Deciles
    if deciles is not None:
        percentile_coord = percentile_levels if percentile_levels is not None else np.arange(10, 101, 10)
        ds['elevation_deciles'] = xr.DataArray(
            deciles,
            dims=['latitude', 'longitude', 'percentile'],
            coords={'percentile': percentile_coord},
            attrs={
                'description': 'Elevation deciles (p10-p100) from intra-cell DEM pixels',
                'units': 'meters',
                'source': 'SRTM 90m DEM',
            }
        )

    # Posibilidad 2: Statistics
    if stats is not None:
        names = stat_names or ['mean', 'std', 'skewness', 'kurtosis', 'range']
        for k, name in enumerate(names):
            ds[f'dem_{name}'] = xr.DataArray(
                stats[:, :, k],
                dims=['latitude', 'longitude'],
                attrs={
                    'description': f'Intra-cell DEM {name}',
                    'units': 'meters' if name != 'skewness' and name != 'kurtosis' else 'dimensionless',
                }
            )

    # Posibilidad 3: PCA
    if pca_features is not None:
        n_comp = pca_features.shape[2]
        ds['pca_features'] = xr.DataArray(
            pca_features,
            dims=['latitude', 'longitude', 'pca_component'],
            coords={'pca_component': np.arange(1, n_comp + 1)},
            attrs={
                'description': 'PCA components of intra-cell DEM pixel distributions',
                'method': 'PCA on sorted, interpolated DEM pixels per cell',
            }
        )
        if pca_variance is not None:
            ds['pca_explained_variance'] = xr.DataArray(
                pca_variance,
                dims=['pca_component'],
                coords={'pca_component': np.arange(1, len(pca_variance) + 1)},
                attrs={'description': 'Explained variance ratio per PCA component'},
            )

    # Posibilidad 3: UMAP
    if umap_features is not None:
        n_comp = umap_features.shape[2]
        ds['umap_features'] = xr.DataArray(
            umap_features,
            dims=['latitude', 'longitude', 'umap_component'],
            coords={'umap_component': np.arange(1, n_comp + 1)},
            attrs={
                'description': 'UMAP components of intra-cell DEM pixel distributions',
                'method': 'UMAP on sorted, interpolated DEM pixels per cell',
            }
        )

    ds.to_netcdf(str(output_path))
    logger.info(f"Features saved: {list(ds.data_vars)}")
    ds.close()


def integrate_into_dataset(features_path, dataset_path, output_path):
    """Integrate static DEM features into the main time-series dataset.

    Static features are broadcast across all time steps.
    """
    logger.info(f"Integrating features into dataset...")
    logger.info(f"  Features: {features_path}")
    logger.info(f"  Dataset: {dataset_path}")

    features = xr.open_dataset(str(features_path))
    dataset = xr.open_dataset(str(dataset_path))

    time_dim = 'time' if 'time' in dataset.dims else 'month_index'
    n_time = dataset.sizes[time_dim]

    # Add decile features (expand to time dimension)
    if 'elevation_deciles' in features:
        deciles = features['elevation_deciles'].values  # (61, 65, 10)
        percentile_levels = features['percentile'].values
        for k, pct in enumerate(percentile_levels):
            var_name = f'dem_p{int(pct):02d}'
            # Broadcast static feature across time
            data_3d = np.broadcast_to(
                deciles[:, :, k][np.newaxis, :, :],
                (n_time, deciles.shape[0], deciles.shape[1])
            ).copy()
            dataset[var_name] = xr.DataArray(
                data_3d,
                dims=[time_dim, 'latitude', 'longitude'],
                attrs={'units': 'meters', 'description': f'Intra-cell elevation percentile {int(pct)}'}
            )

    # Add statistics
    for stat_name in ['mean', 'std', 'skewness', 'kurtosis', 'range']:
        feat_var = f'dem_{stat_name}'
        if feat_var in features:
            data_2d = features[feat_var].values
            data_3d = np.broadcast_to(
                data_2d[np.newaxis, :, :],
                (n_time, data_2d.shape[0], data_2d.shape[1])
            ).copy()
            dataset[feat_var] = xr.DataArray(
                data_3d,
                dims=[time_dim, 'latitude', 'longitude'],
                attrs=features[feat_var].attrs,
            )

    # Add PCA features
    if 'pca_features' in features:
        pca = features['pca_features'].values  # (61, 65, n_comp)
        n_comp = pca.shape[2]
        for k in range(n_comp):
            var_name = f'dem_pca_{k+1}'
            data_3d = np.broadcast_to(
                pca[:, :, k][np.newaxis, :, :],
                (n_time, pca.shape[0], pca.shape[1])
            ).copy()
            dataset[var_name] = xr.DataArray(
                data_3d,
                dims=[time_dim, 'latitude', 'longitude'],
                attrs={'description': f'DEM PCA component {k+1}'}
            )

    # Add UMAP features
    if 'umap_features' in features:
        umap_data = features['umap_features'].values
        n_comp = umap_data.shape[2]
        for k in range(n_comp):
            var_name = f'dem_umap_{k+1}'
            data_3d = np.broadcast_to(
                umap_data[:, :, k][np.newaxis, :, :],
                (n_time, umap_data.shape[0], umap_data.shape[1])
            ).copy()
            dataset[var_name] = xr.DataArray(
                data_3d,
                dims=[time_dim, 'latitude', 'longitude'],
                attrs={'description': f'DEM UMAP component {k+1}'}
            )

    logger.info(f"Saving extended dataset to {output_path}...")
    dataset.to_netcdf(str(output_path))
    logger.info(f"Done. New variables: {len(dataset.data_vars)} total")

    features.close()
    dataset.close()


# ============================================================================
# Visualization
# ============================================================================

def generate_diagnostic_figures(features_path, output_dir):
    """Generate diagnostic figures for DEM intra-cell features."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    logger.info("Generating diagnostic figures...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = xr.open_dataset(str(features_path))
    chirps_lat = features['latitude'].values
    chirps_lon = features['longitude'].values

    # Color scheme (Okabe-Ito)
    colors = {
        'blue': '#0072B2',
        'orange': '#E69F00',
        'green': '#009E73',
        'red': '#D55E00',
        'purple': '#CC79A7',
    }

    # --- Figure 1: Heterogeneity map (std of elevation) ---
    if 'dem_std' in features:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax, var, title, cmap in zip(
            axes,
            ['dem_mean', 'dem_std', 'dem_range'],
            ['Mean Elevation (m)', 'Elevation Std (m)', 'Elevation Range (m)'],
            ['terrain', 'YlOrRd', 'YlOrRd']
        ):
            if var in features:
                data = features[var].values
                im = ax.pcolormesh(chirps_lon, chirps_lat, data, cmap=cmap, shading='auto')
                plt.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title(title, fontsize=11, fontfamily='Arial')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')

        plt.tight_layout()
        fig_path = output_dir / 'dem_heterogeneity_maps.pdf'
        plt.savefig(str(fig_path), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {fig_path}")

    # --- Figure 2: Decile profiles by elevation cluster ---
    if 'elevation_deciles' in features:
        deciles = features['elevation_deciles'].values  # (61, 65, 10)
        mean_elev = features['dem_mean'].values if 'dem_mean' in features else deciles[:, :, 4]

        # Classify cells
        low_mask = mean_elev < 1500
        mid_mask = (mean_elev >= 1500) & (mean_elev < 2500)
        high_mask = mean_elev >= 2500

        fig, ax = plt.subplots(figsize=(8, 5))
        pcts = np.arange(10, 101, 10)

        for mask, label, color in [
            (low_mask, 'Low (<1500m)', colors['green']),
            (mid_mask, 'Medium (1500-2500m)', colors['orange']),
            (high_mask, 'High (>2500m)', colors['red']),
        ]:
            valid = ~np.isnan(mean_elev) & mask
            if valid.sum() == 0:
                continue
            cluster_deciles = deciles[valid]  # (n_cells, 10)
            median_profile = np.median(cluster_deciles, axis=0)
            q25 = np.percentile(cluster_deciles, 25, axis=0)
            q75 = np.percentile(cluster_deciles, 75, axis=0)

            ax.plot(pcts, median_profile, '-o', color=color, label=label, markersize=4)
            ax.fill_between(pcts, q25, q75, color=color, alpha=0.15)

        ax.set_xlabel('Percentile', fontfamily='Arial')
        ax.set_ylabel('Elevation (m)', fontfamily='Arial')
        ax.set_title('Elevation Decile Profiles by Cluster', fontfamily='Arial')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig_path = output_dir / 'dem_decile_profiles.pdf'
        plt.savefig(str(fig_path), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {fig_path}")

    # --- Figure 3: PCA explained variance ---
    if 'pca_explained_variance' in features:
        variance = features['pca_explained_variance'].values
        cumulative = np.cumsum(variance)

        fig, ax = plt.subplots(figsize=(7, 4))
        components = np.arange(1, len(variance) + 1)
        ax.bar(components, variance, color=colors['blue'], alpha=0.7, label='Individual')
        ax.plot(components, cumulative, 'o-', color=colors['red'], label='Cumulative')
        ax.set_xlabel('PCA Component', fontfamily='Arial')
        ax.set_ylabel('Explained Variance Ratio', fontfamily='Arial')
        ax.set_title('PCA Explained Variance', fontfamily='Arial')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(components)

        fig_path = output_dir / 'dem_pca_variance.pdf'
        plt.savefig(str(fig_path), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {fig_path}")

    # --- Figure 4: PCA spatial maps (first 3 components) ---
    if 'pca_features' in features:
        pca = features['pca_features'].values
        n_show = min(3, pca.shape[2])

        fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show + 1, 5))
        if n_show == 1:
            axes = [axes]

        for k in range(n_show):
            im = axes[k].pcolormesh(chirps_lon, chirps_lat, pca[:, :, k],
                                    cmap='RdBu_r', shading='auto')
            plt.colorbar(im, ax=axes[k], shrink=0.8)
            axes[k].set_title(f'PCA Component {k+1}', fontfamily='Arial')
            axes[k].set_xlabel('Longitude')
            axes[k].set_ylabel('Latitude')

        plt.tight_layout()
        fig_path = output_dir / 'dem_pca_spatial.pdf'
        plt.savefig(str(fig_path), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {fig_path}")

    features.close()
    logger.info("Diagnostic figures complete.")


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DEM Intra-Cell Feature Engineering (Pos2 + Pos3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From local GeoTIFF
  python preprocessing/dem_intra_cell_features.py --dem data/input/dem/boyaca_dem_90m.tif

  # From local NetCDF (as produced by data/load/dem90m.py)
  python preprocessing/dem_intra_cell_features.py --dem-nc data/output/dem_boyaca_90.nc

  # Download SRTM tiles automatically
  python preprocessing/dem_intra_cell_features.py --download-srtm

  # Only Pos2 (deciles), skip Pos3
  python preprocessing/dem_intra_cell_features.py --dem my_dem.tif --skip-pos3

  # Integrate into existing dataset
  python preprocessing/dem_intra_cell_features.py --dem my_dem.tif --integrate
        """
    )
    parser.add_argument('--dem', type=str, default=None,
                        help='Path to DEM GeoTIFF (90m resolution)')
    parser.add_argument('--dem-nc', type=str, default=None,
                        help='Path to DEM NetCDF (as produced by data/load/dem90m.py)')
    parser.add_argument('--download-srtm', action='store_true',
                        help='Download SRTM 90m tiles for the region')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for features NetCDF')
    parser.add_argument('--n-pca', type=int, default=6,
                        help='Number of PCA components (default: 6)')
    parser.add_argument('--n-umap', type=int, default=6,
                        help='Number of UMAP components (default: 6)')
    parser.add_argument('--skip-pos3', action='store_true',
                        help='Skip Pos3 (PCA/UMAP)')
    parser.add_argument('--skip-umap', action='store_true',
                        help='Skip UMAP (PCA only)')
    parser.add_argument('--integrate', action='store_true',
                        help='Integrate features into main dataset')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset path for --integrate')
    parser.add_argument('--figures', action='store_true',
                        help='Generate diagnostic figures')
    parser.add_argument('--figures-dir', type=str, default=None,
                        help='Output directory for figures')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  DEM INTRA-CELL FEATURE ENGINEERING")
    logger.info("  Posibilidad 2 (Deciles) + Posibilidad 3 (PCA/UMAP)")
    logger.info("=" * 60)

    # Load DEM
    if args.dem:
        dem_data, dem_lat, dem_lon = load_dem_geotiff(args.dem)
    elif args.dem_nc:
        dem_data, dem_lat, dem_lon = load_dem_netcdf(args.dem_nc)
    elif args.download_srtm:
        srtm_path = PROJECT_ROOT / 'data' / 'input' / 'dem' / 'boyaca_dem_90m.tif'
        download_srtm_dem(srtm_path)
        dem_data, dem_lat, dem_lon = load_dem_geotiff(str(srtm_path))
    else:
        # Try default paths
        default_tif = PROJECT_ROOT / 'data' / 'input' / 'dem' / 'boyaca_dem_90m.tif'
        default_nc = PROJECT_ROOT / 'data' / 'output' / 'dem_boyaca_90.nc'
        if default_tif.exists():
            dem_data, dem_lat, dem_lon = load_dem_geotiff(str(default_tif))
        elif default_nc.exists():
            dem_data, dem_lat, dem_lon = load_dem_netcdf(str(default_nc))
        else:
            logger.error("No DEM file found. Provide one of:")
            logger.error("  --dem <path.tif>       GeoTIFF file")
            logger.error("  --dem-nc <path.nc>     NetCDF file")
            logger.error("  --download-srtm        Download SRTM tiles")
            logger.error(f"  Or place DEM at: {default_tif}")
            sys.exit(1)

    # Get CHIRPS grid
    chirps_lat, chirps_lon = get_chirps_grid()

    # Map DEM pixels to CHIRPS cells
    cell_pixels = map_dem_to_chirps_cells(dem_data, dem_lat, dem_lon, chirps_lat, chirps_lon)

    # Free DEM array memory
    del dem_data

    # --- Posibilidad 2: Deciles + Statistics ---
    deciles, pct_levels = compute_elevation_deciles(cell_pixels, chirps_lat, chirps_lon)
    stats, stat_names = compute_elevation_statistics(cell_pixels, chirps_lat, chirps_lon)

    # --- Posibilidad 3: PCA / UMAP ---
    pca_features = None
    pca_variance = None
    umap_features = None

    if not args.skip_pos3:
        matrix, cell_indices = build_pixel_matrix(cell_pixels, chirps_lat, chirps_lon)

        # PCA
        pca_features, pca_variance = compute_pca_features(
            matrix, cell_indices, chirps_lat, chirps_lon, n_components=args.n_pca)

        # UMAP
        if not args.skip_umap:
            umap_features = compute_umap_features(
                matrix, cell_indices, chirps_lat, chirps_lon, n_components=args.n_umap)

    # Save features
    output_path = args.output or str(PROJECT_ROOT / 'data' / 'output' / 'dem_intra_cell_features.nc')
    save_features_netcdf(
        output_path, chirps_lat, chirps_lon,
        deciles=deciles, percentile_levels=pct_levels,
        stats=stats, stat_names=stat_names,
        pca_features=pca_features, pca_variance=pca_variance,
        umap_features=umap_features,
    )

    # Generate figures
    if args.figures:
        figures_dir = args.figures_dir or str(PROJECT_ROOT / 'scripts' / 'benchmark' / 'output' / 'figures')
        generate_diagnostic_figures(output_path, figures_dir)

    # Integrate into dataset
    if args.integrate:
        dataset_path = args.dataset or str(
            PROJECT_ROOT / 'data' / 'output' /
            'complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc'
        )
        ext_output = str(Path(dataset_path).parent / 'complete_dataset_extended_dem_features.nc')
        integrate_into_dataset(output_path, dataset_path, ext_output)

    logger.info("=" * 60)
    logger.info("  FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Features: {output_path}")
    if args.integrate:
        logger.info(f"  Extended dataset: {ext_output}")


if __name__ == "__main__":
    main()
