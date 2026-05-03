import xarray as xr

# Ruta del archivo original
input_file = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/ds_combined_downscaled.nc"
output_file = "/Users/riperez/Conda/anaconda3/doc/precipitation/output/ds_combined_downscaled_sample.nc"

def generate_sample(input_path, output_path, lat_slice, lon_slice, time_slice):
    ds = xr.open_dataset(input_path)
    print("✅ Dataset original cargado con éxito.")

    # Mostrar rangos disponibles
    print(f"Rango de latitudes: {ds.latitude.min().item()} a {ds.latitude.max().item()}")
    print(f"Rango de longitudes: {ds.longitude.min().item()} a {ds.longitude.max().item()}")
    print(f"Rango de fechas: {ds.time.min().item()} a {ds.time.max().item()}")

    # Asegurar slicing correcto independientemente del orden
    lat_sorted = ds.latitude.values[0] < ds.latitude.values[-1]
    lon_sorted = ds.longitude.values[0] < ds.longitude.values[-1]

    lat_slice_final = slice(*sorted([lat_slice.start, lat_slice.stop])) if lat_sorted else slice(*sorted([lat_slice.stop, lat_slice.start])[::-1])
    lon_slice_final = slice(*sorted([lon_slice.start, lon_slice.stop])) if lon_sorted else slice(*sorted([lon_slice.stop, lon_slice.start])[::-1])

    # Selección
    ds_sample = ds.sel(latitude=lat_slice_final, longitude=lon_slice_final, time=time_slice)

    if ds_sample.sizes['latitude'] == 0 or ds_sample.sizes['longitude'] == 0 or ds_sample.sizes['time'] == 0:
        print("⚠️ La muestra está vacía. Verifica los rangos de selección.")
        return

    print("✅ Muestra generada con éxito.")
    print("Dimensiones:", ds_sample.sizes)
    print("Coordenadas:", list(ds_sample.coords))
    print("Variables:", list(ds_sample.data_vars))

    ds_sample.to_netcdf(output_path)
    print(f"💾 Muestra guardada en: {output_path}")

if __name__ == "__main__":
    latitude_range = slice(4.6, 4.7)
    longitude_range = slice(-72, -72.9)
    time_range = slice("2020-01-01", "2021-12-31")

    generate_sample(input_file, output_file, latitude_range, longitude_range, time_range)
