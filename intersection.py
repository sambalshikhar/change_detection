import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.plot import show
from shapely.geometry import box
import matplotlib.pyplot as plt

# Read the shapefile
shapefile_path = '/home/sambal/Desktop/Plaksha-CDA/stubble_burning/converted_shapefile.shp'
gdf = gpd.read_file(shapefile_path)

# Read the satellite image
satellite_image_path = '/home/sambal/Desktop/Plaksha-CDA/stubble_burning/planet_images/20221005_043430_02_2451_3B_AnalyticMS_SR_8b_clip.tif'
with rasterio.open(satellite_image_path) as src:
    image_bounds = src.bounds
    image_crs = src.crs

# Check CRS compatibility
if gdf.crs is None or not gdf.crs == CRS.from_dict(image_crs):
    gdf = gdf.to_crs(image_crs)

# Reset the index of the GeoDataFrame
gdf = gdf.reset_index(drop=True)

# Create a geometry from the image bounds
image_geometry = box(*image_bounds)

# Filter patches that intersect with the satellite image extent
intersects_patches = gdf[gdf.geometry.intersects(image_geometry)]

# Check for patches that completely overlap with the satellite image
completely_overlapping_patches = intersects_patches[intersects_patches.geometry.apply(lambda x: image_geometry.contains(x))]

# Get the count of completely overlapping patches
num_patches = len(completely_overlapping_patches)

# Print the number of completely overlapping patches
print("Number of patches completely overlapping with the satellite image:", num_patches)

# Plot the satellite image
with rasterio.open(satellite_image_path) as src:
    fig, ax = plt.subplots(figsize=(10, 10))
    rasterio.plot.show(src, ax=ax, cmap='gray')

# Plot the completely overlapping patches
completely_overlapping_patches.plot(ax=ax, color='red')
