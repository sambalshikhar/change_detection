import os
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask
from shapely.geometry import box, mapping
import matplotlib.pyplot as plt
import numpy as np
import colorsys
from .indices import *
from skimage.transform import resize


def extract_tiles_from_image(satellite_image_path, output_folder,shp_file):
    # Read the shapefile
    shapefile_path = shp_file
    gdf = gpd.read_file(shapefile_path)

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

        # Iterate over the completely overlapping patches
        for index, patch in completely_overlapping_patches.iterrows():
            # Extract the coordinates of the patch geometry
            patch_geometry = patch.geometry

            # Calculate the centroid of the patch
            patch_centroid = patch_geometry.centroid

            # Create a square tile of size 512x512 around the patch centroid
            tile_size = 1024
            tile_bounds = box(patch_centroid.x - tile_size/2, patch_centroid.y - tile_size/2, patch_centroid.x + tile_size/2, patch_centroid.y + tile_size/2)

            # Check if the tile is within the image bounds
            if image_geometry.contains(tile_bounds):
                # Mask the satellite image with the tile geometry
                masked_image, masked_transform = mask(src, [tile_bounds], crop=True)

                # Check if the tile has valid values
                if np.any(masked_image):
                    # Create a new profile for the output TIFF
                    tile_profile = src.profile.copy()
                    tile_profile.update(width=masked_image.shape[2], height=masked_image.shape[1], transform=masked_transform)

                    # Define the output filename for the tile
                    tile_filename = f"tile_{index}_{patch_centroid.x}_{patch_centroid.y}.tif"
                    tile_filepath = os.path.join(output_folder, tile_filename)

                    # Save the tile as a separate TIFF file
                    with rasterio.open(tile_filepath, 'w', **tile_profile) as dst:
                        dst.write(masked_image)
                        
                        # Add the intersecting patch polygon as a geometry attribute
                        patch_geometry_dict = mapping(patch_geometry)
                        dst.update_tags(1, **{'patch_geometry': patch_geometry_dict})

                    print(f"Saved tile {index}: {tile_filepath}")
            else:
                print(f"Skipping patch {index} as it exceeds image bounds")

def min_max_normalize(arr):
    arr=np.nan_to_num(arr)
    # Calculate the minimum and maximum values for each channel
    min_values = np.min(arr, axis=(0, 1))
    max_values = np.max(arr, axis=(0, 1))

    # Normalize the array channel-wise
    normalized_array = (arr - min_values) / (max_values - min_values)

    return normalized_array

def get_image(file_path,band_infomation):
    
    with rasterio.open(file_path) as src:
        # Read the RGB channels
        image_bounds = src.bounds
        image=src.read()

        ndvi_index=min_max_normalize(ndvi(src,band_infomation).index_array)
        bsi_index=min_max_normalize(bsi(src,band_infomation).index_array)
        bai_index=min_max_normalize(bai(src,band_infomation).index_array)
        char_index=min_max_normalize(char(src,band_infomation).index_array)
        sr_index=min_max_normalize(char(src,band_infomation).index_array)

        all_indices=[ndvi_index,bsi_index,bai_index,char_index,sr_index]

        normalized_image=min_max_normalize(image.transpose(1,2,0))

        # Convert the normalized data to uint8
        uint8_data = normalized_image

    return uint8_data,image_bounds,all_indices

def image_difference(image1, image2):
    # Ensure the images have the same shape
    assert image1.shape == image2.shape, "Images must have the same shape."

    # Calculate the difference between individual bands
    band_differences = image1 - image2

    # Calculate the mean squared difference for each band
    mean_squared_difference = band_differences**2

    # Calculate the square root of the mean squared difference
    difference_image = np.sqrt(mean_squared_difference)


    # Sum all the band-wise differences to generate a single band
    difference_band = np.sum(difference_image, axis=2)

    return difference_band



def rgb_to_hsv(rgb_image):
    # Normalize the RGB values to the range [0, 1]
    normalized_image = rgb_image.astype(np.float32) / 255.0

    # Convert the RGB image to HSV
    hsv_image = np.zeros_like(normalized_image)
    for i in range(rgb_image.shape[0]):
        for j in range(rgb_image.shape[1]):
            r, g, b = normalized_image[i, j]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            hsv_image[i, j] = h, s, v

    return hsv_image

def resize_image(image,factor):
    if len(image.shape)>2:
        h,w,c = image.shape
    else:
        h,w = image.shape 
        c=1

    new_size = h // factor
    new_size = int(new_size*5)
    image = resize(image, (new_size,new_size,c),preserve_range=True, anti_aliasing=True)
    new_size=np.array([new_size,new_size])
    return image,new_size

def get_knee_value(index_array):

    index_array=index_array.flatten()
    get_left_skews=index_array.round(3)
    get_left_skews=np.unique(get_left_skews)
    get_left_skews=sorted(get_left_skews)
    y=[i for i in range(len(get_left_skews))]
    kneedle = KneeLocator(y,get_left_skews,S=1.0, curve="concave", direction="increasing")
    knee_value=kneedle.knee
    stress_value =get_left_skews[knee_value]
    return stress_value


def min_max_normalize(arr):
    arr=np.nan_to_num(arr)
    # Calculate the minimum and maximum values for each channel
    min_values = np.min(arr, axis=(0, 1))
    max_values = np.max(arr, axis=(0, 1))

    # Normalize the array channel-wise
    normalized_array = (arr - min_values) / (max_values - min_values)

    return normalized_array

def get_cosine_sim(array1,array2):

    h,w,c=array1.shape
    chunk_size = 1   # Adjust the chunk size as needed
    array1_flat = np.reshape(array1, (h*w, c))
    array2_flat = np.reshape(array2, (h*w, c))

    # Normalize the arrays
    norm1 = np.linalg.norm(array1_flat, axis=1, keepdims=True)
    norm2 = np.linalg.norm(array2_flat, axis=1, keepdims=True)
    array1_normalized = array1_flat / norm1
    array2_normalized = array2_flat / norm2

    # Calculate the similarity map in chunks
    similarity_map = np.zeros((h, w)).flatten()  # Initialize similarity map
    num_chunks = (h*w) // chunk_size  # Calculate the number of chunks

    for i in range(num_chunks):
        # Calculate similarity for the current chunk
        chunk_similarity = np.dot(array1_normalized[i,:],array2_normalized[i,:].T)
        similarity_map[i]=chunk_similarity
    similarity_map=similarity_map.reshape(h,w)

    return similarity_map

def get_channel_sim(all_indices_1,all_indices_2):
    
    for i,indices in enumerate(all_indices_1):
        all_indices_1[i],_=resize_image(indices,5)
        
    for i,indices in enumerate(all_indices_2):
        all_indices_2[i],_=resize_image(indices,5)
           
    image_1_all_channels=all_indices_1
    image_2_all_channels=all_indices_2

    image_1_data=np.dstack(image_1_all_channels)
    image_1_data=np.nan_to_num(image_1_data)
    image_2_data=np.dstack(image_2_all_channels)
    image_2_data=np.nan_to_num(image_2_data)

    #image_1_data=min_max_normalize(image_1_data)
    #image_2_data=min_max_normalize(image_2_data)

    dot_product=get_cosine_sim(image_1_data,image_2_data)

    #knee_value=get_knee_value(dot_product)
    #print(knee_value,dot_product.flatten().mean())
    mean_value=dot_product.flatten().mean()

    sim_map=np.where(dot_product<mean_value,1,0)

    return sim_map,dot_product