#!/usr/bin/env python3
"""
WorldCover Download Script for Sydney AOI
==========================================

This script downloads ESA WorldCover 2021 data for the Sydney area of interest.

WorldCover provides global land cover at 10m resolution with 11 classes:
- 10: Tree cover
- 20: Shrubland
- 30: Grassland
- 40: Cropland
- 50: Built-up
- 60: Bare/sparse vegetation
- 70: Snow and ice
- 80: Permanent water bodies
- 90: Herbaceous wetland
- 95: Mangroves
- 100: Moss and lichen

Requirements:
    pip install geopandas rasterio requests tqdm
"""

import os
import json
import requests
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
AOI_FILE = "/Users/timgotschim/Documents/LLM/infrared.city/aois_json/Sydney.geojson"
OUTPUT_DIR = "/Users/timgotschim/Documents/LLM/infrared.city/worldcover"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Sydney_WorldCover_2021.tif")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# WorldCover tile naming convention
# Sydney is approximately at: 151.2°E, -33.9°S
# This falls in tile: S30E150 (covers 30-40°S, 150-160°E)
WORLDCOVER_TILES = [
    "ESA_WorldCover_10m_2021_v200_S30_E150_Map.tif"  # Main tile covering Sydney
]

# WorldCover base URL
WORLDCOVER_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"


# ============================================================================
# FUNCTIONS
# ============================================================================

def download_worldcover_tile(tile_name, output_dir):
    """Download a single WorldCover tile."""
    url = f"{WORLDCOVER_BASE_URL}{tile_name}"
    output_path = os.path.join(output_dir, tile_name)
    
    if os.path.exists(output_path):
        print(f"✓ Tile already exists: {tile_name}")
        return output_path
    
    print(f"Downloading: {tile_name}")
    print(f"  URL: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=tile_name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✓ Downloaded: {tile_name}")
        return output_path
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading {tile_name}: {e}")
        return None


def clip_to_aoi(worldcover_tiles, aoi_file, output_file):
    """Clip WorldCover tiles to AOI."""
    print("\n" + "="*70)
    print("CLIPPING TO AOI")
    print("="*70)
    
    # Load AOI
    aoi = gpd.read_file(aoi_file)
    aoi = aoi.to_crs("EPSG:4326")
    
    print(f"AOI bounds: {aoi.total_bounds}")
    
    # If multiple tiles, merge them first
    if len(worldcover_tiles) > 1:
        print(f"Merging {len(worldcover_tiles)} tiles...")
        
        src_files = []
        for tile_path in worldcover_tiles:
            if tile_path and os.path.exists(tile_path):
                src_files.append(rasterio.open(tile_path))
        
        if not src_files:
            raise ValueError("No valid tiles found to merge")
        
        mosaic, out_transform = merge(src_files)
        
        # Close source files
        for src in src_files:
            src.close()
        
        # Create temporary merged file
        temp_merged = output_file.replace('.tif', '_temp_merged.tif')
        with rasterio.open(
            temp_merged, 'w',
            driver='GTiff',
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            count=1,
            dtype=mosaic.dtype,
            crs='EPSG:4326',
            transform=out_transform
        ) as dst:
            dst.write(mosaic[0], 1)
        
        clip_source = temp_merged
    else:
        clip_source = worldcover_tiles[0]
    
    # Clip to AOI
    print("Clipping to AOI geometry...")
    
    with rasterio.open(clip_source) as src:
        # Get geometries in the same CRS as the raster
        geometries = [feature["geometry"] for feature in aoi.__geo_interface__["features"]]
        
        # Clip
        out_image, out_transform = mask(src, geometries, crop=True)
        out_meta = src.meta.copy()
        
        # Update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw"
        })
        
        # Save clipped result
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(out_image)
    
    # Clean up temporary merged file if it exists
    if len(worldcover_tiles) > 1 and os.path.exists(temp_merged):
        os.remove(temp_merged)
    
    print(f"✓ Clipped WorldCover saved: {output_file}")
    print(f"  Dimensions: {out_image.shape[1]}x{out_image.shape[2]} pixels")
    
    return output_file


def analyze_worldcover(worldcover_file):
    """Analyze WorldCover classes in the clipped file."""
    print("\n" + "="*70)
    print("WORLDCOVER ANALYSIS")
    print("="*70)
    
    class_names = {
        10: "Tree cover",
        20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Bare/sparse vegetation",
        70: "Snow and ice",
        80: "Permanent water bodies",
        90: "Herbaceous wetland",
        95: "Mangroves",
        100: "Moss and lichen"
    }
    
    with rasterio.open(worldcover_file) as src:
        data = src.read(1)
    
    total_pixels = data.size
    unique, counts = np.unique(data, return_counts=True)
    
    print("\nLand Cover Distribution:")
    print("-" * 70)
    
    green_pixels = 0
    for value, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        class_name = class_names.get(value, f"Unknown ({value})")
        print(f"  {class_name:.<30} {count:>10,} pixels ({percentage:>5.2f}%)")
        
        # Count green classes
        if value in [10, 20, 30, 95]:
            green_pixels += count
    
    print("-" * 70)
    print(f"  {'Total Green (10,20,30,95)':.<30} {green_pixels:>10,} pixels ({100*green_pixels/total_pixels:>5.2f}%)")
    print(f"  {'Non-Green':.<30} {total_pixels-green_pixels:>10,} pixels ({100*(total_pixels-green_pixels)/total_pixels:>5.2f}%)")
    print("="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import numpy as np
    
    print("="*70)
    print("ESA WORLDCOVER 2021 DOWNLOAD FOR SYDNEY")
    print("="*70)
    print(f"\nAOI: {AOI_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Tiles to download: {len(WORLDCOVER_TILES)}")
    
    # Download tiles
    print("\n" + "="*70)
    print("DOWNLOADING WORLDCOVER TILES")
    print("="*70)
    
    downloaded_tiles = []
    for tile_name in WORLDCOVER_TILES:
        tile_path = download_worldcover_tile(tile_name, OUTPUT_DIR)
        if tile_path:
            downloaded_tiles.append(tile_path)
    
    if not downloaded_tiles:
        print("\n✗ No tiles were successfully downloaded!")
        exit(1)
    
    print(f"\n✓ Successfully downloaded {len(downloaded_tiles)}/{len(WORLDCOVER_TILES)} tiles")
    
    # Clip to AOI
    clipped_file = clip_to_aoi(downloaded_tiles, AOI_FILE, OUTPUT_FILE)
    
    # Analyze results
    analyze_worldcover(clipped_file)
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"\n✓ WorldCover data ready: {OUTPUT_FILE}")
    print("\nYou can now use this file in your notebook:")
    print(f'  worldcover_file = "{OUTPUT_FILE}"')
    print("\n" + "="*70)
