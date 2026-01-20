#!/usr/bin/env python3
"""
Sentinel-2 Processing Web Application
======================================

A user-friendly web interface for processing Sentinel-2 data with drag & drop.

Features:
- Drag & drop Sentinel-2 folders and GeoJSON files
- Automatic clipping to AOI
- Calculates NDVI, EVI, SAVI
- Downloads multi-band GeoTIFF stack

Installation:
    pip install flask rasterio rioxarray xarray geopandas numpy werkzeug

Usage:
    python sentinel2_web_app.py
    
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, send_file, jsonify
import os
import tempfile
import shutil
import zipfile
import glob
import numpy as np
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
from datetime import datetime
from pathlib import Path
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max upload
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentinel-2 Processor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        
        .drop-zone {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 60px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8f9ff;
            margin-bottom: 20px;
        }
        
        .drop-zone:hover {
            border-color: #764ba2;
            background: #f0f1ff;
        }
        
        .drop-zone.dragover {
            border-color: #4CAF50;
            background: #e8f5e9;
            transform: scale(1.02);
        }
        
        .drop-zone.uploaded {
            border-color: #4CAF50;
            background: #e8f5e9;
        }
        
        .drop-zone i {
            font-size: 3em;
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .drop-zone h3 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .drop-zone p {
            color: #666;
            font-size: 0.95em;
        }
        
        .file-input {
            display: none;
        }
        
        .uploaded-file {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            display: none;
        }
        
        .uploaded-file.show {
            display: block;
        }
        
        .uploaded-file h4 {
            color: #2e7d32;
            margin-bottom: 5px;
        }
        
        .uploaded-file p {
            color: #555;
            font-size: 0.9em;
        }
        
        .process-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
            transition: all 0.3s ease;
        }
        
        .process-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .process-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .progress-container {
            display: none;
            margin-top: 20px;
        }
        
        .progress-container.show {
            display: block;
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            width: 0%;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .status-message {
            margin-top: 15px;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .status-message.info {
            background: #e3f2fd;
            color: #1976d2;
        }
        
        .status-message.success {
            background: #e8f5e9;
            color: #2e7d32;
        }
        
        .status-message.error {
            background: #ffebee;
            color: #c62828;
        }
        
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .feature {
            text-align: center;
            padding: 20px;
        }
        
        .feature i {
            font-size: 2.5em;
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .feature h3 {
            color: #333;
            margin-bottom: 5px;
        }
        
        .feature p {
            color: #666;
            font-size: 0.9em;
        }
        
        .download-btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            border-radius: 8px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            margin-top: 20px;
            transition: all 0.3s ease;
        }
        
        .download-btn:hover {
            background: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(76, 175, 80, 0.4);
        }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛰️ Sentinel-2 Processor</h1>
            <p>Drag & Drop your Sentinel-2 data and AOI to create multi-band stacks</p>
        </div>
        
        <div class="card">
            <h2 style="margin-bottom: 20px;">📁 Upload Files</h2>
            
            <!-- Sentinel-2 Drop Zone -->
            <div class="drop-zone" id="sentinel-drop-zone">
                <i class="fas fa-satellite"></i>
                <h3>Drop Sentinel-2 Folder Here</h3>
                <p>Or click to browse (supports .zip or folder with .jp2 files)</p>
                <input type="file" id="sentinel-input" class="file-input" webkitdirectory directory multiple>
            </div>
            <div class="uploaded-file" id="sentinel-uploaded">
                <h4>✓ Sentinel-2 Data Uploaded</h4>
                <p id="sentinel-info"></p>
            </div>
            
            <!-- GeoJSON Drop Zone -->
            <div class="drop-zone" id="geojson-drop-zone">
                <i class="fas fa-map-marked-alt"></i>
                <h3>Drop GeoJSON (AOI) Here</h3>
                <p>Or click to browse (.geojson file)</p>
                <input type="file" id="geojson-input" class="file-input" accept=".geojson,.json">
            </div>
            <div class="uploaded-file" id="geojson-uploaded">
                <h4>✓ GeoJSON Uploaded</h4>
                <p id="geojson-info"></p>
            </div>
            
            <!-- Process Button -->
            <button class="process-btn" id="process-btn" disabled>
                <i class="fas fa-cogs"></i> Process Data
            </button>
            
            <!-- Progress -->
            <div class="progress-container" id="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill">0%</div>
                </div>
                <div class="status-message info" id="status-message"></div>
            </div>
        </div>
        
        <div class="card">
            <h2 style="margin-bottom: 20px;">✨ Features</h2>
            <div class="features">
                <div class="feature">
                    <i class="fas fa-layer-group"></i>
                    <h3>Multi-Band Stack</h3>
                    <p>Combines all spectral bands into one file</p>
                </div>
                <div class="feature">
                    <i class="fas fa-leaf"></i>
                    <h3>Vegetation Indices</h3>
                    <p>Automatically calculates NDVI, EVI, SAVI</p>
                </div>
                <div class="feature">
                    <i class="fas fa-cut"></i>
                    <h3>Auto Clipping</h3>
                    <p>Clips to your exact area of interest</p>
                </div>
                <div class="feature">
                    <i class="fas fa-download"></i>
                    <h3>Ready to Use</h3>
                    <p>Downloads as GeoTIFF for immediate use</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let sentinelFiles = null;
        let geojsonFile = null;
        
        // Sentinel-2 Drop Zone
        const sentinelZone = document.getElementById('sentinel-drop-zone');
        const sentinelInput = document.getElementById('sentinel-input');
        const sentinelUploaded = document.getElementById('sentinel-uploaded');
        const sentinelInfo = document.getElementById('sentinel-info');
        
        // GeoJSON Drop Zone
        const geojsonZone = document.getElementById('geojson-drop-zone');
        const geojsonInput = document.getElementById('geojson-input');
        const geojsonUploaded = document.getElementById('geojson-uploaded');
        const geojsonInfo = document.getElementById('geojson-info');
        
        // Process Button
        const processBtn = document.getElementById('process-btn');
        const progressContainer = document.getElementById('progress-container');
        const progressFill = document.getElementById('progress-fill');
        const statusMessage = document.getElementById('status-message');
        
        // Setup Sentinel-2 drop zone
        setupDropZone(sentinelZone, sentinelInput, (files) => {
            sentinelFiles = files;
            sentinelZone.classList.add('uploaded');
            sentinelUploaded.classList.add('show');
            sentinelInfo.textContent = `${files.length} files selected`;
            checkReadyToProcess();
        });
        
        // Setup GeoJSON drop zone
        setupDropZone(geojsonZone, geojsonInput, (files) => {
            geojsonFile = files[0];
            geojsonZone.classList.add('uploaded');
            geojsonUploaded.classList.add('show');
            geojsonInfo.textContent = geojsonFile.name;
            checkReadyToProcess();
        });
        
        function setupDropZone(zone, input, callback) {
            zone.addEventListener('click', () => input.click());
            
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('dragover');
            });
            
            zone.addEventListener('dragleave', () => {
                zone.classList.remove('dragover');
            });
            
            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
                
                const files = Array.from(e.dataTransfer.files);
                callback(files);
            });
            
            input.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                callback(files);
            });
        }
        
        function checkReadyToProcess() {
            if (sentinelFiles && geojsonFile) {
                processBtn.disabled = false;
            }
        }
        
        processBtn.addEventListener('click', async () => {
            processBtn.disabled = true;
            progressContainer.classList.add('show');
            
            const formData = new FormData();
            
            // Add Sentinel-2 files
            for (let file of sentinelFiles) {
                formData.append('sentinel_files', file);
            }
            
            // Add GeoJSON
            formData.append('geojson', geojsonFile);
            
            try {
                updateProgress(10, 'Uploading files...');
                
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });
                
                updateProgress(50, 'Processing Sentinel-2 data...');
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Processing failed');
                }
                
                updateProgress(90, 'Finalizing...');
                
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'Sentinel2_Stack.tif';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                updateProgress(100, 'Download complete!', 'success');
                
            } catch (error) {
                updateProgress(0, `Error: ${error.message}`, 'error');
                processBtn.disabled = false;
            }
        });
        
        function updateProgress(percent, message, type = 'info') {
            progressFill.style.width = percent + '%';
            progressFill.textContent = percent + '%';
            statusMessage.textContent = message;
            statusMessage.className = 'status-message ' + type;
        }
    </script>
</body>
</html>
"""


def process_sentinel2_data(sentinel_files, geojson_file, output_file):
    """
    Process Sentinel-2 data: clip to AOI and calculate vegetation indices.
    """
    # Load GeoJSON (AOI)
    aoi = gpd.read_file(geojson_file)
    if aoi.crs is None:
        aoi.set_crs("EPSG:4326", inplace=True)
    if aoi.crs.to_epsg() != 4326:
        aoi = aoi.to_crs("EPSG:4326")
    
    geometries = [aoi.geometry.iloc[0]]
    
    # Find and process bands
    bands_to_load = ["B02", "B03", "B04", "B08"]
    band_arrays = []
    band_names = []
    
    # Load spectral bands
    for band_name in bands_to_load:
        # Find file matching band name
        band_file = None
        for file in sentinel_files:
            if band_name in file and file.endswith('.jp2'):
                band_file = file
                break
        
        if not band_file:
            continue
        
        # Load and clip
        band = rxr.open_rasterio(band_file, masked=True).squeeze()
        band_clipped = band.rio.clip(geometries, crs="EPSG:4326")
        
        band_arrays.append(band_clipped)
        band_names.append(band_name)
    
    # Calculate vegetation indices
    if len(band_arrays) >= 4:
        # Get band data
        blue = band_arrays[0].astype(np.float32)   # B02
        green = band_arrays[1].astype(np.float32)  # B03
        red = band_arrays[2].astype(np.float32)    # B04
        nir = band_arrays[3].astype(np.float32)    # B08
        
        # NDVI
        ndvi = (nir - red) / (nir + red)
        ndvi = xr.where(np.isfinite(ndvi), ndvi, np.nan)
        band_arrays.append(ndvi)
        band_names.append("NDVI")
        
        # EVI
        evi = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)
        evi = xr.where(np.isfinite(evi), evi, np.nan)
        band_arrays.append(evi)
        band_names.append("EVI")
        
        # SAVI
        L = 0.5
        savi = ((nir - red) * (1 + L)) / (nir + red + L)
        savi = xr.where(np.isfinite(savi), savi, np.nan)
        band_arrays.append(savi)
        band_names.append("SAVI")
    
    # Stack all bands
    stack = xr.concat(band_arrays, dim="band")
    stack = stack.assign_coords(band=band_names)
    stack = stack.astype(np.float32)
    
    # Save
    stack.rio.to_raster(output_file, dtype=np.float32)
    
    return output_file


@app.route('/')
def index():
    return HTML_TEMPLATE


@app.route('/process', methods=['POST'])
def process():
    try:
        # Get uploaded files
        sentinel_files = request.files.getlist('sentinel_files')
        geojson_file = request.files['geojson']
        
        if not sentinel_files or not geojson_file:
            return jsonify({'error': 'Missing files'}), 400
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Save Sentinel-2 files
        sentinel_paths = []
        for file in sentinel_files:
            if file.filename.endswith('.jp2'):
                path = os.path.join(temp_dir, file.filename)
                file.save(path)
                sentinel_paths.append(path)
        
        # Save GeoJSON
        geojson_path = os.path.join(temp_dir, 'aoi.geojson')
        geojson_file.save(geojson_path)
        
        # Process
        output_file = os.path.join(temp_dir, 'Sentinel2_Stack.tif')
        process_sentinel2_data(sentinel_paths, geojson_path, output_file)
        
        # Send file
        return send_file(
            output_file,
            as_attachment=True,
            download_name=f'Sentinel2_Stack_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tif'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    print("="*80)
    print("🛰️  SENTINEL-2 WEB PROCESSOR")
    print("="*80)
    print("\n✓ Starting web server...")
    print("\n📡 Open in browser: http://localhost:5001")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("="*80)
    
    app.run(debug=True, port=5001, host='0.0.0.0')
