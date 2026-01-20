#!/usr/bin/env python3
"""
Sentinel-2 Multi-Month Processing Web Application
==================================================

A user-friendly web interface for processing multi-month Sentinel-2 data.

Features:
- Drag & drop Sentinel-2 folders for multiple months
- Automatic clipping to AOI
- Calculates NDVI, EVI, SAVI for each month
- Downloads 21-band multi-month GeoTIFF stack

Installation:
    pip install flask rasterio rioxarray xarray geopandas numpy werkzeug

Usage:
    python sentinel2_multimonth_app.py
    
Then open: http://localhost:5002
"""

from flask import Flask, render_template, request, send_file, jsonify
import os
import tempfile
import shutil
import glob
import numpy as np
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB max upload
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# HTML Template with Multi-Month Support
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentinel-2 Multi-Month Processor</title>
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
            max-width: 1400px;
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
        
        .header .subtitle {
            font-size: 0.9em;
            margin-top: 10px;
            opacity: 0.8;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        
        .month-selector {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .month-option {
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
        }
        
        .month-option:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        
        .month-option.selected {
            border-color: #667eea;
            background: #667eea;
            color: white;
        }
        
        .month-option input[type="checkbox"] {
            margin-right: 8px;
        }
        
        .drop-zones-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .month-upload-section {
            display: none;
        }
        
        .month-upload-section.show {
            display: block;
        }
        
        .drop-zone {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8f9ff;
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
            font-size: 2.5em;
            color: #667eea;
            margin-bottom: 10px;
            display: block;
        }
        
        .drop-zone h3 {
            color: #333;
            margin-bottom: 8px;
            font-size: 1.1em;
        }
        
        .drop-zone p {
            color: #666;
            font-size: 0.85em;
        }
        
        .file-input {
            display: none;
        }
        
        .uploaded-file {
            background: #e8f5e9;
            padding: 12px;
            border-radius: 8px;
            margin-top: 10px;
            display: none;
            font-size: 0.9em;
        }
        
        .uploaded-file.show {
            display: block;
        }
        
        .uploaded-file h4 {
            color: #2e7d32;
            margin-bottom: 3px;
            font-size: 0.95em;
        }
        
        .uploaded-file p {
            color: #555;
            font-size: 0.85em;
        }
        
        .geojson-section {
            margin-top: 30px;
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
        
        .info-box {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .info-box h3 {
            color: #e65100;
            margin-bottom: 8px;
        }
        
        .info-box p {
            color: #555;
            font-size: 0.95em;
            line-height: 1.6;
        }
        
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛰️ Sentinel-2 Multi-Month Processor</h1>
            <p>Create multi-temporal stacks with vegetation indices</p>
            <p class="subtitle">Upload data for multiple months to create a 21-band stack</p>
        </div>
        
        <div class="card">
            <div class="info-box">
                <h3>📅 How it works</h3>
                <p>
                    <strong>1.</strong> Select which months you want to process (minimum 1, maximum 3)<br>
                    <strong>2.</strong> Upload Sentinel-2 data for each selected month<br>
                    <strong>3.</strong> Upload your GeoJSON (AOI)<br>
                    <strong>4.</strong> Click Process to create your multi-band stack!
                </p>
            </div>
            
            <h2 style="margin-bottom: 20px;">📅 Select Months</h2>
            <div class="month-selector">
                <div class="month-option" data-month="april">
                    <input type="checkbox" id="month-april">
                    <label for="month-april">🌸 April</label>
                </div>
                <div class="month-option" data-month="august">
                    <input type="checkbox" id="month-august">
                    <label for="month-august">☀️ August</label>
                </div>
                <div class="month-option" data-month="november">
                    <input type="checkbox" id="month-november">
                    <label for="month-november">🍂 November</label>
                </div>
            </div>
            
            <h2 style="margin-bottom: 20px;">📁 Upload Sentinel-2 Data</h2>
            <div class="drop-zones-container">
                <!-- April -->
                <div class="month-upload-section" id="april-section">
                    <div class="drop-zone" data-month="april">
                        <i class="fas fa-satellite"></i>
                        <h3>April Data</h3>
                        <p>Drop folder or click to browse</p>
                        <input type="file" class="file-input" data-month="april" webkitdirectory directory multiple>
                    </div>
                    <div class="uploaded-file" id="april-uploaded">
                        <h4>✓ April Data Uploaded</h4>
                        <p id="april-info"></p>
                    </div>
                </div>
                
                <!-- August -->
                <div class="month-upload-section" id="august-section">
                    <div class="drop-zone" data-month="august">
                        <i class="fas fa-satellite"></i>
                        <h3>August Data</h3>
                        <p>Drop folder or click to browse</p>
                        <input type="file" class="file-input" data-month="august" webkitdirectory directory multiple>
                    </div>
                    <div class="uploaded-file" id="august-uploaded">
                        <h4>✓ August Data Uploaded</h4>
                        <p id="august-info"></p>
                    </div>
                </div>
                
                <!-- November -->
                <div class="month-upload-section" id="november-section">
                    <div class="drop-zone" data-month="november">
                        <i class="fas fa-satellite"></i>
                        <h3>November Data</h3>
                        <p>Drop folder or click to browse</p>
                        <input type="file" class="file-input" data-month="november" webkitdirectory directory multiple>
                    </div>
                    <div class="uploaded-file" id="november-uploaded">
                        <h4>✓ November Data Uploaded</h4>
                        <p id="november-info"></p>
                    </div>
                </div>
            </div>
            
            <div class="geojson-section">
                <h2 style="margin-bottom: 20px;">🗺️ Upload GeoJSON (AOI)</h2>
                <div class="drop-zone" id="geojson-drop-zone">
                    <i class="fas fa-map-marked-alt"></i>
                    <h3>Drop GeoJSON Here</h3>
                    <p>Or click to browse (.geojson file)</p>
                    <input type="file" id="geojson-input" class="file-input" accept=".geojson,.json">
                </div>
                <div class="uploaded-file" id="geojson-uploaded">
                    <h4>✓ GeoJSON Uploaded</h4>
                    <p id="geojson-info"></p>
                </div>
            </div>
            
            <button class="process-btn" id="process-btn" disabled>
                <i class="fas fa-cogs"></i> Process Multi-Month Data
            </button>
            
            <div class="progress-container" id="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill">0%</div>
                </div>
                <div class="status-message info" id="status-message"></div>
            </div>
        </div>
        
        <div class="card">
            <h2 style="margin-bottom: 20px;">📦 Output Structure</h2>
            <div class="info-box" style="background: #e3f2fd; border-left-color: #2196f3;">
                <h3 style="color: #1565c0;">21-Band Multi-Month Stack</h3>
                <p style="font-family: monospace; font-size: 0.9em; line-height: 2;">
                    Band 1-7: April (B02, B03, B04, B08, NDVI, EVI, SAVI)<br>
                    Band 8-14: August (B02, B03, B04, B08, NDVI, EVI, SAVI)<br>
                    Band 15-21: November (B02, B03, B04, B08, NDVI, EVI, SAVI)
                </p>
            </div>
            
            <h2 style="margin-bottom: 20px; margin-top: 30px;">✨ Features</h2>
            <div class="features">
                <div class="feature">
                    <i class="fas fa-calendar-alt"></i>
                    <h3>Multi-Temporal</h3>
                    <p>Process 1-3 months of data</p>
                </div>
                <div class="feature">
                    <i class="fas fa-layer-group"></i>
                    <h3>21 Bands</h3>
                    <p>7 bands per month</p>
                </div>
                <div class="feature">
                    <i class="fas fa-leaf"></i>
                    <h3>Vegetation Indices</h3>
                    <p>NDVI, EVI, SAVI per month</p>
                </div>
                <div class="feature">
                    <i class="fas fa-cut"></i>
                    <h3>Auto Clipping</h3>
                    <p>Clips to your AOI</p>
                </div>
                <div class="feature">
                    <i class="fas fa-download"></i>
                    <h3>Ready to Use</h3>
                    <p>GeoTIFF output</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const monthData = {
            april: null,
            august: null,
            november: null
        };
        let geojsonFile = null;
        let selectedMonths = new Set();
        
        // Month selection
        document.querySelectorAll('.month-option').forEach(option => {
            option.addEventListener('click', function() {
                const month = this.dataset.month;
                const checkbox = this.querySelector('input[type="checkbox"]');
                checkbox.checked = !checkbox.checked;
                
                if (checkbox.checked) {
                    this.classList.add('selected');
                    selectedMonths.add(month);
                    document.getElementById(`${month}-section`).classList.add('show');
                } else {
                    this.classList.remove('selected');
                    selectedMonths.delete(month);
                    document.getElementById(`${month}-section`).classList.remove('show');
                    monthData[month] = null;
                    document.getElementById(`${month}-uploaded`).classList.remove('show');
                }
                
                checkReadyToProcess();
            });
        });
        
        // Setup drop zones for each month
        ['april', 'august', 'november'].forEach(month => {
            const zone = document.querySelector(`.drop-zone[data-month="${month}"]`);
            const input = document.querySelector(`.file-input[data-month="${month}"]`);
            const uploaded = document.getElementById(`${month}-uploaded`);
            const info = document.getElementById(`${month}-info`);
            
            setupDropZone(zone, input, (files) => {
                monthData[month] = files;
                zone.classList.add('uploaded');
                uploaded.classList.add('show');
                info.textContent = `${files.length} files selected`;
                checkReadyToProcess();
            });
        });
        
        // Setup GeoJSON drop zone
        const geojsonZone = document.getElementById('geojson-drop-zone');
        const geojsonInput = document.getElementById('geojson-input');
        const geojsonUploaded = document.getElementById('geojson-uploaded');
        const geojsonInfo = document.getElementById('geojson-info');
        
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
            const processBtn = document.getElementById('process-btn');
            
            // Check if at least one month is selected and uploaded
            let hasData = false;
            for (const month of selectedMonths) {
                if (monthData[month]) {
                    hasData = true;
                    break;
                }
            }
            
            // Enable button if we have at least one month of data and a GeoJSON
            if (hasData && geojsonFile) {
                processBtn.disabled = false;
            } else {
                processBtn.disabled = true;
            }
        }
        
        document.getElementById('process-btn').addEventListener('click', async () => {
            const processBtn = document.getElementById('process-btn');
            const progressContainer = document.getElementById('progress-container');
            
            processBtn.disabled = true;
            progressContainer.classList.add('show');
            
            const formData = new FormData();
            
            // Add files for each selected month
            for (const month of selectedMonths) {
                if (monthData[month]) {
                    for (let file of monthData[month]) {
                        formData.append(`${month}_files`, file);
                    }
                }
            }
            
            // Add GeoJSON
            formData.append('geojson', geojsonFile);
            
            // Add selected months list
            formData.append('months', Array.from(selectedMonths).join(','));
            
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
                
                updateProgress(90, 'Creating multi-band stack...');
                
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'Sentinel2_MultiMonth_Stack.tif';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                updateProgress(100, `✓ Download complete! Created ${selectedMonths.size}-month stack with ${selectedMonths.size * 7} bands`, 'success');
                
            } catch (error) {
                updateProgress(0, `Error: ${error.message}`, 'error');
                processBtn.disabled = false;
            }
        });
        
        function updateProgress(percent, message, type = 'info') {
            const progressFill = document.getElementById('progress-fill');
            const statusMessage = document.getElementById('status-message');
            
            progressFill.style.width = percent + '%';
            progressFill.textContent = percent + '%';
            statusMessage.textContent = message;
            statusMessage.className = 'status-message ' + type;
        }
    </script>
</body>
</html>
"""


def process_multimonth_sentinel2(month_files_dict, geojson_file, output_file):
    """
    Process multi-month Sentinel-2 data.
    
    month_files_dict: dict with keys like 'april', 'august', 'november'
                     and values as lists of file paths
    """
    # Load GeoJSON (AOI)
    aoi = gpd.read_file(geojson_file)
    if aoi.crs is None:
        aoi.set_crs("EPSG:4326", inplace=True)
    if aoi.crs.to_epsg() != 4326:
        aoi = aoi.to_crs("EPSG:4326")
    
    geometries = [aoi.geometry.iloc[0]]
    
    # Storage for all bands
    all_band_arrays = []
    all_band_names = []
    
    # Month name mapping
    month_names = {
        'april': 'Apr',
        'august': 'Aug',
        'november': 'Nov'
    }
    
    # Process each month
    for month_key in ['april', 'august', 'november']:
        if month_key not in month_files_dict:
            continue
        
        month_files = month_files_dict[month_key]
        month_name = month_names[month_key]
        
        # Find and load spectral bands
        bands_to_load = ["B02", "B03", "B04", "B08"]
        month_bands = {}
        
        for band_name in bands_to_load:
            # Find file matching band name
            band_file = None
            for file in month_files:
                if band_name in os.path.basename(file) and file.endswith('.jp2'):
                    band_file = file
                    break
            
            if not band_file:
                continue
            
            # Load and clip
            band = rxr.open_rasterio(band_file, masked=True).squeeze()
            band_clipped = band.rio.clip(geometries, crs="EPSG:4326")
            
            all_band_arrays.append(band_clipped)
            all_band_names.append(f"{band_name}-{month_name}")
            month_bands[band_name] = band_clipped
        
        # Calculate vegetation indices for this month
        if len(month_bands) >= 4:
            blue = month_bands["B02"].astype(np.float32)
            green = month_bands["B03"].astype(np.float32)
            red = month_bands["B04"].astype(np.float32)
            nir = month_bands["B08"].astype(np.float32)
            
            # NDVI
            ndvi = (nir - red) / (nir + red)
            ndvi = xr.where(np.isfinite(ndvi), ndvi, np.nan)
            all_band_arrays.append(ndvi)
            all_band_names.append(f"NDVI-{month_name}")
            
            # EVI
            evi = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)
            evi = xr.where(np.isfinite(evi), evi, np.nan)
            all_band_arrays.append(evi)
            all_band_names.append(f"EVI-{month_name}")
            
            # SAVI
            L = 0.5
            savi = ((nir - red) * (1 + L)) / (nir + red + L)
            savi = xr.where(np.isfinite(savi), savi, np.nan)
            all_band_arrays.append(savi)
            all_band_names.append(f"SAVI-{month_name}")
    
    # Stack all bands
    stack = xr.concat(all_band_arrays, dim="band")
    stack = stack.assign_coords(band=all_band_names)
    stack = stack.astype(np.float32)
    
    # Save
    stack.rio.to_raster(output_file, dtype=np.float32)
    
    return output_file, len(all_band_names)


@app.route('/')
def index():
    return HTML_TEMPLATE


@app.route('/process', methods=['POST'])
def process():
    temp_dir = None
    try:
        # Get selected months
        selected_months = request.form.get('months', '').split(',')
        
        # Get GeoJSON
        geojson_file = request.files.get('geojson')
        if not geojson_file:
            return jsonify({'error': 'Missing GeoJSON file'}), 400
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Save GeoJSON
        geojson_path = os.path.join(temp_dir, 'aoi.geojson')
        geojson_file.save(geojson_path)
        
        # Collect files for each month
        month_files_dict = {}
        
        for month in selected_months:
            if not month:
                continue
            
            files = request.files.getlist(f'{month}_files')
            if not files:
                continue
            
            # Save files
            month_dir = os.path.join(temp_dir, month)
            os.makedirs(month_dir, exist_ok=True)
            
            month_paths = []
            for file in files:
                if file.filename.endswith('.jp2'):
                    path = os.path.join(month_dir, file.filename)
                    file.save(path)
                    month_paths.append(path)
            
            if month_paths:
                month_files_dict[month] = month_paths
        
        if not month_files_dict:
            return jsonify({'error': 'No valid Sentinel-2 files uploaded'}), 400
        
        # Process
        output_file = os.path.join(temp_dir, 'Sentinel2_MultiMonth_Stack.tif')
        result_file, num_bands = process_multimonth_sentinel2(
            month_files_dict, 
            geojson_path, 
            output_file
        )
        
        # Send file
        return send_file(
            result_file,
            as_attachment=True,
            download_name=f'Sentinel2_{len(month_files_dict)}Month_{num_bands}Band_Stack_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tif'
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    print("="*80)
    print("🛰️  SENTINEL-2 MULTI-MONTH WEB PROCESSOR")
    print("="*80)
    print("\n✓ Starting web server...")
    print("\n📡 Open in browser: http://localhost:5002")
    print("\n✨ Features:")
    print("  - Process 1-3 months of Sentinel-2 data")
    print("  - 7 bands per month (4 spectral + 3 indices)")
    print("  - Creates 7-21 band multi-temporal stack")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("="*80)
    
    app.run(debug=True, port=5002, host='0.0.0.0')
