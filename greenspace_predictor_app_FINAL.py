#!/usr/bin/env python3
"""
Green Space Prediction Web Application
=======================================

Upload Sentinel-2 multi-month stacks and get instant green space predictions
using a pre-trained Random Forest model.

Features:
- Drag & drop Sentinel-2 multi-month stack (21 bands)
- Instant green space prediction
- Interactive visualization
- Downloadable results (prediction map + visualization)

Installation:
    pip install flask rasterio numpy scikit-learn matplotlib pillow werkzeug joblib

Usage:
    1. Train a model first (or load existing model)
    2. python greenspace_predictor_app.py
    3. Open: http://localhost:5004
"""

from flask import Flask, render_template, request, send_file, jsonify
import os
import tempfile
import shutil
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from datetime import datetime
import joblib

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# Global model variable (will be loaded or created)
TRAINED_MODEL = None
MODEL_PATH = "/Users/tyomachka/Desktop/WU/Data_Lab.TMP/rep.infrared.city/random_forest_model.pkl"

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Green Space Predictor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
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
            font-size: 2.8em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header .icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.95;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 35px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 25px;
        }
        
        .model-status {
            display: flex;
            align-items: center;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            font-weight: 500;
        }
        
        .model-status.loaded {
            background: #e8f5e9;
            color: #2e7d32;
            border-left: 5px solid #4caf50;
        }
        
        .model-status.not-loaded {
            background: #fff3e0;
            color: #e65100;
            border-left: 5px solid #ff9800;
        }
        
        .model-status i {
            font-size: 2em;
            margin-right: 15px;
        }
        
        .drop-zone {
            border: 3px dashed #4caf50;
            border-radius: 12px;
            padding: 60px 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f1f8e9;
            margin-bottom: 25px;
        }
        
        .drop-zone:hover {
            border-color: #2e7d32;
            background: #dcedc8;
            transform: scale(1.01);
        }
        
        .drop-zone.dragover {
            border-color: #1b5e20;
            background: #c5e1a5;
            transform: scale(1.03);
        }
        
        .drop-zone.uploaded {
            border-color: #4caf50;
            background: #e8f5e9;
        }
        
        .drop-zone i {
            font-size: 4em;
            color: #4caf50;
            margin-bottom: 20px;
        }
        
        .drop-zone h3 {
            color: #1b5e20;
            margin-bottom: 12px;
            font-size: 1.4em;
        }
        
        .drop-zone p {
            color: #558b2f;
            font-size: 1em;
        }
        
        .file-input {
            display: none;
        }
        
        .uploaded-file {
            background: #e8f5e9;
            padding: 18px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }
        
        .uploaded-file.show {
            display: block;
        }
        
        .uploaded-file h4 {
            color: #2e7d32;
            margin-bottom: 8px;
            font-size: 1.1em;
        }
        
        .uploaded-file p {
            color: #555;
            font-size: 0.95em;
        }
        
        .predict-btn {
            background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
            color: white;
            border: none;
            padding: 18px 45px;
            font-size: 1.2em;
            border-radius: 10px;
            cursor: pointer;
            width: 100%;
            margin-top: 25px;
            transition: all 0.3s ease;
            font-weight: 600;
        }
        
        .predict-btn:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
        }
        
        .predict-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .progress-container {
            display: none;
            margin-top: 25px;
        }
        
        .progress-container.show {
            display: block;
        }
        
        .progress-bar {
            width: 100%;
            height: 35px;
            background: #e0e0e0;
            border-radius: 17px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
            width: 0%;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 1em;
        }
        
        .status-message {
            margin-top: 18px;
            padding: 18px;
            border-radius: 10px;
            text-align: center;
            font-weight: 500;
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
        
        .results-container {
            display: none;
            margin-top: 30px;
        }
        
        .results-container.show {
            display: block;
        }
        
        .visualization {
            margin-top: 25px;
        }
        
        .visualization img {
            width: 100%;
            border-radius: 12px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }
        
        .stat-card {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-card h3 {
            color: #2e7d32;
            font-size: 2em;
            margin-bottom: 8px;
        }
        
        .stat-card p {
            color: #666;
            font-size: 0.95em;
        }
        
        .download-btn {
            background: #2196f3;
            color: white;
            border: none;
            padding: 15px 35px;
            font-size: 1.05em;
            border-radius: 8px;
            cursor: pointer;
            margin: 10px 5px;
            transition: all 0.3s ease;
        }
        
        .download-btn:hover {
            background: #1976d2;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(33, 150, 243, 0.4);
        }
        
        .info-box {
            background: #fff3e0;
            border-left: 5px solid #ff9800;
            padding: 18px;
            border-radius: 8px;
            margin-bottom: 25px;
        }
        
        .info-box h3 {
            color: #e65100;
            margin-bottom: 10px;
        }
        
        .info-box ul {
            color: #555;
            line-height: 1.8;
            margin-left: 20px;
        }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="icon">🌳</div>
            <h1>Green Space Predictor</h1>
            <p>Upload Sentinel-2 data and get instant green space predictions</p>
        </div>
        
        <div class="card">
            <div class="model-status {{ 'loaded' if model_loaded else 'not-loaded' }}">
                <i class="fas fa-{{ 'check-circle' if model_loaded else 'exclamation-triangle' }}"></i>
                <div>
                    {% if model_loaded %}
                    <strong>✓ Model Loaded</strong>
                    <p style="margin-top: 5px; font-weight: normal;">
                        Trained on {{ model_info.n_cities }} cities with {{ model_info.accuracy }}% accuracy
                    </p>
                    {% else %}
                    <strong>⚠ Demo Mode</strong>
                    <p style="margin-top: 5px; font-weight: normal;">
                        Using demo model. Train your own model for better results.
                    </p>
                    {% endif %}
                </div>
            </div>
            
            <div class="info-box">
                <h3>📋 Requirements</h3>
                <ul>
                    <li><strong>Input:</strong> Sentinel-2 multi-month stack (.tif)</li>
                    <li><strong>Bands:</strong> 21 bands (7 per month × 3 months)</li>
                    <li><strong>Format:</strong> GeoTIFF with spectral bands + vegetation indices</li>
                </ul>
            </div>
            
            <h2 style="margin-bottom: 20px;">📤 Upload Sentinel-2 Stack</h2>
            
            <div class="drop-zone" id="sentinel-drop-zone">
                <i class="fas fa-cloud-upload-alt"></i>
                <h3>Drop Sentinel-2 Stack Here</h3>
                <p>Or click to browse (21-band GeoTIFF file)</p>
                <input type="file" id="sentinel-input" class="file-input" accept=".tif,.tiff">
            </div>
            
            <div class="uploaded-file" id="sentinel-uploaded">
                <h4>✓ Sentinel-2 Stack Uploaded</h4>
                <p id="sentinel-info"></p>
            </div>
            
            <button class="predict-btn" id="predict-btn" disabled>
                <i class="fas fa-magic"></i> Predict Green Spaces
            </button>
            
            <div class="progress-container" id="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill">0%</div>
                </div>
                <div class="status-message info" id="status-message"></div>
            </div>
            
            <div class="results-container" id="results-container">
                <h2 style="margin-bottom: 20px;">📊 Results</h2>
                
                <div class="stats-grid" id="stats-grid"></div>
                
                <div class="visualization">
                    <img id="result-image" src="" alt="Green Space Prediction">
                </div>
                
                <div style="text-align: center; margin-top: 25px;">
                    <button class="download-btn" id="download-prediction">
                        <i class="fas fa-download"></i> Download Prediction Map
                    </button>
                    <button class="download-btn" id="download-visualization">
                        <i class="fas fa-image"></i> Download Visualization
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let sentinelFile = null;
        let predictionData = null;
        
        const sentinelZone = document.getElementById('sentinel-drop-zone');
        const sentinelInput = document.getElementById('sentinel-input');
        const sentinelUploaded = document.getElementById('sentinel-uploaded');
        const sentinelInfo = document.getElementById('sentinel-info');
        const predictBtn = document.getElementById('predict-btn');
        const progressContainer = document.getElementById('progress-container');
        const progressFill = document.getElementById('progress-fill');
        const statusMessage = document.getElementById('status-message');
        const resultsContainer = document.getElementById('results-container');
        
        // Setup drop zone
        setupDropZone(sentinelZone, sentinelInput, (file) => {
            sentinelFile = file;
            sentinelZone.classList.add('uploaded');
            sentinelUploaded.classList.add('show');
            sentinelInfo.textContent = `${file.name} (${(file.size / (1024*1024)).toFixed(2)} MB)`;
            predictBtn.disabled = false;
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
                if (e.dataTransfer.files.length > 0) {
                    callback(e.dataTransfer.files[0]);
                }
            });
            
            input.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    callback(e.target.files[0]);
                }
            });
        }
        
        predictBtn.addEventListener('click', async () => {
            predictBtn.disabled = true;
            progressContainer.classList.add('show');
            resultsContainer.classList.remove('show');
            
            const formData = new FormData();
            formData.append('sentinel_file', sentinelFile);
            
            try {
                updateProgress(10, 'Uploading Sentinel-2 data...');
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                updateProgress(50, 'Running prediction model...');
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Prediction failed');
                }
                
                updateProgress(80, 'Generating visualization...');
                
                const result = await response.json();
                predictionData = result;
                
                // Display results
                displayResults(result);
                
                updateProgress(100, '✓ Prediction complete!', 'success');
                resultsContainer.classList.add('show');
                
            } catch (error) {
                updateProgress(0, `Error: ${error.message}`, 'error');
                predictBtn.disabled = false;
            }
        });
        
        function displayResults(result) {
            // Update stats
            const statsGrid = document.getElementById('stats-grid');
            statsGrid.innerHTML = `
                <div class="stat-card">
                    <h3>${result.stats.green_percentage.toFixed(1)}%</h3>
                    <p>Green Coverage</p>
                </div>
                <div class="stat-card">
                    <h3>${result.stats.green_pixels.toLocaleString()}</h3>
                    <p>Green Pixels</p>
                </div>
                <div class="stat-card">
                    <h3>${result.stats.total_pixels.toLocaleString()}</h3>
                    <p>Total Pixels</p>
                </div>
                <div class="stat-card">
                    <h3>${result.stats.dimensions}</h3>
                    <p>Image Size</p>
                </div>
                <div class="stat-card">
                    <h3>${result.stats.n_bands || 21}</h3>
                    <p>Spectral Bands</p>
                </div>
            `;
            
            // Display visualization
            document.getElementById('result-image').src = 'data:image/png;base64,' + result.visualization;
        }
        
        document.getElementById('download-prediction').addEventListener('click', async () => {
            const response = await fetch('/download/prediction');
            const blob = await response.blob();
            downloadBlob(blob, 'green_space_prediction.tif');
        });
        
        document.getElementById('download-visualization').addEventListener('click', async () => {
            const response = await fetch('/download/visualization');
            const blob = await response.blob();
            downloadBlob(blob, 'green_space_visualization.png');
        });
        
        function downloadBlob(blob, filename) {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }
        
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

# Global variables for storing prediction results
latest_prediction = None
latest_visualization = None


def load_or_create_model():
    """Load existing model or create a demo model."""
    global TRAINED_MODEL
    
    if os.path.exists(MODEL_PATH):
        print(f"✓ Loading trained model from {MODEL_PATH}")
        TRAINED_MODEL = joblib.load(MODEL_PATH)
        
        # Try to load metrics from the same folder
        model_dir = os.path.dirname(MODEL_PATH)
        metrics_file = os.path.join(model_dir, "metrics.json")
        
        model_info = {"n_cities": "Multiple", "accuracy": "Trained"}
        
        if os.path.exists(metrics_file):
            try:
                import json
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                accuracy = metrics.get('accuracy', 0) * 100
                n_cities = metrics.get('n_cities', 'Multiple')
                
                model_info = {
                    "n_cities": n_cities,
                    "accuracy": f"{accuracy:.1f}"
                }
                
                print(f"  Model trained on {n_cities} cities")
                print(f"  Accuracy: {accuracy:.1f}%")
                print(f"  Precision: {metrics.get('precision', 0)*100:.1f}%")
                print(f"  Recall: {metrics.get('recall', 0)*100:.1f}%")
                print(f"  F1-Score: {metrics.get('f1_score', 0)*100:.1f}%")
                
            except Exception as e:
                print(f"  Could not load metrics: {e}")
        
        return True, model_info
    else:
        print("⚠ No trained model found. Creating demo model...")
        print(f"  Expected model at: {MODEL_PATH}")
        # Create a simple demo model (you should train a real one)
        TRAINED_MODEL = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42)
        return False, {"n_cities": "Demo", "accuracy": "N/A"}


def predict_green_spaces(sentinel_file):
    """
    Predict green spaces from Sentinel-2 stack.
    """
    global latest_prediction, latest_visualization
    
    # Load Sentinel-2 stack
    with rasterio.open(sentinel_file) as src:
        X_stack = src.read()
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        
    n_bands, height, width = X_stack.shape
    print(f"Loaded stack: {n_bands} bands, {height}×{width} pixels")
    
    # Reshape for prediction
    X = X_stack.reshape(n_bands, -1).T  # (n_pixels, n_bands)
    
    # Handle NaN values
    valid_mask = ~np.isnan(X).any(axis=1)
    X_valid = X[valid_mask]
    
    print(f"Valid pixels: {len(X_valid):,} / {len(X):,} ({100*len(X_valid)/len(X):.1f}%)")
    
    # Predict
    if hasattr(TRAINED_MODEL, 'predict'):
        print("Running Random Forest prediction...")
        y_pred_valid = TRAINED_MODEL.predict(X_valid)
        print(f"Predicted {np.sum(y_pred_valid == 1):,} green pixels")
    else:
        print("WARNING: Using demo NDVI-based prediction")
        # Demo prediction based on NDVI-like bands
        # Assuming bands 4, 11, 18 are NDVI for each month
        ndvi_indices = [4, 11, 18] if n_bands >= 21 else [min(4, n_bands-1)]
        ndvi_indices = [i for i in ndvi_indices if i < n_bands]
        
        if ndvi_indices:
            ndvi_mean = np.mean([X_valid[:, i] for i in ndvi_indices], axis=0)
            y_pred_valid = (ndvi_mean > 0.3).astype(int)
        else:
            # Fallback: use first available band
            y_pred_valid = (X_valid[:, 0] > np.median(X_valid[:, 0])).astype(int)
    
    # Create full prediction map
    y_pred = np.full(height * width, np.nan)
    y_pred[valid_mask] = y_pred_valid
    y_pred_map = y_pred.reshape(height, width)
    
    # Calculate statistics
    green_pixels = np.sum(y_pred_map == 1)
    total_valid_pixels = np.sum(~np.isnan(y_pred_map))
    green_percentage = (green_pixels / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0
    
    stats = {
        "green_pixels": int(green_pixels),
        "total_pixels": int(total_valid_pixels),
        "green_percentage": float(green_percentage),
        "dimensions": f"{height}×{width}",
        "n_bands": int(n_bands)
    }
    
    print(f"\nPrediction complete:")
    print(f"  Green coverage: {green_percentage:.1f}%")
    print(f"  Green pixels: {green_pixels:,}")
    
    # Save prediction map
    temp_dir = tempfile.mkdtemp()
    prediction_file = os.path.join(temp_dir, 'green_space_prediction.tif')
    
    profile.update(count=1, dtype=np.float32, compress='lzw')
    with rasterio.open(prediction_file, 'w', **profile) as dst:
        dst.write(y_pred_map.astype(np.float32), 1)
    
    latest_prediction = prediction_file
    
    # Create visualization
    visualization_file = create_visualization(X_stack, y_pred_map, stats)
    latest_visualization = visualization_file
    
    # Convert visualization to base64
    with open(visualization_file, 'rb') as f:
        viz_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    return {
        "stats": stats,
        "visualization": viz_base64
    }


def create_visualization(X_stack, prediction_map, stats):
    """Create a beautiful visualization of the results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. RGB Composite (using first month)
    rgb_bands = [2, 1, 0]  # B04, B03, B02
    if X_stack.shape[0] >= 3:
        rgb = X_stack[rgb_bands, :, :].transpose(1, 2, 0)
        rgb_norm = np.clip(rgb / 3000, 0, 1)
        axes[0].imshow(rgb_norm)
        axes[0].set_title('RGB (Sentinel-2)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
    
    # 2. NDVI (if available)
    if X_stack.shape[0] >= 5:
        ndvi = X_stack[4, :, :]  # First NDVI band
        im = axes[1].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        axes[1].set_title('NDVI', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Prediction
    axes[2].imshow(prediction_map, cmap='RdYlGn', vmin=0, vmax=1)
    axes[2].set_title(f'Green Space Prediction\n{stats["green_percentage"]:.1f}% Green', 
                     fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle('Green Space Detection Results', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    temp_dir = tempfile.mkdtemp()
    viz_file = os.path.join(temp_dir, 'visualization.png')
    plt.savefig(viz_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    return viz_file


@app.route('/')
def index():
    model_loaded, model_info = load_or_create_model()
    
    # Replace template variables
    html = HTML_TEMPLATE
    html = html.replace("{{ 'loaded' if model_loaded else 'not-loaded' }}", 
                       'loaded' if model_loaded else 'not-loaded')
    html = html.replace("{{ 'check-circle' if model_loaded else 'exclamation-triangle' }}", 
                       'check-circle' if model_loaded else 'exclamation-triangle')
    
    if model_loaded:
        html = html.replace("{% if model_loaded %}", "")
        html = html.replace("{% else %}", "<!--")
        html = html.replace("{% endif %}", "-->")
        html = html.replace("{{ model_info.n_cities }}", str(model_info['n_cities']))
        html = html.replace("{{ model_info.accuracy }}", str(model_info['accuracy']))
    else:
        html = html.replace("{% if model_loaded %}", "<!--")
        html = html.replace("{% else %}", "")
        html = html.replace("{% endif %}", "-->")
    
    return html


@app.route('/predict', methods=['POST'])
def predict():
    try:
        sentinel_file = request.files.get('sentinel_file')
        if not sentinel_file:
            return jsonify({'error': 'No file uploaded'}), 400
        
        # Save temporarily
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, 'sentinel_stack.tif')
        sentinel_file.save(temp_file)
        
        # Predict
        result = predict_green_spaces(temp_file)
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download/<file_type>')
def download(file_type):
    if file_type == 'prediction' and latest_prediction:
        return send_file(latest_prediction, as_attachment=True, 
                        download_name=f'green_space_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tif')
    elif file_type == 'visualization' and latest_visualization:
        return send_file(latest_visualization, as_attachment=True,
                        download_name=f'green_space_visualization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    else:
        return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    print("="*80)
    print("🌳 GREEN SPACE PREDICTOR WEB APPLICATION")
    print("="*80)
    print("\n✓ Starting web server...")
    
    # Load or create model
    model_loaded, model_info = load_or_create_model()
    
    if not model_loaded:
        print("\n⚠️  WARNING: No trained model found!")
        print("   Using demo model with limited accuracy.")
        print("   To use a trained model:")
        print(f"   1. Train a model using the Multi_City_WorldCover_Training notebook")
        print(f"   2. Save it as: {MODEL_PATH}")
        print(f"   3. Restart this application")
    
    print("\n📡 Open in browser: http://localhost:5004")
    print("\n✨ Features:")
    print("  - Upload 21-band Sentinel-2 stack")
    print("  - Instant green space prediction")
    print("  - Interactive visualization")
    print("  - Downloadable results")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("="*80)
    
    app.run(debug=True, port=5004, host='0.0.0.0')