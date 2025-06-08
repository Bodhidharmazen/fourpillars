import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
import time
import threading
import json
from flask import Flask, render_template, jsonify

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# Parameters for the simulation
N = 100  # Grid size
params = {
    'coupling': 1.0,      # Field coupling (Self-Instantiation)
    'nonlinearity': 0.6,  # Nonlinear processing (World-Building)
    'noise': 0.02,        # Noise level (Disturbance-Driven)
    'resonance': 0.5,     # Resonance (Cognitive Resonance)
    'convergence': 0.7    # Convergence rate (Temporal Continuity)
}

# Global variables for simulation state
field1 = None
field2 = None
phenomenal = None
last_frame_time = time.time()
frame_times = []
metrics = {
    'fps': 0.0,              # Processing Rate
    'entropy': 0.0,          # Phenomenal Integration
    'field_diff': 0.0,       # Self-Pattern Stability
    'resonance': 0.0,        # Global Workspace Coherence
    'complexity': 0.0        # World Model Complexity
}

# Mexican hat kernel for convolution
kernel = np.array([
    [0.05, 0.1, 0.05],
    [0.1, -0.9, 0.1],
    [0.05, 0.1, 0.05]
])

# Initialize fields with concentric rings pattern
def initialize_fields():
    global field1, field2
    x, y = np.meshgrid(np.linspace(-4, 4, N), np.linspace(-4, 4, N))
    r = np.sqrt(x**2 + y**2)
    field1 = np.sin(r*1.5) * 0.4
    field2 = np.sin(r*1.2) * 0.4

# Initialize fields at startup
initialize_fields()

# Update neural fields based on MDO dynamics
def update_fields():
    global field1, field2, phenomenal, metrics, last_frame_time, frame_times
    
    # Convolution
    f1_conv = convolve2d(field1, kernel, mode='same', boundary='wrap')
    f2_conv = convolve2d(field2, kernel, mode='same', boundary='wrap')
    
    # Extract parameters
    coupling = params['coupling']
    nonlinearity = params['nonlinearity']
    noise_level = params['noise']
    resonance = params['resonance']
    convergence = params['convergence']
    
    # Update fields with MDO dynamics
    f1_new = convergence * field1 + (1-convergence) * np.tanh(f1_conv + coupling * field2)
    f2_new = convergence * field2 + (1-convergence) * np.tanh(f2_conv + coupling * field1)
    
    # Nonlinear resonance terms
    f1_new += nonlinearity * resonance * np.sin(field2 * 2.0) * 0.2
    f2_new += nonlinearity * resonance * np.sin(field1 * 2.0) * 0.2
    
    # Add noise
    f1_new += noise_level * np.random.randn(*field1.shape)
    f2_new += noise_level * np.random.randn(*field2.shape)
    
    field1, field2 = f1_new, f2_new
    
    # Calculate phenomenal world
    phenomenal = np.abs(field1 - field2) + 0.1 * (field1 + field2)**2
    
    # Enhance contrast
    p_min, p_max = np.min(phenomenal), np.max(phenomenal)
    if p_max > p_min:
        phenomenal = (phenomenal - p_min) / (p_max - p_min)
        phenomenal = np.power(phenomenal, 0.85)  # Gamma correction
    
    # Calculate FPS
    current_time = time.time()
    elapsed = current_time - last_frame_time
    last_frame_time = current_time
    
    frame_times.append(elapsed)
    if len(frame_times) > 20:
        frame_times.pop(0)
    
    if frame_times:
        metrics['fps'] = 1.0 / (sum(frame_times) / len(frame_times))
    
    # Calculate MDO metrics
    metrics['entropy'] = -np.sum(phenomenal * np.log2(phenomenal + 1e-10)) / (N * N)
    metrics['field_diff'] = np.mean(np.abs(field1 - field2))
    metrics['resonance'] = np.mean(np.abs(field1 * field2)) / (np.mean(np.abs(field1)) * np.mean(np.abs(field2)) + 1e-10)
    
    # Calculate dynamic complexity
    f1_fft = np.abs(np.fft.fft2(field1))
    metrics['complexity'] = -np.sum((f1_fft * np.log2(f1_fft + 1e-10))) / np.sum(f1_fft)

# Generate visualization frame
def generate_frame():
    # Update the simulation
    update_fields()
    
    # Create a figure with the phenomenal field
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Custom colormap - vibrant psychedelic gradient
    colors = [(0, 0, 0.3), (0, 0.3, 0.8), (0, 0.8, 0.8), 
              (0.8, 0.8, 0), (0.8, 0.4, 0), (0.8, 0, 0.8)]
    cm = matplotlib.colors.LinearSegmentedColormap.from_list("psychedelic", colors, N=256)
    
    # Plot the phenomenal field
    im = ax.imshow(phenomenal, cmap=cm, origin='lower', interpolation='bicubic')
    plt.colorbar(im, ax=ax, label='Phenomenal Field Intensity')
    
    # Clean up the plot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('MDO Phenomenal World', color='cyan')
    plt.tight_layout()
    
    # Convert figure to PNG image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    
    return buf

@app.route('/')
def index():
    return render_template('four_pillars.html')

@app.route('/get-frame')
def get_frame():
    buf = generate_frame()
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    return jsonify({
        'image': img_data,
        'metrics': metrics,
        'params': params
    })

@app.route('/update-param/<param>/<value>')
def update_param(param, value):
    global params
    
    try:
        value = float(value)
        # Map conceptual parameters to actual parameters
        if param == 'coupling' or param == 'self-instantiation':
            params['coupling'] = value
        elif param == 'nonlinearity' or param == 'world-building':
            params['nonlinearity'] = value
        elif param == 'noise' or param == 'disturbance-sensitivity':
            params['noise'] = value
        elif param == 'resonance' or param == 'cognitive-resonance':
            params['resonance'] = value
        elif param == 'convergence' or param == 'temporal-continuity':
            params['convergence'] = value
        
        return jsonify({'message': f'Parameter {param} updated to {value}'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/reset/<pattern_type>')
def reset(pattern_type):
    global field1, field2
    x, y = np.meshgrid(np.linspace(-4, 4, N), np.linspace(-4, 4, N))
    
    if pattern_type == 'rings':
        # Concentric rings pattern
        r = np.sqrt(x**2 + y**2)
        field1 = np.sin(r*np.random.uniform(1.0, 2.0)) * 0.4
        field2 = np.sin(r*np.random.uniform(1.0, 2.0)) * 0.4
    elif pattern_type == 'spiral':
        # Spiral pattern
        theta = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
        field1 = np.sin(theta*4 + r*0.5) * 0.4
        field2 = np.sin(theta*3 - r*0.5) * 0.4
    elif pattern_type == 'waves':
        # Wave interference pattern
        field1 = np.sin(x*1.5) * np.cos(y*1.5) * 0.4
        field2 = np.cos(x*1.0) * np.sin(y*1.0) * 0.4
    
    return jsonify({'message': f'Pattern reset to {pattern_type}'})

if __name__ == '__main__':
    # For local development
    # app.run(debug=True, port=5050)
    # For production deployment
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5050)))
