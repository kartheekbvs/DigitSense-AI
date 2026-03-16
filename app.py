import os
import io
import base64
import numpy as np
from PIL import Image
import PIL.ImageOps
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

try:
    import torch
    import torch.nn as nn
    import joblib
    import threading
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    print("WARNING: ML dependencies (Torch/Joblib) not installed.")

app = Flask(__name__)
CORS(app)

# Thread safety for lazy loading
model_lock = threading.Lock()
models_loaded = False

# Load Models
MODEL_DIR = 'models'
pytorch_model = None
pca_transformer = None
lr_model = None

# Robust architecture definition for PyTorch loading
class DigitCNN(nn.Module if HAS_ML_DEPS else object):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_all_models():
    global pca_transformer, lr_model, pytorch_model, models_loaded
    if not HAS_ML_DEPS or models_loaded:
        return
    
    with model_lock:
        if models_loaded: return
    try:
        print(f"Loading models from {os.path.abspath(MODEL_DIR)}...")
        
        # 1. Load PyTorch CNN (Primary)
        pth_path = os.path.join(MODEL_DIR, 'digit_cnn.pth')
        if os.path.exists(pth_path):
            try:
                pytorch_model = DigitCNN()
                pytorch_model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
                pytorch_model.eval()
                print("Successfully loaded PyTorch CNN model.")
            except Exception as e:
                print(f"Failed to load PyTorch binary: {e}")

        # 2. PCA and LR (Fallback)
        pca_path = os.path.join(MODEL_DIR, 'pca_transformer.pkl')
        if os.path.exists(pca_path):
            pca_transformer = joblib.load(pca_path)
            print("Successfully loaded PCA transformer.")
            
        lr_path = os.path.join(MODEL_DIR, 'lr_model.pkl')
        if os.path.exists(lr_path):
            lr_model = joblib.load(lr_path)
            print("Successfully loaded Logistic Regression model.")
            
        if pytorch_model:
            print("--- DEEP LEARNING MODEL READY ---")
        elif pca_transformer and lr_model:
            print("--- CLASSICAL FALLBACK READY ---")
            
        models_loaded = True
    except Exception as e:
        print(f"Exception during load_all_models: {e}")
        import traceback
        traceback.print_exc()

# Removed top-level load_all_models() to prevent startup timeouts

def preprocess_image(base64_string):
    """ Converts base64 image from canvas to normalized numpy array. """
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
        
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data)).convert('RGBA')
    
    background = Image.new('RGBA', img.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, img)
    gray_img = alpha_composite.convert('L')
    
    # Invert colors
    inverted_img = PIL.ImageOps.invert(gray_img)
    
    # If we have CNN, we need 28x28.
    img_resized = inverted_img.resize((28, 28), resample=Image.Resampling.LANCZOS)
    img_array = np.array(img_resized).astype('float32') / 255.0
    
    if pytorch_model:
        return img_array.reshape(1, 1, 28, 28) # Torch format: (B, C, H, W)
    else:
        # Flatten to 784 for PCA + LR
        return img_array.reshape(1, 784)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'models_loaded': models_loaded})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Lazy load models on the first request
        if not models_loaded:
            print("First request received. Lazy loading models...")
            load_all_models()
            
        if not (pytorch_model or (pca_transformer and lr_model)):
            return jsonify({'error': 'Models are not loaded properly on the server.'}), 503

        processed_input = preprocess_image(data['image'])
        
        # 1. Try PyTorch CNN (High Accuracy)
        if pytorch_model:
            # MNIST normalization used in training: (0.1307,), (0.3081,)
            # Our input is already (1, 1, 28, 28) from preprocess_image
            norm_input = (processed_input - 0.1307) / 0.3081
            input_tensor = torch.from_numpy(norm_input).float()
            
            with torch.no_grad():
                output = pytorch_model(input_tensor)
                probabilities = torch.softmax(output, dim=1).numpy()[0]
                predicted_class = int(np.argmax(probabilities))
        
        # 2. Fallback to Classical ML
        elif pca_transformer and lr_model:
            features = processed_input.reshape(1, 784)
            pca_features = pca_transformer.transform(features)
            probabilities = lr_model.predict_proba(pca_features)[0]
            predicted_class = int(np.argmax(probabilities))

        confidences = []
        for digit, prob in enumerate(probabilities):
            confidences.append({
                'digit': digit,
                'conf': round(float(prob) * 100, 1)
            })
        confidences.sort(key=lambda x: x['conf'], reverse=True)

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidences': confidences,
            'method': 'CNN' if pytorch_model else 'Classical'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
