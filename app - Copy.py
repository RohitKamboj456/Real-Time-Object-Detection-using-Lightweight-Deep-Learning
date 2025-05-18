from flask import Flask, request, render_template_string
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import cv2
import numpy as np
import base64
import pickle
import torchvision.transforms as T
 
# Initialize Flask app
app = Flask(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load label map
try:
    with open('label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
    reverse_label_map = {v: k for k, v in label_map.items()}
except FileNotFoundError:
    raise Exception("label_map.pkl not found. Run train.py first.")

# Load model
try:
    num_classes = len(label_map) + 1  # Include background
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.to(device)
    model.eval()
except FileNotFoundError:
    raise Exception("model.pth not found. Run train.py first.")
except Exception as e:
    raise Exception(f"Error loading model: {e}")

# Image transformation
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# HTML template
HTML_TEMPLATE = """
 <!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Object Detection</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-align: center;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #ff9a9e, #fad0c4, #fad0c4, #a1c4fd, #c2e9fb);
            color: #333;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            animation: gradientShift 10s ease infinite;
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .container {
            max-width: 800px;
            margin: 50px auto;
            padding: 30px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            transition: transform 0.5s ease, box-shadow 0.5s ease;
        }
        .container:hover {
            transform: translateY(-10px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        h1 {
            font-size: 2.8em;
            margin-bottom: 25px;
            background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            color: transparent;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            animation: textGlow 2s ease infinite;
        }
        @keyframes textGlow {
            0% { text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
            50% { text-shadow: 2px 2px 8px rgba(255, 107, 107, 0.5); }
            100% { text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
        }
        form {
            margin: 25px 0;
        }
        input[type="file"] {
            display: none;
        }
        .custom-file-upload {
            display: inline-block;
            padding: 12px 25px;
            cursor: pointer;
            background: linear-gradient(135deg, #ff9966, #ff5e62);
            color: white;
            border-radius: 8px;
            font-size: 1.1em;
            margin-bottom: 25px;
            transition: all 0.4s ease;
        }
        .custom-file-upload:hover {
            background: linear-gradient(135deg, #ff5e62, #ff9966);
            transform: scale(1.1) rotate(2deg);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }
        input[type="submit"] {
            padding: 12px 25px;
            background: linear-gradient(135deg, #00c6ff, #0072ff);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1.1em;
            transition: all 0.4s ease;
        }
        input[type="submit"]:hover {
            background: linear-gradient(135deg, #0072ff, #00c6ff);
            transform: scale(1.1) rotate(-2deg);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }
        #loader {
            display: none;
            font-size: 1.3em;
            color: #ff6b6b;
            margin: 25px 0;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .result-container {
            margin-top: 35px;
            padding: 20px;
            background: linear-gradient(135deg, #f6d365, #fda085);
            border-radius: 10px;
            transition: transform 0.5s ease;
        }
        .result-container:hover {
            transform: scale(1.03);
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            transition: transform 0.4s ease;
        }
        img:hover {
            transform: scale(1.05) rotate(1deg);
        }
        .error {
            color: #ff4444;
            font-size: 1.3em;
            margin: 25px 0;
            background: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 8px;
        }
    </style>
    <script>
        function showLoader() {
            document.getElementById('loader').style.display = 'block';
        }
        function triggerFileUpload() {
            document.getElementById('fileInput').click();
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>Upload an Image for Object Detection</h1>
        <form method="post" enctype="multipart/form-data" onsubmit="showLoader()">
            <input type="file" id="fileInput" name="file" accept="image/*" required>
            <label for="fileInput" class="custom-file-upload">Choose Image</label>
            <input type="submit" value="Upload">
        </form>
        <div id="loader">Processing...</div>
        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}
        {% if image_base64 %}
            <div class="result-container">
                <h2>Result:</h2>
                <img src="data:image/jpeg;base64,{{ image_base64 }}" alt="Detection Result">
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle image upload and inference."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template_string(HTML_TEMPLATE, error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template_string(HTML_TEMPLATE, error="No file selected")

        try:
            # Preprocess image
            img = Image.open(file.stream).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            # Perform inference
            with torch.no_grad():
                predictions = model(img_tensor)[0]

            # Filter predictions
            boxes = predictions['boxes'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            threshold = 0.5
            mask = scores > threshold
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]

            # Draw bounding boxes
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, box)
                class_name = reverse_label_map.get(label, "Unknown")
                text = f"{class_name}: {score:.2f}"
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_cv, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', img_cv)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return render_template_string(HTML_TEMPLATE, image_base64=img_base64)

        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=f"Error processing image: {e}")

    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)