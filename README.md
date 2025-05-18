# Real-Time-Object-Detection-using-Lightweight-Deep-Learning
 
A full-stack Python project that performs real-time object detection using a lightweight Faster R-CNN model. This project uses `PyTorch` and `TorchVision` for training, and `Flask` with `HTML/CSS/JS` for building a clean user-friendly web interface. It works with custom-labeled data in COCO-like CSV format for training and testing.

## 🚀 Project Overview

This project aims to:
- Train a lightweight object detection model using PyTorch.
- Use CSV annotation format (from Roboflow or similar).
- Deploy the model with a simple and responsive Flask web app.
- Provide real-time detection on uploaded images.

## 📁 Dataset Structure

Ensure your dataset follows this structure:
D:\New folder (2)\
│
├── train\
│ ├── image1.jpg
│ ├── image2.jpg
│ └── _annotations.csv
│├── test\
│ ├── imageA.jpg
│ ├── imageB.jpg
│ └── _annotations.csv

# The `_annotations.csv` should follow this format:
filename,width,height,class,xmin,ymin,xmax,ymax
image1.jpg,640,480,object1,34,45,120,160
🛠️ Tech Stack
Component	Library/Tool
Model Training	PyTorch, TorchVision
Web App	Flask, HTML, CSS, JS
Data Handling	Pandas, NumPy
Visualization	Matplotlib, PIL
Training the Model
Train your Faster R-CNN model using:

bash
Always show details

Copy
python train.py
Make sure the paths to train and test folders are correctly set in train.py.

Uses FasterRCNN_ResNet50_FPN_Weights.DEFAULT for up-to-date weights.

Model is saved as model.pth.
 Running the Web Application
Place the trained model.pth in the project root.

Start the Flask server:

bash
Always show details

Copy
python app.py
Visit http://127.0.0.1:5000/ in your browser.

You can upload images and the model will detect and display the objects on them in real-time
Features
✅ Real-time object detection with bounding boxes

✅ Lightweight model (Faster R-CNN)

✅ Custom dataset support via CSV

✅ Responsive HTML/CSS/JS frontend

✅ Clean and modular code
🧠 Future Enhancements
Add image/video webcam detection

Add support for different models (YOLOv5, SSD)

Deploy on cloud (Render, Hugging Face Spaces)

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first.

📜 License
This project is licensed under the MIT License.

👤 Author
Rohit Kamboj
LinkedIn | GitHub



