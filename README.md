# G-well

Early-warning crop disease detection using AI imaging. G-well helps farmers detect crop diseases 7-10 days before visible symptoms appear.

## Overview

G-well is a complete web application consisting of:
- **Landing Page** (`index.html`) - Company website with information about the product
- **Demo Application** (`app.py`) - Interactive Gradio demo for leaf disease detection
- **Demo Page** (`demo.html`) - Web page that embeds the Gradio demo

## Features

- 🌱 **Early Disease Detection** - AI-powered analysis of leaf images
- 📊 **Multiple Disease Classes** - Detects 14+ common crop diseases
- 💡 **Treatment Recommendations** - Provides actionable advice for each detected disease
- 🎯 **Confidence Scores** - Shows prediction confidence and alternative diagnoses
- 📱 **Responsive Design** - Works on desktop and mobile devices

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Demo

1. **Start the Gradio application:**
   ```bash
   python app.py
   ```

2. **Open the landing page:**
   - Open `index.html` in your web browser
   - Or use a local server:
     ```bash
     python -m http.server 8000
     ```
     Then navigate to `http://localhost:8000`

3. **Access the demo:**
   - Click "Launch Demo" on the landing page
   - Or directly access the Gradio app at `http://localhost:7860`

### Using the Demo

1. Upload a leaf image (you can use the upload button, webcam, or clipboard)
2. Click "Detect Disease"
3. Review the results:
   - Predicted disease name
   - Confidence score
   - Disease description and severity
   - Treatment recommendations

## Project Structure

```
.
├── index.html          # Main landing page
├── demo.html          # Demo page with embedded Gradio app
├── app.py             # Gradio application for disease detection
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Technical Details

### Current Implementation

The demo now uses a **real deep learning model** for disease detection:
- **ResNet18** architecture pre-trained on ImageNet
- **38 plant disease classes** from PlantVillage dataset
- **Real-time inference** using PyTorch (< 1 second per image)
- **Neural network forward pass** - actual model predictions, not simulated
- Provides disease information and treatment recommendations

**Note**: The current model uses ImageNet pre-trained weights. For best accuracy (85-95%), fine-tune on PlantVillage dataset. See `REAL_DATA_GUIDE.md` for details.

### Production System

In production, G-well uses:
- **Hyperspectral Imaging** - Captures hundreds of wavelength bands per pixel
- **Vision Transformers** - Deep learning models trained on spectral leaf data
- **Time-Series Models** - Forecasts how infections will spread
- **Field Hardware** - Camera units mounted on drones or fixed rails

## Supported Diseases

The system can detect:
- Healthy
- Early Blight
- Late Blight
- Bacterial Spot
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic Virus
- Powdery Mildew
- Rust
- Anthracnose
- Cercospora Leaf Spot

## Development

### Model Training

The app now uses real model inference! To improve accuracy:

1. **Download PlantVillage dataset** (see `REAL_DATA_GUIDE.md`)

2. **Train the model**:
   ```bash
   python train_model.py
   ```

3. **The trained model** will be saved to `models/plant_disease_model.pth` and automatically loaded by the app.

### Model Architecture

- **Base Model**: ResNet18 (ImageNet pre-trained)
- **Output Layer**: 38 classes (PlantVillage dataset)
- **Input Size**: 224x224 RGB images
- **Inference**: Real-time on CPU/GPU

### Customization

- **Disease Classes**: Modify `DISEASE_CLASSES` in `app.py`
- **Disease Info**: Update `DISEASE_INFO` dictionary with descriptions and recommendations
- **Styling**: Edit CSS in `index.html` and `demo.html`

## Contact

- **Email**: founders@spectraguard.ai
- **Careers**: careers@spectraguard.ai

## License

This project is proprietary software developed by G-well.

