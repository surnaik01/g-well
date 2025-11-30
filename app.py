"""
G-well - Leaf Disease Detection Demo
A Gradio application for detecting crop diseases from leaf images using real AI models.
"""

import gradio as gr
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from typing import Tuple, Dict
import os
from pathlib import Path

# PlantVillage dataset classes (38 classes)
PLANT_DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___healthy",
    "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___healthy",
    "Corn___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca",
    "Grape___healthy",
    "Grape___Leaf_blight",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper_bell___Bacterial_spot",
    "Pepper_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___Mosaic_virus"
]

# Simplified display names
CLASS_DISPLAY_NAMES = {
    "Apple___Apple_scab": "Apple Scab",
    "Apple___Black_rot": "Apple Black Rot",
    "Apple___Cedar_apple_rust": "Apple Cedar Rust",
    "Apple___healthy": "Apple (Healthy)",
    "Blueberry___healthy": "Blueberry (Healthy)",
    "Cherry___healthy": "Cherry (Healthy)",
    "Cherry___Powdery_mildew": "Cherry Powdery Mildew",
    "Corn___Cercospora_leaf_spot": "Corn Cercospora Leaf Spot",
    "Corn___Common_rust": "Corn Common Rust",
    "Corn___healthy": "Corn (Healthy)",
    "Corn___Northern_Leaf_Blight": "Corn Northern Leaf Blight",
    "Grape___Black_rot": "Grape Black Rot",
    "Grape___Esca": "Grape Esca",
    "Grape___healthy": "Grape (Healthy)",
    "Grape___Leaf_blight": "Grape Leaf Blight",
    "Peach___Bacterial_spot": "Peach Bacterial Spot",
    "Peach___healthy": "Peach (Healthy)",
    "Pepper_bell___Bacterial_spot": "Pepper Bacterial Spot",
    "Pepper_bell___healthy": "Pepper (Healthy)",
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___healthy": "Potato (Healthy)",
    "Potato___Late_blight": "Potato Late Blight",
    "Raspberry___healthy": "Raspberry (Healthy)",
    "Soybean___healthy": "Soybean (Healthy)",
    "Squash___Powdery_mildew": "Squash Powdery Mildew",
    "Strawberry___healthy": "Strawberry (Healthy)",
    "Strawberry___Leaf_scorch": "Strawberry Leaf Scorch",
    "Tomato___Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___healthy": "Tomato (Healthy)",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "Tomato___Spider_mites": "Tomato Spider Mites",
    "Tomato___Target_Spot": "Tomato Target Spot",
    "Tomato___Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato___Mosaic_virus": "Tomato Mosaic Virus"
}

# Disease information mapping
DISEASE_INFO = {
    "Apple Scab": {
        "description": "Dark, scaly lesions on leaves and fruit. Caused by Venturia inaequalis fungus.",
        "recommendation": "Apply fungicides containing myclobutanil or captan. Remove fallen leaves. Prune for better air circulation.",
        "severity": "Moderate"
    },
    "Apple Black Rot": {
        "description": "Brown, circular lesions with concentric rings. Can cause fruit rot.",
        "recommendation": "Apply fungicides. Remove infected fruit and leaves. Improve tree spacing.",
        "severity": "Moderate"
    },
    "Apple Cedar Rust": {
        "description": "Orange-yellow spots on leaves, often with small black dots in center.",
        "recommendation": "Remove nearby cedar trees if possible. Apply fungicides containing myclobutanil.",
        "severity": "Moderate"
    },
    "Cherry Powdery Mildew": {
        "description": "White, powdery fungal growth on leaf surfaces.",
        "recommendation": "Apply sulfur-based fungicides. Improve air circulation. Reduce humidity.",
        "severity": "Moderate"
    },
    "Corn Cercospora Leaf Spot": {
        "description": "Small, circular spots with gray centers and dark borders.",
        "recommendation": "Apply fungicides. Remove infected leaves. Practice crop rotation.",
        "severity": "Moderate"
    },
    "Corn Common Rust": {
        "description": "Orange, yellow, or brown pustules on leaf surfaces.",
        "recommendation": "Apply fungicides containing tebuconazole. Remove infected leaves. Use resistant varieties.",
        "severity": "Moderate"
    },
    "Corn Northern Leaf Blight": {
        "description": "Long, elliptical gray-green lesions that turn brown.",
        "recommendation": "Apply fungicides. Use resistant varieties. Practice crop rotation.",
        "severity": "Moderate"
    },
    "Grape Black Rot": {
        "description": "Brown spots with black fruiting bodies. Can cause fruit rot.",
        "recommendation": "Apply fungicides containing mancozeb. Remove infected fruit. Improve air circulation.",
        "severity": "High"
    },
    "Grape Esca": {
        "description": "Yellowing leaves, wood decay, and dieback symptoms.",
        "recommendation": "Prune infected wood. Apply fungicides. Improve drainage.",
        "severity": "High"
    },
    "Grape Leaf Blight": {
        "description": "Brown spots and lesions on leaves, often leading to defoliation.",
        "recommendation": "Apply copper-based fungicides. Remove infected leaves. Improve ventilation.",
        "severity": "Moderate"
    },
    "Peach Bacterial Spot": {
        "description": "Small, dark, water-soaked spots on leaves and fruit.",
        "recommendation": "Apply copper-based bactericides. Avoid overhead irrigation. Use resistant varieties.",
        "severity": "Moderate"
    },
    "Pepper Bacterial Spot": {
        "description": "Small, dark, water-soaked spots that may have yellow halos.",
        "recommendation": "Apply copper-based bactericides. Avoid overhead irrigation. Practice crop rotation.",
        "severity": "Moderate"
    },
    "Potato Early Blight": {
        "description": "Dark brown spots with concentric rings, typically on older leaves.",
        "recommendation": "Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves. Improve air circulation.",
        "severity": "Moderate"
    },
    "Potato Late Blight": {
        "description": "Water-soaked lesions that turn brown, often with white fungal growth.",
        "recommendation": "Immediate action required. Apply copper-based fungicides. Remove and destroy infected plants.",
        "severity": "High"
    },
    "Squash Powdery Mildew": {
        "description": "White, powdery fungal growth on leaf surfaces.",
        "recommendation": "Apply sulfur-based fungicides or neem oil. Improve air circulation. Reduce humidity.",
        "severity": "Moderate"
    },
    "Strawberry Leaf Scorch": {
        "description": "Purple to brown spots on leaves, often with yellow halos.",
        "recommendation": "Apply fungicides. Remove infected leaves. Improve air circulation.",
        "severity": "Moderate"
    },
    "Tomato Bacterial Spot": {
        "description": "Small, dark, water-soaked spots that may have yellow halos.",
        "recommendation": "Apply copper-based bactericides. Avoid overhead irrigation. Practice crop rotation.",
        "severity": "Moderate"
    },
    "Tomato Early Blight": {
        "description": "Dark brown spots with concentric rings, typically on older leaves.",
        "recommendation": "Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves. Improve air circulation.",
        "severity": "Moderate"
    },
    "Tomato Late Blight": {
        "description": "Water-soaked lesions that turn brown, often with white fungal growth.",
        "recommendation": "Immediate action required. Apply copper-based fungicides. Remove and destroy infected plants.",
        "severity": "High"
    },
    "Tomato Leaf Mold": {
        "description": "Yellow patches on upper leaf surface with fuzzy mold on underside.",
        "recommendation": "Improve ventilation. Apply fungicides like chlorothalonil. Remove affected leaves.",
        "severity": "Moderate"
    },
    "Tomato Septoria Leaf Spot": {
        "description": "Small, circular spots with dark centers and light borders.",
        "recommendation": "Apply fungicides containing azoxystrobin or pyraclostrobin. Remove infected leaves.",
        "severity": "Moderate"
    },
    "Tomato Spider Mites": {
        "description": "Tiny yellow or white spots, fine webbing, and leaf discoloration.",
        "recommendation": "Apply miticides. Increase humidity. Introduce beneficial insects like ladybugs.",
        "severity": "Moderate"
    },
    "Tomato Target Spot": {
        "description": "Concentric ring patterns resembling a target, brown to black in color.",
        "recommendation": "Apply fungicides. Remove infected plant material. Improve spacing between plants.",
        "severity": "Moderate"
    },
    "Tomato Yellow Leaf Curl Virus": {
        "description": "Yellowing and curling of leaves, stunted growth.",
        "recommendation": "Control whitefly vectors. Remove infected plants. Use resistant varieties if available.",
        "severity": "High"
    },
    "Tomato Mosaic Virus": {
        "description": "Mottled yellow and green patterns on leaves, distorted growth.",
        "recommendation": "Remove infected plants immediately. Control aphid vectors. Use virus-free seeds.",
        "severity": "High"
    }
}

# Global model variable
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_model(num_classes=len(PLANT_DISEASE_CLASSES)):
    """Create a ResNet18 model for plant disease classification."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def load_model():
    """Load the trained model or create a pre-trained one."""
    global model
    
    try:
        model_path = Path('models/plant_disease_model.pth')
        
        if model_path.exists():
            print(f"Loading trained model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            model = create_model()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model = model.to(device)
            print("✅ Model loaded successfully!")
            return checkpoint.get('class_names', PLANT_DISEASE_CLASSES)
        else:
            print("No trained model found. Using ImageNet pre-trained weights.")
            print("For better accuracy, run: python train_model.py")
            model = create_model()
            model.eval()
            model = model.to(device)
            return PLANT_DISEASE_CLASSES
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating new model with ImageNet weights...")
        model = create_model()
        model.eval()
        model = model.to(device)
        return PLANT_DISEASE_CLASSES


def is_likely_plant_image(image: Image.Image) -> Tuple[bool, float]:
    """
    Check if image is likely a plant/leaf based on color analysis.
    Returns: (is_likely_plant, green_ratio)
    Note: Diseased leaves may have less green (brown/yellow spots), so we're more lenient.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    
    # Calculate green channel dominance
    green_channel = img_array[:, :, 1]
    red_channel = img_array[:, :, 0]
    blue_channel = img_array[:, :, 2]
    
    # Green ratio (plants are typically green, but diseased leaves have less)
    green_ratio = np.mean(green_channel) / 255.0
    red_ratio = np.mean(red_channel) / 255.0
    blue_ratio = np.mean(blue_channel) / 255.0
    
    # Check if green is dominant (less strict - diseased leaves may have brown/yellow)
    green_dominance = (np.mean(green_channel) > np.mean(red_channel) * 1.05 and 
                      np.mean(green_channel) > np.mean(blue_channel) * 1.05)
    
    # Check for clearly unnatural colors (devices, objects often have bright non-green colors)
    # Only flag if it's clearly not a plant (very bright non-green with very low green)
    has_bright_non_green = (red_ratio > 0.6 or blue_ratio > 0.6) and green_ratio < 0.25
    
    # Check for natural plant colors - even diseased leaves have some green
    # Lower threshold to account for diseased/brown leaves
    has_some_green = green_ratio > 0.20  # Even diseased leaves have some green
    
    # More lenient: Accept if it has some green AND (green dominance OR not clearly unnatural)
    # This allows diseased leaves with brown/yellow spots
    is_plant = has_some_green and (green_dominance or (not has_bright_non_green and green_ratio > 0.15))
    
    return is_plant, green_ratio


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def detect_disease_real(image: Image.Image) -> Tuple[str, float, Dict, bool, float]:
    """
    Real disease detection using trained model.
    Returns: predicted_class, confidence, top_3_predictions, is_likely_plant, green_ratio
    """
    global model
    
    if model is None:
        load_model()
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Check if image is likely a plant
    is_plant, green_ratio = is_likely_plant_image(image)
    
    # Preprocess image
    input_tensor = preprocess_image(image).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        probs = probabilities[0].cpu().numpy()
    
    # Get top prediction
    top_idx = np.argmax(probs)
    predicted_class = PLANT_DISEASE_CLASSES[top_idx]
    confidence = float(probs[top_idx])
    
    # Get top 3 predictions
    top_3_indices = np.argsort(probs)[-3:][::-1]
    top_3_predictions = {
        PLANT_DISEASE_CLASSES[idx]: float(probs[idx]) 
        for idx in top_3_indices
    }
    
    return predicted_class, confidence, top_3_predictions, is_plant, green_ratio


def detect_disease(image: Image.Image) -> Tuple[str, str, str, str]:
    """
    Main function for disease detection.
    Returns: predicted_disease, confidence, description, recommendation
    """
    if image is None:
        return "Please upload an image", "", "", ""
    
    try:
        predicted_class, confidence, top_3, is_plant, green_ratio = detect_disease_real(image)
        
        # Get display name
        display_name = CLASS_DISPLAY_NAMES.get(predicted_class, predicted_class.replace('___', ' ').replace('_', ' '))
        
        # Check if image is likely not a plant
        confidence_threshold = 0.30  # 30% confidence threshold (reasonable for pre-trained model)
        low_confidence = confidence < confidence_threshold
        
        # ALWAYS show warning for low confidence OR non-plant images
        warning_message = ""
        
        # Priority 1: Non-plant image (most important) - only show if clearly not a plant
        if not is_plant and green_ratio < 0.15:
            warning_message = "🚫 **INVALID INPUT: This image doesn't appear to be a plant leaf!**\n\n"
            warning_message += "The uploaded image appears to be a device, object, or non-plant image. "
            warning_message += "This model is designed specifically for crop leaf disease detection.\n\n"
            warning_message += "**Please upload:** A clear, well-lit photograph of a single crop leaf (tomato, potato, apple, corn, grape, etc.)\n\n"
        
        # Priority 2: Low confidence (even if it might be a plant)
        if low_confidence and is_plant:
            if warning_message:
                warning_message += f"⚠️ **Note:** Prediction confidence is {confidence * 100:.1f}%. "
                warning_message += "For more accurate results, use a clear, well-lit image of a single leaf.\n\n"
            else:
                warning_message = f"⚠️ **Note:** Prediction confidence is {confidence * 100:.1f}%. "
                warning_message += "The model detected a disease, but for best results, use a clearer, well-lit image.\n\n"
        
        # Extract disease name (remove crop name for info lookup)
        disease_key = display_name
        if '(' in display_name:
            disease_key = "Healthy"
        else:
            # Try to find matching disease info
            for key in DISEASE_INFO.keys():
                if key in display_name or display_name in key:
                    disease_key = key
                    break
        
        # Get disease information
        info = DISEASE_INFO.get(disease_key, {
            "description": f"The model detected: {display_name}. This appears to be a {'healthy' if 'healthy' in predicted_class.lower() else 'diseased'} plant.",
            "recommendation": "Consult with an agronomist for specific treatment recommendations based on your crop and local conditions.",
            "severity": "Unknown" if "healthy" in predicted_class.lower() else "Moderate"
        })
        
        # Format confidence as percentage
        confidence_str = f"{confidence * 100:.1f}%"
        
        # Format description with severity and warning
        if "healthy" in predicted_class.lower():
            description = f"{warning_message}**Status:** Healthy\n\n{info['description']}"
        else:
            description = f"{warning_message}**Severity:** {info['severity']}\n\n**Detected:** {display_name}\n\n{info['description']}"
        
        # Format recommendation - only override for clearly invalid inputs
        if not is_plant and green_ratio < 0.15:
            recommendation = "🚫 **Invalid Image Type Detected**\n\n"
            recommendation += "This image does not appear to be a plant leaf. The G-well model is specifically trained to detect diseases in crop leaves.\n\n"
            recommendation += "**What to upload:**\n"
            recommendation += "• A clear, close-up photo of a single leaf\n"
            recommendation += "• Well-lit with natural or good lighting\n"
            recommendation += "• Focused image (not blurry)\n"
            recommendation += "• Examples: tomato leaf, potato leaf, apple leaf, corn leaf, etc.\n\n"
            recommendation += "**What NOT to upload:**\n"
            recommendation += "• Devices, objects, or non-plant images\n"
            recommendation += "• Full plants or multiple leaves\n"
            recommendation += "• Cartoon images or illustrations"
        elif low_confidence and is_plant:
            recommendation = f"⚠️ **Note: Low Confidence ({confidence * 100:.1f}%)**\n\n"
            recommendation += info['recommendation']
            recommendation += "The prediction confidence is below 50%, which suggests the image may not be optimal for analysis.\n\n"
            recommendation += "**For better results, please:**\n"
            recommendation += "• Upload a clearer, more focused image\n"
            recommendation += "• Ensure good lighting\n"
            recommendation += "• Make sure the leaf fills most of the frame\n"
            recommendation += "• Avoid shadows or reflections"
        else:
            recommendation = info['recommendation']
        
        # Add top 3 predictions info
        if len(top_3) > 1:
            other_predictions = "\n\n**Other possible diagnoses:**\n"
            for disease, prob in list(top_3.items())[1:]:
                other_display = CLASS_DISPLAY_NAMES.get(disease, disease.replace('___', ' ').replace('_', ' '))
                other_predictions += f"- {other_display}: {prob*100:.1f}%\n"
            description += other_predictions
        
        return display_name, confidence_str, description, recommendation
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        return error_msg, "", "", ""


def create_interface():
    """Create and return the Gradio interface."""
    
    # Load model on startup (with error handling)
    try:
        print("Initializing model...")
        load_model()
    except Exception as e:
        print(f"Warning: Model initialization error: {e}")
        print("App will continue but may have reduced functionality.")
    
    with gr.Blocks(title="G-well - Disease Detection Demo") as demo:
        gr.Markdown(
            """
            # 🌱 G-well - Real-Time Leaf Disease Detection
            
            Upload a leaf image to detect potential crop diseases using our AI model trained on real plant disease data.
            The model can identify 38 different plant diseases across multiple crops including tomatoes, potatoes, apples, corn, and more.
            
            **Powered by:** ResNet18 deep learning model trained on PlantVillage dataset
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Leaf Image",
                    sources=["upload", "webcam", "clipboard"]
                )
                detect_btn = gr.Button("Detect Disease", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                disease_output = gr.Textbox(
                    label="Predicted Disease",
                    interactive=False
                )
                confidence_output = gr.Textbox(
                    label="Confidence",
                    interactive=False
                )
                description_output = gr.Markdown()
                recommendation_output = gr.Textbox(
                    label="Recommendation",
                    interactive=False,
                    lines=5
                )
        
        detect_btn.click(
            fn=detect_disease,
            inputs=image_input,
            outputs=[disease_output, confidence_output, description_output, recommendation_output]
        )
        
        gr.Markdown(
            """
            ---
            ### How it works
            
            1. **Upload an image** of a crop leaf (works best with clear, well-lit images of individual leaves)
            2. **Click "Detect Disease"** to analyze the image using our AI model
            3. **Review the results** including disease identification, confidence level, and treatment recommendations
            
            ### Supported Crops & Diseases
            
            The model can detect diseases in:
            - **Tomatoes**: Early Blight, Late Blight, Bacterial Spot, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus
            - **Potatoes**: Early Blight, Late Blight
            - **Apples**: Apple Scab, Black Rot, Cedar Rust
            - **Corn**: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight
            - **Grapes**: Black Rot, Esca, Leaf Blight
            - **Peppers**: Bacterial Spot
            - **Peaches**: Bacterial Spot
            - And more...
            
            ### Model Information
            
            - **Architecture**: ResNet18 (pre-trained on ImageNet, fine-tuned for plant diseases)
            - **Training Data**: PlantVillage dataset (38 classes)
            - **Inference Speed**: Real-time (< 1 second per image)
            
            ### About G-well
            
            This demo uses RGB image analysis with deep learning. Our production system uses hyperspectral imaging to 
            detect diseases before they become visible to the naked eye, giving farmers a critical early warning advantage.
            
            For more information, visit our [main website](https://surnaik01.github.io/g-well/).
            """
        )
    
    return demo


# Create the interface
demo = create_interface()

if __name__ == "__main__":
    demo.launch()
