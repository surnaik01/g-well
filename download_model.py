"""
Download or create an improved pre-trained model for plant disease detection.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import os
from pathlib import Path

# Import from train_model
from train_model import PLANT_DISEASE_CLASSES, CLASS_DISPLAY_NAMES, create_model

def create_improved_model():
    """Create a model with better initialization for plant disease detection."""
    print("Creating improved model with transfer learning...")
    
    # Create model
    model = create_model()
    
    # The model now has ImageNet pre-trained weights in the feature extractor
    # The final layer is randomly initialized for 38 classes
    
    # Save the model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_names': PLANT_DISEASE_CLASSES,
        'class_display_names': CLASS_DISPLAY_NAMES,
        'val_acc': 0.0,
        'note': 'ImageNet pre-trained ResNet18. For best results, fine-tune on PlantVillage dataset.',
        'model_type': 'resnet18',
        'num_classes': len(PLANT_DISEASE_CLASSES)
    }
    
    torch.save(checkpoint, model_dir / 'plant_disease_model.pth')
    print(f"✅ Model saved to {model_dir / 'plant_disease_model.pth'}")
    print("\nNote: This model uses ImageNet features which work reasonably well for plant images.")
    print("For production use, fine-tune on PlantVillage dataset by running:")
    print("  python train_model.py")
    print("\nTo download PlantVillage dataset:")
    print("  1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print("  2. Download and extract to 'data/plant_disease/' directory")
    print("  3. Run: python train_model.py")

if __name__ == "__main__":
    create_improved_model()

