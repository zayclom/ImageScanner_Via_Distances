import os
import cv2
import numpy as np
import timm
import torch
from torchvision import transforms
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image_path):
    """Extract GLCM features from a grayscale image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    glcm = graycomatrix(image, distances=[5], angles=[0], symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = [graycoprops(glcm, prop).flatten() for prop in props]
    return np.hstack(features)

def extract_bit_features(image_path):
    """Extract features using a BiT model from the timm library."""
    # Load BiT model from timm
    model = timm.create_model('vit_base_patch16_224', pretrained=True)  # Replace with the specific BiT model variant
    model.eval()  # Set model to evaluation mode
    
    # Define image transformations
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization
    ])
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_tensor = preprocess(image_rgb).unsqueeze(0)  # Add batch dimension

    # Extract features
    with torch.no_grad():
        features = model(image_tensor)
    
    return features.flatten().numpy()

def extract_features(image_path, descriptor='glcm'):
    """Extract features based on the selected descriptor."""
    if descriptor == 'glcm':
        return extract_glcm_features(image_path)
    elif descriptor == 'bit':
        return extract_bit_features(image_path)
    else:
        raise ValueError("Unsupported descriptor")
