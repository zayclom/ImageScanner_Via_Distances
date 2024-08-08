import cv2
import os
import numpy as np
from features_management import extract_glcm_features, extract_bit_features

def find_similar_images(uploaded_image_features, dataset_path, descriptor):
    bf = cv2.BFMatcher()
    similar_images = []

    # Determine the feature extraction method based on the descriptor
    feature_extraction_function = extract_glcm_features if descriptor == 'glcm' else extract_bit_features

    for root, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
                dataset_image_path = os.path.join(root, filename)
                dataset_image_features = feature_extraction_function(dataset_image_path)

                # Match descriptors
                matches = bf.knnMatch(uploaded_image_features, dataset_image_features, k=2)

                # Apply ratio test
                good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

                if len(good_matches) > 10:
                    similar_images.append({
                        'image': dataset_image_path,  # Store the full path to the image
                        'matches': len(good_matches)
                    })

    # Sort by number of matches
    similar_images.sort(key=lambda x: x['matches'], reverse=True)
    return similar_images
