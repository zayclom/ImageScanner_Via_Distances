import streamlit as st
import numpy as np
import os
from features_management import extract_glcm_features, extract_bit_features
from similarity_measures import euclidean_distance, manhattan_distance, chebyshev_distance, canberra_distance

# Load pre-calculated features
@st.cache_data
def load_features(descriptor='glcm'):
    feature_files = []
    for root, _, files in os.walk(f'features/{descriptor}'):
        for file in files:
            if file.lower().endswith('.npy'):
                feature_files.append(os.path.join(root, file))
    
    features = {}
    for file in feature_files:
        image_name = os.path.basename(file).split('_')[0]
        features[image_name] = np.load(file)
    return features

# Find similar images
def find_similar_images(uploaded_image_features, features, distance_measure, num_results):
    distances = []
    for image_name, feature in features.items():
        if distance_measure == 'Euclidean':
            distance = euclidean_distance(uploaded_image_features, feature)
        elif distance_measure == 'Manhattan':
            distance = manhattan_distance(uploaded_image_features, feature)
        elif distance_measure == 'Chebyshev':
            distance = chebyshev_distance(uploaded_image_features, feature)
        elif distance_measure == 'Canberra':
            distance = canberra_distance(uploaded_image_features, feature)
        else:
            raise ValueError("Unsupported distance measure")
        distances.append((image_name, distance))
    distances.sort(key=lambda x: x[1])
    return distances[:num_results]

# Streamlit app
st.title("Image Similarity Finder")

uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png'])
descriptor = st.selectbox("Select Descriptor", ["GLCM", "BiT"])
distance_measure = st.selectbox("Select Distance Measure", ["Euclidean", "Manhattan", "Chebyshev", "Canberra"])
num_results = st.slider("Number of Similar Images to Display", 1, 10, 5)

if uploaded_file is not None:
    file_path = f"uploads/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if descriptor == "GLCM":
        uploaded_image_features = extract_glcm_features(file_path)
    else:
        uploaded_image_features = extract_bit_features(file_path)

    features = load_features(descriptor.lower())
    similar_images = find_similar_images(uploaded_image_features, features, distance_measure, num_results)

    st.write("Similar Images:")
    for image_name, distance in similar_images:
        image_found = False
        for root, _, files in os.walk('dataset/Projet1_Dataset'):
            for file in files:
                if file.startswith(image_name) and file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
                    st.write(f"{image_name} (Distance: {distance})")
                    st.image(os.path.join(root, file), width=100)
                    image_found = True
                    break
            if image_found:
                break
        if not image_found:
            st.write(f"Image {image_name} not found in dataset.")
