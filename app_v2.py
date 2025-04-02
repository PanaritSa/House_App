import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import numpy as np
import os
import cv2
from PIL import Image
import pickle
import tensorflow

# Add CSS for layout adjustments
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
    }
    .stApp {
        background-color: #d3d3d3;
    }
    h1 {
        font-family: Arial, sans-serif;
        color: #333;
        margin-left: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Top layout with image
st.image("H_vector.jpg", width=800)

# Main content inside styled div
st.markdown('<div class="main">', unsafe_allow_html=True)

# Load models and data functions
def initialize_classification_model():
    return load_model('best_model13cls.keras')

def initialize_recommendation_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return tensorflow.keras.Sequential([base_model, GlobalMaxPooling2D()])

def load_image_data():
    dir = os.path.join(os.getcwd(), 'dataset')
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    data_set = datagen.flow_from_directory(dir, target_size=(224, 224), batch_size=32, class_mode='sparse')
    labels = dict((v, k) for k, v in data_set.class_indices.items())
    return labels

def load_feature_data():
    try:
        feature_list = np.array(pickle.load(open("featurevector.pkl", "rb")))
        filenames = pickle.load(open("filenames.pkl", "rb"))
        st.write("Loaded feature vector and filenames successfully.")
        return feature_list, filenames
    except Exception as e:
        st.write(f"Error loading files: {e}")
        st.stop()

st.title("Image Classification and Recommendation System")
st.write("Upload an image to classify it, or select a house style from the dropdown to view related houses.")

# Load models and data
model_classification = initialize_classification_model()
model_recommendation = initialize_recommendation_model()
labels = load_image_data()
feature_list, filenames = load_feature_data()

# Dropdown list for house styles
house_styles = ["Select House Style"] + [value.replace('ML-AR-', '') for value in labels.values()]
selected_class = st.selectbox("Select a house style:", options=house_styles)

if selected_class and selected_class != "Select House Style":
    st.write(f"Displaying houses for style: {selected_class}")
    matched_files = [file for file in filenames if selected_class.lower() in file.lower()]

    if matched_files:
        # Pagination logic
        page = st.number_input("Page", min_value=1, max_value=(len(matched_files) - 1) // 5 + 1, step=1, value=1)
        start = (page - 1) * 5
        end = start + 5

        for file in matched_files[start:end]:
            st.image(file, caption=os.path.basename(file), use_container_width=True)
            if st.button(f"Analyze {os.path.basename(file)}"):
                # Perform analysis on the selected image
                img = cv2.imread(file)
                img = cv2.resize(img, (224, 224))
                img_array = np.expand_dims(img, axis=0)
                img_array = preprocess_input(img_array)
                pred = model_classification.predict(img_array)
                pred_probabilities = pred[0]
                top_indices = pred_probabilities.argsort()[-3:][::-1]
                top_labels = [(labels[i].replace('ML-AR-', ''), pred_probabilities[i]) for i in top_indices]

                # Display barchart for analysis
                st.write("Prediction Scores:")
                sorted_top_labels = sorted(top_labels, key=lambda x: x[1], reverse=True)
                chart_labels = [label for label, _ in sorted_top_labels]
                chart_scores = [score * 100 for _, score in sorted_top_labels]

                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.barh(chart_labels, chart_scores, color='skyblue', height=0.4)

                for bar, score in zip(bars, chart_scores):
                    ax.text(score + 1, bar.get_y() + bar.get_height()/2, f'{score:.2f}%', va='center')

                ax.set_title("Top 3 Predictions")
                ax.set_xlabel("Score (%)")
                ax.set_xlim(0, 100)
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                ax.invert_yaxis()
                st.pyplot(fig)
    else:
        st.write("No images found for this style.")

# Input Method Toggle
input_method = st.radio("Select Input Method:", ("Upload Image", "Use Camera"))

# Determine the source of the image
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image_source = uploaded_file
else:
    camera_file = st.camera_input("Take a picture")
    image_source = camera_file

if image_source is not None:
    # Handle image input
    if input_method == "Upload Image":
        image_bytes = image_source.read()
    else:  # Camera input
        image_bytes = image_source.getvalue()

    st.image(image_bytes, caption='Input Image', use_container_width=True)

    # Save the image temporarily for processing
    with open("temp_image.jpg", "wb") as f:
        f.write(image_bytes)

    # Load and process the image for classification
    st.write("Classifying the image...")
    img = load_img("temp_image.jpg", target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    pred = model_classification.predict(img_array)
    pred_probabilities = pred[0]
    top_indices = pred_probabilities.argsort()[-3:][::-1]
    top_labels = [(labels[i].replace('ML-AR-', ''), pred_probabilities[i]) for i in top_indices]

    # Horizontal Bar Chart for Top Predictions
    st.write("Prediction Scores:")
    sorted_top_labels = sorted(top_labels, key=lambda x: x[1], reverse=True)
    chart_labels = [label for label, _ in sorted_top_labels]
    chart_scores = [score * 100 for _, score in sorted_top_labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(chart_labels, chart_scores, color='skyblue', height=0.4)

    for bar, score in zip(bars, chart_scores):
        ax.text(score + 1, bar.get_y() + bar.get_height()/2, f'{score:.2f}%', va='center')

    ax.set_title("Top 3 Predictions")
    ax.set_xlabel("Score (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.invert_yaxis()
    st.pyplot(fig)

    # Part 2: Image Recommendation
    st.write("Finding similar images...")

    def extract_feature(img_path, model):
        img = cv2.imread(img_path)
        if img is None:
            st.write("Error: Could not read the image with OpenCV.")
            return None
        img = cv2.resize(img, (224, 224))
        expand_img = np.expand_dims(img, axis=0)
        pre_img = preprocess_input(expand_img)
        result = model.predict(pre_img).flatten()
        normalized = result / norm(result)
        return normalized

    features = extract_feature("temp_image.jpg", model_recommendation)
    if features is not None:
        def recommend(features, feature_list):
            neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
            neighbors.fit(feature_list)
            _, indices = neighbors.kneighbors([features])
            return indices

        indices = recommend(features, feature_list)
        if indices is not None:
            st.write("Recommended Images:")
            col1, col2, col3, col4, col5 = st.columns(5)
            for i, col in enumerate([col1, col2, col3, col4, col5]):
                try:
                    image_path = filenames[indices[0][i]]
                    if image_path:
                        folder_name = os.path.basename(os.path.dirname(image_path))
                        with col:
                            st.image(image_path, caption=folder_name, use_container_width=True)
                    else:
                        st.write("Error: Invalid image path.")
                except IndexError:
                    st.write("Error: Not enough recommended images to display.")
    else:
        st.write("Feature extraction failed. No recommendations available.")

st.markdown('</div>', unsafe_allow_html=True)
