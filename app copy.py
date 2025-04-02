# import streamlit as st
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# for dirname, _, filenames in os.walk('D:\Real Estate\_PLP-ML-AI\House_data\dataset'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# dir = r'D:\Real Estate\_PLP-ML-AI\House_data\dataset'
# datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
# data_set = datagen.flow_from_directory(dir, target_size = (224,224), batch_size = 32, class_mode = 'sparse')
# ids, counts = np.unique(data_set.classes, return_counts = True)
# labels = (data_set.class_indices)
# labels = dict((v,k) for k,v in labels.items())

# # Load the pre-trained model (replace 'best_model13cls.keras' with your model file)
# model = load_model('best_model13cls.keras')


# # Assuming `labels` is defined as a list of class names
# # labels = ["Class1", "Class2", "Class3", ..., "Class13"]  # Replace with actual class names

# # Streamlit app
# st.title("Image Classification with ResNet50")
# st.write("Upload an image to classify it into one of 13 categories.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.image(uploaded_file, caption='Uploaded Image', use_column_wipython dth=True)
#     st.write("")
#     st.write("Classifying...")

#     # Load and preprocess the image
#     img = load_img(uploaded_file, target_size=(224, 224))  # Adjust target size as per your model input
#     img = img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = preprocess_input(img)

#     # Make predictions
#     pred = model.predict(img)
#     pred_probabilities = pred[0]

#     # Get the top 3 predictions
#     top_indices = pred_probabilities.argsort()[-3:][::-1]  # Indices of top 3 predictions
#     top_labels = [(labels[i], pred_probabilities[i]) for i in top_indices]  # Using `labels` list

#     # Display the top 3 predictions with their probabilities
#     st.write("Top 3 Predictions:")
#     for label, probability in top_labels:
#         st.write(f"{label}: {round(probability * 100, 2)}%")
    
#     # Get the predicted class (highest probability)
#     pred_cls = labels[np.argmax(pred, -1)[0]]
#     st.write('Prediction:', pred_cls)

import streamlit as st
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

# Load models and data functions
def initialize_classification_model():
    # Load the pre-trained classification model
    return load_model('best_model13cls.keras')

def initialize_recommendation_model():
    # Initialize the ResNet50-based feature extraction model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return tensorflow.keras.Sequential([base_model, GlobalMaxPooling2D()])

def load_image_data():
    # dir = r'D:\Real Estate\House_Style_Project\House_style_ML\dataset'
    dir = os.path.join(os.getcwd(), 'dataset')
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    data_set = datagen.flow_from_directory(dir, target_size=(224, 224), batch_size=32, class_mode='sparse')
    ids, counts = np.unique(data_set.classes, return_counts=True)
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

# Streamlit App
st.title("Image Classification and Recommendation System")
st.write("Upload an image to classify it and receive similar image recommendations.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load models and data
    model_classification = initialize_classification_model()
    model_recommendation = initialize_recommendation_model()
    labels = load_image_data()
    feature_list, filenames = load_feature_data()

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    # Part 1: Image Classification
    st.write("Classifying the image...")
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    pred = model_classification.predict(img_array)
    pred_probabilities = pred[0]
    top_indices = pred_probabilities.argsort()[-3:][::-1]
    top_labels = [(labels[i], pred_probabilities[i]) for i in top_indices]
    st.write("Top 3 Predictions:")
    for label, probability in top_labels:
        st.write(f"{label}: {round(probability * 100, 2)}%")
    pred_cls = labels[np.argmax(pred, -1)[0]]
    st.write('Prediction:', pred_cls)

    # Part 2: Image Recommendation
    st.write("Finding similar images...")
    
    # Save uploaded file for feature extraction
    def save_uploaded_file(uploaded_file):
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path

    file_path = save_uploaded_file(uploaded_file)

    # Feature Extraction for Recommendation
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

    features = extract_feature(file_path, model_recommendation)
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
                    if image_path:  # Check if image_path is valid
                        folder_name = os.path.basename(os.path.dirname(image_path))
                        with col:
                            st.image(image_path, caption=folder_name)
                    else:
                        st.write("Error: Invalid image path.")
                except IndexError:
                    st.write("Error: Not enough recommended images to display.")
    else:
        st.write("Feature extraction failed. No recommendations available.")



