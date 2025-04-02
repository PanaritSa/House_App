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
import keras
from keras import layers
from scipy.io import loadmat
import matplotlib.pyplot as plt

def initialize_classification_model():
    return load_model('best_model13cls.keras')

def initialize_recommendation_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return keras.Sequential([base_model, GlobalMaxPooling2D()])

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
        return feature_list, filenames
    except Exception as e:
        st.write(f"Error loading files: {e}")
        st.stop()

def save_uploaded_file(uploaded_file):
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    file_path = os.path.join('uploads', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_feature(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    _, indices = neighbors.kneighbors([features])
    return indices

def load_segmentation_model():
    return load_model("segmentation_model.keras")

def read_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    return np.array(img) / 255.0

def segment_image(image, model):
    img_tensor = np.expand_dims(image, axis=0)
    predictions = model.predict(img_tensor)
    predictions = np.squeeze(predictions)
    mask = np.argmax(predictions, axis=2)
    return mask

def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def get_overlay(image, colored_mask):
    image = keras.utils.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay

def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(keras.utils.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()

# Load colormap
colormap = loadmat("colormap_13_classes.mat")["colormap"]
colormap = (colormap * 100).astype(np.uint8)

# Streamlit App
st.title("Image Classification, Recommendation, and Semantic Segmentation")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    model_classification = initialize_classification_model()
    model_recommendation = initialize_recommendation_model()
    labels = load_image_data()
    feature_list, filenames = load_feature_data()
    segmentation_model = load_segmentation_model()

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Classification
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

    # Recommendation
    file_path = save_uploaded_file(uploaded_file)
    features = extract_feature(file_path, model_recommendation)
    indices = recommend(features, feature_list)
    st.write("Recommended Images:")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        try:
            image_path = filenames[indices[0][i]]
            folder_name = os.path.basename(os.path.dirname(image_path))
            with col:
                st.image(image_path, caption=folder_name)
        except IndexError:
            pass

    # Segmentation
    img = read_image(uploaded_file)
    segmentation_mask = segment_image(img, segmentation_model)
    colored_mask = decode_segmentation_masks(segmentation_mask, colormap, 13)
    overlay = get_overlay(img, colored_mask)
    st.image(overlay, caption="Segmented Overlay", use_column_width=True)
