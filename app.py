import requests
from PIL import Image
import numpy as np
import pandas as pd
import io
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed data
with open('image_names.pkl', 'rb') as file:
    image_names = pickle.load(file)

with open('features_matrix.pkl', 'rb') as file:
    features_matrix = pickle.load(file)

image_df = pd.read_csv('images.csv')

# VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_feature_from_image(img, model):
    """Extract features from a single image using the model."""
    img = preprocess_image(img)
    features = model.predict(img)
    return features.reshape(-1)  # Flatten the feature vector

def preprocess_image(img, target_size=(224, 224)):
    """Preprocess a single image for feature extraction."""
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def find_similar_images(custom_image_features, features_matrix, top_n=5):
    """Find the most similar images to the custom image."""
    similarities = cosine_similarity([custom_image_features], features_matrix)
    most_similar_indices = np.argsort(similarities[0])[::-1][:top_n]
    return most_similar_indices

def recommend_similar_images(img):
    """Recommend similar images based on the uploaded image."""
    query_feature = extract_feature_from_image(img, model)
    similar_indices = find_similar_images(query_feature, features_matrix, top_n=5)
    similar_images = [image_df['link'][image_df['filename'] == image_names[idx]].values[0] for idx in similar_indices]
    return similar_images

def get_recommendations_from_image(uploaded_image):
    """Recommend similar images based on the uploaded image."""
    img = uploaded_image
    similar_image_urls = recommend_similar_images(img)
    similar_images = [Image.open(requests.get(img_url, stream=True).raw) for img_url in similar_image_urls]
    return similar_images

# Gradio interface
image_input = gr.Image(type="pil", label="Upload Image")
outputs = gr.Gallery(label="Recommended Images", columns=3)  # Set the number of columns to fit images better

gr.Interface(fn=get_recommendations_from_image, inputs=image_input, outputs=outputs, examples=['shirt.jpg','perfume.jpg','shoes.jpg','kurti.jpg','watches.jpg','heels.jpg','necklace.jpg'],
             title="Fashion Image Recommender").launch()
