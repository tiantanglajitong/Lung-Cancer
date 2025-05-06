import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os
import json
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Lung Cancer CT Image Prediction System",
    page_icon="🏥",
    layout="wide"
)

def load_model_complete(save_dir='model_data'):
    """
    Load the model along with its training history and configuration.
    """
    try:
        # Load model
        model_path = os.path.join(save_dir, 'ConvNext_CLAHE.h5')
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")

        # Load training history
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'r') as f:
            history = json.load(f)
        print("Training history loaded successfully.")

        # Load configuration
        config_path = os.path.join(save_dir, 'model_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Configuration loaded successfully.")

        return model, history, config

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Load model at the start
model, history, config = load_model_complete()

def apply_clahe(image):
    """
    Apply CLAHE to enhance contrast in the grayscale image.
    """
    if len(image.shape) == 2:  # 如果是灰度图，直接处理
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if gray_image.dtype != np.uint8:
        gray_image = np.uint8(gray_image)

    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)
    
    return cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)  # 转回 RGB

def preprocess_image(image):
    """
    Preprocess input image to match training preprocessing.
    """
    img = np.array(image)

    # 确保 3 通道格式
    if img.shape[-1] == 4:
        img = img[..., :3]

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, (224, 224))

    # 应用 CLAHE
    img = apply_clahe(img)

    # 归一化
    # img = img.astype('float32') / 255.0

    # 增加 batch 维度
    img = np.expand_dims(img, axis=0)

    return img


def main():
    # Page title
    st.title("Lung Cancer CT Image Prediction System")
    st.write("Upload a lung CT image for prediction")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("This system uses a deep learning model to analyze lung CT images and predict whether lung cancer is present.")
        st.write("Supported image formats: JPG, PNG")
        
        st.header("How to Use")
        st.write("1. Upload a CT image")
        st.write("2. Wait for system analysis")
        st.write("3. View prediction results")

        # Show model configuration details
        if config:
            st.subheader("Model Information")
            st.json(config)  # Display model config JSON

    # File upload
    uploaded_file = st.file_uploader("Select a CT image file", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded CT Image", use_column_width=True)

        # Preprocess image
        processed_img = preprocess_image(image)

        with col2:
            st.subheader("Enhanced Image")
            enhanced_img = apply_clahe(np.array(image))
            st.image(enhanced_img, caption="CLAHE Enhanced Image", use_column_width=True)

        if model:
            # Add prediction button
            if st.button("Start Prediction"):
                with st.spinner('Analyzing the image...'):
                    prediction = model.predict(processed_img)
                    predicted_class = np.argmax(prediction[0])
                    confidence = float(prediction[0][predicted_class])

                    # Display prediction result
                    st.subheader("Prediction Result")
                    
                    # Define class labels
                    class_names = ['Benign', 'Malignant', 'Normal']
                    result = class_names[predicted_class]
                    
                    # Set result color
                    if result == 'Normal':
                        color = 'green'
                    elif result == 'Benign':
                        color = 'orange'
                    else:
                        color = 'red'
                    
                    # Display result
                    st.markdown(f"<h3 style='color: {color};'>Predicted Class: {result}</h3>", 
                              unsafe_allow_html=True)
                    st.write(f"Confidence: {confidence:.2%}")

                    # Display probability distribution
                    st.subheader("Prediction Probabilities")
                    for i, (label, prob) in enumerate(zip(class_names, prediction[0])):
                        st.progress(float(prob))
                        st.write(f"{label}: {prob:.2%}")

if __name__ == "__main__":
    main()


