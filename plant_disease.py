import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# TensorFlow Model Prediction
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

# Prediction function
def model_prediction(test_image):
    model = load_model()
    image = Image.open(test_image)
    image = image.resize((128, 128))
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)

    top_per_5 = np.sort(predictions).flatten()[::-1]
    top_5 = np.argsort(predictions).flatten()[::-1]

    class_names = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
                   'Blueberry__healthy', 'Cherry(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)__healthy', 'Corn(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)__Common_rust', 'Corn_(maize)__Northern_Leaf_Blight', 'Corn(maize)___healthy',
                   'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
                   'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
                   'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy',
                   'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy',
                   'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew',
                   'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot',
                   'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold',
                   'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
                   'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                   'Tomato___healthy']

    disease = top_5[:5]
    percentage = top_per_5[:5] * 100
    title_disease = "\n".join([f"{class_names[disease[i]]}: {percentage[i]:.2f}%\n" for i in range(5)])
    return title_disease

# Image segmentation
def segment_diseased_area(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] != 3 else image
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_bound = np.array([10, 50, 50])
    upper_bound = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)
    mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    combined_image = np.hstack((image, mask_3_channel, contour_image))
    return combined_image

# Streamlit UI
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = r"C:\Users\mithe\Desktop\Mitheel crop disease prediction\picture1.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("Welcome to the Plant Disease Recognition System! 🌿🔍")

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset contains images of healthy and diseased crop leaves categorized into 38 different classes.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)
        if st.button("Predict"):
            prediction_result = model_prediction(test_image)
            st.write("Model Prediction Result: \n", prediction_result)
            image = Image.open(test_image)
            image = np.array(image)
            combined_image = segment_diseased_area(image)
            st.image(combined_image, caption="Diseased Areas", use_container_width=True)
#streamlit run your_script_name.pystr