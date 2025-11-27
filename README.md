# 🌱 Plant Disease Detection App

This project is a simple **Streamlit web app** that detects plant diseases from leaf images.
It uses a **deep learning model** trained on 38 types of plant diseases.

## 🚀 Features

* Upload a leaf image and get **top 5 disease predictions**
* Shows **percentage accuracy** for each prediction
* Highlights the **diseased areas** using image segmentation
* Easy-to-use interface with Home, About, and Disease Recognition pages

## 🛠️ Technologies Used

* **Python**
* **TensorFlow / Keras**
* **Streamlit**
* **OpenCV**
* **NumPy**
* **Pillow (PIL)**

## ▶️ How to Run

1. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:

   ```bash
   streamlit run app.py
   ```
3. Open the link shown in the terminal (usually `http://localhost:8501`).

## 📸 How It Works

1. You upload a leaf image.
2. The model predicts the disease.
3. The app shows:

   * Top 5 disease names
   * Confidence percentages
   * Segmented image showing infected areas

## 📁 Files

* `app.py` → Main Streamlit app
* `plant_disease_model.keras` → Trained model
* `requirements.txt` → Dependencies

## ✔️ Purpose
Helps farmers and students quickly identify plant diseases and take action to reduce crop loss.


If you want an even **shorter** version, tell me!
