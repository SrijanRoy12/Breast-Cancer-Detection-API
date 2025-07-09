# Breast-Cancer-Detection-API
An interactive and visually enhanced Machine Learning web app built with Streamlit that predicts whether a tumor is Benign or Malignant using the Breast Cancer Wisconsin dataset.
🎗️ Breast Cancer Prediction App

A Machine Learning-powered web application that predicts the likelihood of breast cancer based on medical input features such as mean radius, texture, perimeter, and more. Built to assist in early detection using data science.

![App Demo](assets/breast_cancer_app_demo.png) <!-- Replace with your actual image path -->

---

## 🚀 Features

- 🧠 Predicts whether a tumor is **benign** or **malignant**
- ⚙️ User-friendly interface built with **Streamlit**
- 🤖 Uses a trained **classification model** (e.g., Random Forest / Logistic Regression)
- 📊 Visualizes model output and input feature importance
- 📦 Lightweight and deployable on Streamlit Cloud

---

## 🛠 Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python, Pandas, NumPy  
- **Machine Learning**: Scikit-learn  
- **Visualization**: Matplotlib, Seaborn  
- **IDE**: Visual Studio Code  

---

## 📂 Project Structure

├── app.py # Main Streamlit app
├── breast_cancer_model.pkl # Trained ML model
├── data.csv # Breast cancer dataset
├── requirements.txt # List of dependencies
├── assets/
│ └── breast_cancer_app_demo.png # Demo screenshot
├── README.md # Project documentation

yaml
Copy
Edit

---

## 📈 Model Details

- **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset from sklearn  
- **Model Used**: Logistic Regression / Random Forest  
- **Metrics**: Accuracy, Precision, Recall, F1-Score  
- **Goal**: Predict `Malignant` or `Benign` status

---

## ▶️ How to Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/your-username/Breast-Cancer-Predictor.git
cd Breast-Cancer-Predictor
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app

bash
Copy
Edit
streamlit run app.py
💡 Use Cases
Assistive diagnostic tool

Learning project for healthcare AI

Awareness campaigns with interactive data science
