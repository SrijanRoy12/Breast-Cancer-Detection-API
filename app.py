import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import base64
import io

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split and train models
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "SVM (RBF Kernel)": SVC(probability=True)
}

for name in models:
    models[name].fit(X_train, y_train)

# Streamlit App
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")
st.title("🔬 Breast Cancer Detection App")

# Sidebar
st.sidebar.title("🔧 App Controls")
model_choice = st.sidebar.selectbox("Choose Classifier", list(models.keys()))
example_button = st.sidebar.button("📋 Autofill with Example Patient")

# Input Features
st.subheader("🧪 Input All 30 Features")
user_input = []

cols = st.columns(3)
for i, feature in enumerate(feature_names):
    min_val = float(X[:, i].min())
    max_val = float(X[:, i].max())
    mean_val = float(X[:, i].mean())
    val = cols[i % 3].slider(feature, min_val, max_val, mean_val, step=(max_val - min_val) / 100)
    user_input.append(val)

user_input = np.array(user_input).reshape(1, -1)
input_scaled = scaler.transform(user_input)

# Autofill
if example_button:
    user_input = data.data[0].reshape(1, -1)
    input_scaled = scaler.transform(user_input)
    st.success("🔁 Example data loaded. You can still adjust sliders if needed.")

# Predict
if st.button("🔍 Predict"):
    model = models[model_choice]
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    # Display result
    if prediction == 0:
        st.error("⚠️ The tumor is likely **Malignant** (cancerous).")
    else:
        st.success("✅ The tumor is likely **Benign** (non-cancerous).")

    # Probability pie chart
    st.subheader("📈 Prediction Confidence")
    fig = px.pie(
        names=["Malignant", "Benign"],
        values=[probabilities[0]*100, probabilities[1]*100],
        color_discrete_sequence=["crimson", "lightgreen"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bar chart for input
    st.subheader("📊 Input Feature Distribution")
    input_df = pd.DataFrame(user_input, columns=feature_names)
    st.bar_chart(input_df.T)

    # Classification report
    y_pred = model.predict(X_test)
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, target_names=data.target_names, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    # Confusion matrix
    st.subheader("🧮 Confusion Matrix")
    matrix = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(matrix, index=["Malignant", "Benign"], columns=["Predicted Malignant", "Predicted Benign"])
    st.dataframe(cm_df)

    # Save to session history
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "Input": user_input.tolist(),
        "Model": model_choice,
        "Prediction": "Benign" if prediction == 1 else "Malignant",
        "Confidence": round(probabilities[1]*100 if prediction == 1 else probabilities[0]*100, 2)
    })

    # Download report
    st.subheader("📥 Download Report")
    report_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": user_input.flatten()
    })
    report_df["Model"] = model_choice
    report_df["Prediction"] = "Benign" if prediction == 1 else "Malignant"
    report_df["Confidence (%)"] = round(probabilities[1]*100 if prediction == 1 else probabilities[0]*100, 2)

    csv = report_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction_report.csv">📄 Download CSV Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# Prediction History
if "history" in st.session_state and len(st.session_state.history) > 0:
    st.markdown("---")
    st.subheader("🕓 Prediction History (This Session)")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df)
