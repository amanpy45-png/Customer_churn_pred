# 📊 Customer Churn Prediction

A Machine Learning-powered web application that predicts whether a telecom customer is likely to churn based on customer demographics, account information, and service usage. The project uses a trained Random Forest Classifier and provides predictions through a Flask web interface.

---

## 🚀 Features

- Predicts customer churn in real time
- User-friendly web interface built with Flask
- Data preprocessing using Label Encoding and Standard Scaling
- Random Forest Classifier for prediction
- Displays churn probability along with prediction
- Clean and responsive frontend

---

## 📂 Project Structure

```
customer_churn_predictor/
│
├── app.py                  # Flask application
├── Customer-Churn.ipynb    # Data preprocessing, EDA & model training
├── best_model.pkl          # Trained Random Forest model
├── encoder.pkl             # Saved Label Encoders
├── scaler.pkl              # Saved StandardScaler
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Flask
- Pickle

### Machine Learning

- Random Forest Classifier
- Label Encoding
- StandardScaler

---

## 📈 Dataset

Dataset Used:

**Telco Customer Churn Dataset**

The dataset contains customer demographic information, subscribed services, billing details, and churn status.

Target Variable:

- Churn

Features include:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract Type
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

---

## ⚙️ Machine Learning Workflow

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Label Encoding
4. Feature Scaling
5. Train-Test Split
6. Model Training
7. Hyperparameter Tuning
8. Model Evaluation
9. Model Saving using Pickle
10. Flask Deployment

---

## 📊 Model Performance

Model Used:

- Random Forest Classifier

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

---

## 💻 Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/customer_churn_predictor.git

cd customer_churn_predictor
```

---

### Create Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

---

## 📸 Application Workflow

1. Enter customer details.
2. Click **Predict Churn**.
3. Model preprocesses the inputs.
4. Random Forest predicts churn.
5. Prediction probability is displayed.

---

## 🎯 Future Improvements

- Deploy using Render or Railway
- Docker support
- REST API using FastAPI
- User Authentication
- Feature Importance Visualization
- SHAP Explainability
- Database Integration
- Prediction History Dashboard

---

## 📚 Learning Outcomes

This project demonstrates:

- End-to-End Machine Learning Pipeline
- Feature Engineering
- Data Preprocessing
- Model Serialization
- Flask Deployment
- Frontend-Backend Integration
- Machine Learning Model Deployment

---

## 👨‍💻 Author

**Aman Negi**

- B.Tech Computer Science Engineering
- DIT University
- Aspiring Data Scientist | Machine Learning Enthusiast


LinkedIn: https://www.linkedin.com/in/amanpy45/

---

## ⭐ If you found this project useful, don't forget to star the repository!
