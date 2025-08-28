# 📈 Stock Price Prediction

A **Machine Learning project** that predicts stock prices using historical data.  
The system is built with **Python, Flask, and Firebase** for authentication and data storage.

## 🚀 Features
- Predict stock prices using ML models (e.g., LSTM/Random Forest)
- Flask web interface for user input & visualization
- Firebase Authentication (user login/signup)
- Firebase Firestore for saving predictions/history
- Interactive charts for stock trends

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn / TensorFlow / Keras
- **Database & Auth:** Firebase (Firestore + Authentication)
- **Frontend:** HTML, CSS, Bootstrap, Chart.js

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-price-prediction.git
   cd stock-price-prediction
Create a virtual environment and install dependencies:

bash
Copy code
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
Add Firebase configuration:

Create a .env file and add your Firebase details:

env
Copy code
FIREBASE_API_KEY=your_firebase_api_key
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_DATABASE_URL=your_database_url
Keep your service account JSON file locally and never commit it to GitHub.

Run the Flask app:

bash
Copy code
python app.py
Open in browser:

cpp
Copy code
http://127.0.0.1:5000/
