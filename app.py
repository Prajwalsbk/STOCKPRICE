from flask import Flask, render_template, request, redirect, url_for, flash, session
import yfinance as yf
import joblib
from forex_python.converter import CurrencyRates
import matplotlib
matplotlib.use('Agg')  # Use backend that does not require a GUI
import matplotlib.pyplot as plt
import os
import firebase_admin
from firebase_admin import credentials, auth

# Initialize Firebase Admin SDK
cred = credentials.Certificate('E:\AIML\stock price\stock-price-9e2df-firebase-adminsdk-fbsvc-07ae36e972.json')
firebase_admin.initialize_app(cred)

app = Flask(__name__)
secret_key = os.urandom(24)  # Generates a random 24-byte string
app.secret_key = secret_key

# Load model and scaler
model = joblib.load('lr_stock_model.pkl')  # Load your model
scaler = joblib.load('scaler.pkl')  # Load the scaler

# Get stock summary (Open, High, Low, Close)
def get_stock_summary(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d')
        if not data.empty:
            summary = {
                'open': round(data['Open'].iloc[-1], 2),
                'high': round(data['High'].iloc[-1], 2),
                'low': round(data['Low'].iloc[-1], 2),
                'close': round(data['Close'].iloc[-1], 2),
            }
            return summary
    except Exception as e:
        print(f"Error fetching stock summary for {ticker}: {e}")
    return None

# Convert USD to INR
def convert_usd_to_inr(amount):
    c = CurrencyRates()
    try:
        return round(c.convert('USD', 'INR', amount), 2)
    except:
        return None

# Get current price of the stock
def get_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        todays_data = stock.history(period='1d')
        if not todays_data.empty:
            current_price = todays_data['Close'].iloc[-1]
            return round(current_price, 2)
    except Exception as e:
        print(f"Error fetching current price for {ticker}: {e}")
    return None

# Dashboard Route
@app.route("/", methods=["GET"])
def dashboard():
    # Check if the user is logged in
    if 'user_id' in session:
        return redirect(url_for('prediction'))
    return render_template("dashboard.html")

# Signup Route
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        
        try:
            # Create user in Firebase Authentication
            user = auth.create_user(
                email=email,
                password=password,
            )
            flash("Account created successfully!", "success")
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"Error creating user: {str(e)}", "danger")
    
    return render_template("signup.html")

# Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        
        try:
            # Authenticate user with Firebase
            user = auth.get_user_by_email(email)
            # Use Firebase's ID token to validate the login
            token = auth.create_custom_token(user.uid)  # This is just a custom token
            
            # Save user data in session after login
            session['user_id'] = user.uid
            flash("Logged in successfully!", "success")
            return redirect(url_for('prediction'))
        except auth.UserNotFoundError:
            flash("Invalid email or password", "danger")
        except Exception as e:
            flash(f"Error: {str(e)}", "danger")
    
    return render_template("login.html")

# Prediction Route (Protected)
@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if 'user_id' not in session:
        flash("Please log in to access the prediction page.", "danger")
        return redirect(url_for('login'))

    ticker = None
    prediction = None
    stock_summary = None
    inr_price = None
    current_price = None
    plot_url = None
    error_message = None

    if request.method == "POST":
        ticker = request.form['ticker'].upper()  # Get ticker from form
        
        # Fetch stock data
        stock_summary = get_stock_summary(ticker)
        current_price = get_current_price(ticker)

        if stock_summary is None or current_price is None:
            error_message = f"Unable to fetch data for stock ticker: {ticker}. Please check the ticker and try again."
        else:
            if stock_summary:
                inr_price = convert_usd_to_inr(stock_summary['close'])

            # Prepare model input (Last 60 days for prediction)
            data = yf.download(ticker, start='2015-01-01')
            close_prices = data['Close'].values.reshape(-1, 1)
            scaled = scaler.transform(close_prices)
            last_60 = scaled[-60:].reshape(1, -1)
            pred_scaled = model.predict(last_60)[0]
            pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
            prediction = f"${pred_price:.2f}"

            # Plot actual vs predicted price
            plt.figure(figsize=(10, 4))
            plt.plot(close_prices[-100:], label='Actual Price (Last 100 Days)', color='blue')
            plt.plot(len(close_prices)-1, pred_price, marker='o', markersize=8, label='Predicted Price', color='red')
            plt.title(f"{ticker} - Actual vs Predicted")
            plt.xlabel("Days")
            plt.ylabel("Price ($)")
            plt.legend()
            plot_path = os.path.join('static', 'plot.png')
            plt.savefig(plot_path)
            plt.close()
            plot_url = plot_path

    return render_template("index.html", ticker=ticker, 
                           stock_summary=stock_summary, 
                           inr_price=inr_price, 
                           prediction=prediction, 
                           current_price=current_price, 
                           plot_url=plot_url, 
                           error_message=error_message)

# Logout Route
@app.route("/logout")
def logout():
    session.pop('user_id', None)  # Remove user from session
    flash("You have been logged out", "info")
    return redirect(url_for('dashboard'))

if __name__ == "__main__":
    app.run(debug=True)
