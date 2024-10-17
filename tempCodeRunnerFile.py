from flask import Flask, request, render_template, make_response
import mysql.connector
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import os
import logging
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# MySQL Connection
def db_connection():
    try:
        conn = mysql.connector.connect(
            host="127.0.0.7",
            user="root",
            password=os.getenv('DB_PASSWORD', 'password01'),  # Use environment variable for password
            database="pharmacy_management"
        )
        return conn
    except mysql.connector.Error as err:
        logging.error(f"Database connection error: {err}")
        return None

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route for Demand Prediction Form
@app.route('/predict_demand', methods=['GET', 'POST'])
def predict_demand():
    if request.method == 'POST':
        drug_name = request.form.get('drug_name', '').strip()
        if not drug_name:
            error = "Drug name is required."
            return render_template('predict.html', error=error)

        conn = db_connection()
        if conn is None:
            error = "We're currently unable to access our database. Please try again later."
            return render_template('predict.html', error=error)

        cursor = conn.cursor(dictionary=True)

        # Fetch sales data from the database
        try:
            cursor.execute(""" 
                SELECT sale_date, drug_name, SUM(quantity_sold) AS total_sold 
                FROM medication_sales 
                WHERE drug_name = %s
                GROUP BY sale_date, drug_name
            """, (drug_name,))
            data = cursor.fetchall()
            logging.debug(f"Fetched data for {drug_name}: {data}")

        except Exception as e:
            logging.error(f"Error fetching sales data: {e}")
            error = "Error fetching sales data."
            return render_template('predict.html', error=error)
        finally:
            cursor.close()
            conn.close()

        # Convert to DataFrame for ARIMA
        df = pd.DataFrame(data)
        if df.empty:
            error = "No sales data found."
            return render_template('predict.html', error=error)

        try:
            df['sale_date'] = pd.to_datetime(df['sale_date'])
            df.sort_values('sale_date', inplace=True)
            df.set_index('sale_date', inplace=True)
            df = df.asfreq('D').fillna(0)
        except Exception as e:
            logging.error(f"Error processing DataFrame: {e}")
            error = "Error processing sales data."
            return render_template('predict.html', error=error)

        drug_sales = df['total_sold']

        # Check if there is enough data for ARIMA
        if len(drug_sales) < 5:
            error = "Insufficient data for ARIMA model."
            return render_template('predict.html', error=error)

        # Fit ARIMA model
        try:
            drug_sales = pd.to_numeric(drug_sales, errors='coerce').dropna()
            logging.debug(f"Drug sales data after conversion: {drug_sales}")

            model = ARIMA(drug_sales, order=(5, 1, 0))
            model_fit = model.fit()

            # Forecast the next 30 days
            forecast = model_fit.forecast(steps=30)
            forecast_list = forecast.tolist()
            logging.debug(f"Predicted demand for {drug_name}: {forecast_list}")

            # Render the prediction results on the frontend
            return render_template('predict.html', drug_name=drug_name, forecast=forecast_list)

        except Exception as e:
            logging.error(f"Error fitting model: {str(e)}")
            error = "Error fitting ARIMA model."
            return render_template('predict.html', error=error)

    # If GET request, render the form
    return render_template('predict.html')

# Route to plot and display the forecast graph
@app.route('/predict_demand_with_graph', methods=['POST'])
def predict_demand_with_graph():
    drug_name = request.form.get('drug_name', '').strip()
    if not drug_name:
        error = "Drug name is required."
        return render_template('predict.html', error=error)

    conn = db_connection()
    if conn is None:
        error = "We're currently unable to access our database. Please try again later."
        return render_template('predict.html', error=error)

    cursor = conn.cursor(dictionary=True)

    # Fetch sales data from the database
    try:
        cursor.execute(""" 
            SELECT sale_date, drug_name, SUM(quantity_sold) AS total_sold 
            FROM medication_sales 
            WHERE drug_name = %s
            GROUP BY sale_date, drug_name
        """, (drug_name,))
        data = cursor.fetchall()
    except Exception as e:
        logging.error(f"Error fetching sales data for plotting: {e}")
        return render_template('predict.html', error="Error fetching sales data for plotting.")
    finally:
        cursor.close()
        conn.close()

    # Convert to DataFrame for ARIMA
    df = pd.DataFrame(data)
    if df.empty:
        return render_template('predict.html', error="No sales data found for plotting.")

    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df.sort_values('sale_date', inplace=True)
    df.set_index('sale_date', inplace=True)
    df = df.asfreq('D').fillna(0)

    drug_sales = df['total_sold']

    # Fit ARIMA model
    try:
        drug_sales = pd.to_numeric(drug_sales, errors='coerce').dropna()
        model = ARIMA(drug_sales, order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast the next 30 days
        forecast = model_fit.forecast(steps=30)
    except Exception as e:
        logging.error(f"Error fitting ARIMA model for plotting: {e}")
        return render_template('predict.html', error="Error fitting ARIMA model for plotting.")

    # Generate the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(drug_sales.index, drug_sales, label='Historical Sales')
    future_dates = pd.date_range(start=drug_sales.index[-1] + timedelta(days=1), periods=30)
    ax.plot(future_dates, forecast, label='Forecasted Sales', linestyle='--')
    ax.set_title(f"Sales Forecast for {drug_name}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Quantity Sold')
    ax.legend()

    # Convert the plot to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)

    # Convert the plot to a base64 string
    img = base64.b64encode(output.getvalue()).decode('utf-8')
    
    # Render the prediction results on the frontend with graph
    return render_template('predict.html', drug_name=drug_name, graph_url=img)

# Route for Expiration Notifications
@app.route('/check_expiry', methods=['GET'])
def check_expiry():
    conn = db_connection()
    if conn is None:
        error = "We're currently unable to access our database. Please try again later."
        return render_template('expiry.html', error=error)

    cursor = conn.cursor(dictionary=True)

    # Fetch drugs with expiry dates within the next 10 days
    today = datetime.today()
    notification_date = today + timedelta(days=10)

    try:
        cursor.execute("SELECT drug_name, expiry_date FROM medication_stock WHERE expiry_date <= %s", (notification_date,))
        expiring_drugs = cursor.fetchall()
        logging.debug(f"Fetched expiring drugs: {expiring_drugs}")

    except Exception as e:
        logging.error(f"Error fetching expiring drugs: {e}")
        error = "Error fetching expiring drugs."
        return render_template('expiry.html', error=error)
    finally:
        cursor.close()
        conn.close()

    # Render the expiring drugs on the frontend
    return render_template('expiry.html', expiring_drugs=expiring_drugs)

if __name__ == '__main__':
    app.run(debug=True)
