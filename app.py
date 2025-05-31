from flask import Flask, render_template_string, request
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
import io
import base64
app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<title>Stock Price Forecast - By Ibraheem Malik </title>
<style>
    body { font-family: Arial, sans-serif; }
    table.dataframe {
        border-collapse: collapse;
        width: 60%%;
        margin-bottom: 20px;
    }
    table.dataframe th, table.dataframe td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: right;
    }
    table.dataframe th {
        background-color: #f2f2f2;
        color: #333;
    }
    h2, h3 { color: #2c3e50; }
</style>
<h2 style="color: blue;">Stock Price Forecast</h2>
<form method="post">
    Please enter Stock Ticker: <input name="ticker" value="{{ ticker or '' }}">
    <input type="submit" value="Forecast">
</form>
{% if error %}
    <p><b style="color:red;">{{ error }}</b></p>
{% endif %}
{% if forecast_df is not none %}
    <h3>{{ ticker }} Price Forecast for Next Week:</h3>
    {{ forecast_df.to_html(index=False, classes="dataframe") | safe }}
    <img src="data:image/png;base64,{{ plot_url }}" style="max-width:100%%;">
{% endif %}
<br></br>
<br>
<br>
<br>
<footer style="color: #89CFF0;"><b> This page was created by Ibraheem Malik - <a href="https://www.linkedin.com/in/ibraheem-malik/"> Connect with me on Linkedin! </a> </b> <br> <img src="https://media.licdn.com/dms/image/v2/D5603AQEWvDn1nTv_Cw/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1715194598425?e=1753920000&v=beta&t=ma6FaoxlusnoSZfeLSBGscT1Gr745pWXyaE_TBeYPXc" style="width:50px;"> </footer>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    forecast_df = None
    plot_url = None
    ticker = None
    error = None
    if request.method == 'POST':
        ticker = request.form['ticker'].strip().upper()
        if not ticker:
            error = "The field is blank, please input a ticker."
        else:
            df = yf.download(ticker, period="6mo", interval="1d")
            if df.empty:
                error = "This is not a valid ticker."
            else:
                df = df[["Close"]].dropna()
                df.reset_index(inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])
                model = ARIMA(df['Close'], order=(5,1,1))
                model_fit = model.fit()
                # Predict next 7 weekdays
                future_dates = []
                date = df['Date'].max()
                while len(future_dates) < 7:
                    date += timedelta(days=1)
                    if date.weekday() < 5:
                        future_dates.append(date)
                forecast = model_fit.forecast(steps=7)
                forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Close': forecast.values})
                # Plot
                df_weekly = df.set_index('Date').resample('W-FRI').last().reset_index()
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_weekly['Date'], df_weekly['Close'], label="Historical Close (Weekly)")
                ax.plot(forecast_df['Date'], forecast_df['Forecasted Close'], label="Forecasted Close", linestyle='--')
                ax.legend()
                ax.set_title(f"{ticker} Price Forecast for Next Week:")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.grid()
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                plt.close(fig)
                buf.seek(0)
                plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    return render_template_string(TEMPLATE, forecast_df=forecast_df, plot_url=plot_url, ticker=ticker, error=error)

if __name__ == '__main__':
    app.run(debug=True)