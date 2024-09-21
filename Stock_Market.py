import numpy as np
np.float_ = np.float64
import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet

st.title('Stock Price and Volume Chart')

# List of popular tickers for autocomplete
nse_bse_tickers = {
    'Reliance Industries Limited': 'RELIANCE.NS',
    'Tata Consultancy Services Limited': 'TCS.NS',
    'HDFC Bank Limited': 'HDFCBANK.NS',
    'Infosys Limited': 'INFY.NS',
    'ICICI Bank Limited': 'ICICIBANK.NS',
    'State Bank of India': 'SBIN.NS',
    'Bajaj Finance Limited': 'BAJFINANCE.NS',
    'Bharti Airtel Limited': 'BHARTIARTL.NS',
    'Hindustan Unilever Limited': 'HINDUNILVR.NS',
    'Kotak Mahindra Bank Limited': 'KOTAKBANK.NS',
    'Reliance Industries Limited (BSE)': 'RELIANCE.BO',
    'Tata Consultancy Services Limited (BSE)': 'TCS.BO',
    'HDFC Bank Limited (BSE)': 'HDFCBANK.BO',
    'Infosys Limited (BSE)': 'INFY.BO',
    'ICICI Bank Limited (BSE)': 'ICICIBANK.BO',
    'State Bank of India (BSE)': 'SBIN.BO',
    'Bajaj Finance Limited (BSE)': 'BAJFINANCE.BO',
    'Bharti Airtel Limited (BSE)': 'BHARTIARTL.BO',
    'Hindustan Unilever Limited (BSE)': 'HINDUNILVR.BO',
    'Kotak Mahindra Bank Limited (BSE)': 'KOTAKBANK.BO'
}

ticker = nse_bse_tickers.values()

# Autocomplete ticker input
ticker = st.selectbox('Select Stock Ticker', options=ticker)
selected_company = nse_bse_tickers.get(ticker)
# Date input widgets for start and end dates
start_date = st.date_input('Start Date', value=datetime.today() - timedelta(days=30))
end_date = st.date_input('End Date', value=datetime.today())

# Time frame selection using horizontal radio buttons below the charts
st.write("Select Time Frame:")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button('1 Day'):
        start_date = end_date - timedelta(days=1)
with col2:
    if st.button('3 Days'):
        start_date = end_date - timedelta(days=3)
with col3:
    if st.button('1 Week'):
        start_date = end_date - timedelta(weeks=1)
with col4:
    if st.button('1 Month'):
        start_date = end_date - timedelta(days=30)
with col5:
    if st.button('1 Year'):
        start_date = end_date - timedelta(days=365)

if ticker:
    # Fetch stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    # Count up and down days
    stock_data['Price Change'] = stock_data['Close'].diff()

    up_days = (stock_data['Price Change'] > 0).sum()
    down_days = (stock_data['Price Change'] < 0).sum()
        
    # Calculate percentage moves
    total_up_move = stock_data.loc[stock_data['Price Change'] > 0, 'Price Change'].sum()
    total_down_move = stock_data.loc[stock_data['Price Change'] < 0, 'Price Change'].sum()
    start_price = stock_data['Close'].iloc[0]
    end_price = stock_data['Close'].iloc[-1]
    percent_up_move = (total_up_move / start_price) * 100
    percent_down_move = (total_down_move / start_price) * 100
    total_move = (end_price - start_price)/start_price *100

    if total_move > 0:
            st.markdown(f"**Total Change: {total_move:.2f}%** :green_heart: :arrow_up_small:")
    else:
            st.markdown(f"**Total Change: {total_move:.2f}%** :red_circle: :arrow_down_small:")

    avg_volume = stock_data['Volume'].mean() / 1000  # Convert to thousands
    max_volume = stock_data['Volume'].max() / 1000  # Convert to thousands
    min_volume = stock_data['Volume'].min() / 1000  # Convert to thousands
    max_volume_date = stock_data['Volume'].idxmax().strftime('%Y-%m-%d')
    min_volume_date = stock_data['Volume'].idxmin().strftime('%Y-%m-%d')
        
    data = {
        'Metric': ['Up Days', 'Down Days', 'Percentage Up Move', 'Percentage Down Move', 'Total Change', 'Average Volume (in thousands)', 'Max Volume (in thousands)', 'Min Volume (in thousands)'],
        'Value': [up_days, down_days, f"{percent_up_move:.2f}%", f"{percent_down_move:.2f}%", f"{total_move:.2f}%", f"{avg_volume:.2f}K", f"{max_volume:.2f}K; Date - {max_volume_date}", f"{min_volume:.2f}K;Date - {min_volume_date}"]
    }
    df = pd.DataFrame(data)
    st.table(df)
  # Create a combined chart using Plotly
    #st.subheader(f"Boxplot for 15-minute intervals from {start_date} to {end_date}")
    fig = go.Figure()
        
        # Add box plot trace
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Stock Price'))
        
        # Add volume trace
    fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume', yaxis='y2'))
        
        # Update layout for dual y-axes
    fig.update_layout(
            title=f'({ticker}) Stock Price and Volume',
            xaxis_title='Date',
            yaxis=dict(title='Stock Price'),
            yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False, range=[0, stock_data['Volume'].max() * 4]),
            legend=dict(x=0, y=1.2, orientation='h')
    )
        
        # Display the combined chart
    st.plotly_chart(fig)
      

intervals = [ "5m", "15m", "30m", "60m", "3h", "1d", "5d", "1mo", "3mo", "6mo", "1y"]
selected_intervals = st.selectbox("Select Intervals", intervals)    

if st.button("Fetch Data"):
    data = yf.download(ticker, start=start_date, end=end_date, interval=selected_intervals)
    st.write(f"Data for interval: {selected_intervals}")
    st.write(data)
    
    # Convert data to CSV
    csv = data.to_csv().encode('utf-8')
    st.download_button(label=f"Download CSV for {selected_intervals}", data=csv, file_name=f'{ticker}_{selected_intervals}.csv', mime='text/csv')

forecast_period = st.slider('Select Forecast Period (days)', min_value=1, max_value=365, value=30)
start = datetime.today() - timedelta(days=1825)
end = datetime.today()
stock_data_p =  yf.download(ticker, start=start , end=end)   
stock_data_reset = stock_data_p.reset_index()
#st.write(stock_data_reset.head())  # Debugging step to check the DataFrame structure
#st.write(stock_data_reset.columns)  # Debugging step to check the column names
df_prophet = stock_data_reset[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        
# Initialize and fit the model
model = Prophet()
model.fit(df_prophet)
        
# Create future dataframe
future = model.make_future_dataframe(periods=forecast_period)
        
# Predict
forecast = model.predict(future)
        
# Plot forecast
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Confidence Interval', line=dict(dash='dash')))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Confidence Interval', line=dict(dash='dash')))
        
fig_forecast.update_layout(
            title=f'({ticker}) Stock Price Forecast',
            xaxis_title='Date',
            yaxis_title='Stock Price',
            legend=dict(x=0, y=1.2, orientation='h')
)
        
st.plotly_chart(fig_forecast)
