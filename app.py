#import pandas_datareader as web
import datetime
import yfinance as yf 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from keras.models import load_model
import streamlit as st 

start = datetime.datetime(2015,1,1)
end = datetime.datetime(2024,1,1)


st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start, end)

# Describing data
st.subheader('Data from 2015 - 2024')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
st.pyplot(fig)


# Plotting for the 10 days moving average for the stock.
st.subheader('Closing Price vs Time Chart with 10MA')
ma10 = df.Close.rolling(10).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
plt.plot(ma10)
st.pyplot(fig)



# Plotting for the 100 days moving average for the stock.
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)

# Plotting for the 200 days moving average for the stock.
st.subheader('Closing Price vs Time Chart with 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
plt.plot(ma200)
st.pyplot(fig)


# plotting for the 10, 100 and 200 days moving average for the stock.
st.subheader('\nClosing Price vs Time Chart with 10MA, 100MA & 200MA')
ma10 = df.Close.rolling(10).mean()
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close, 'b', label = 'Original Price')
plt.plot(ma10, 'm--', label = '10MA') 
plt.plot(ma100, 'r', label = '100MA')
plt.plot(ma200, 'k', label = '200MA')
plt.legend(loc = 'upper left')
st.pyplot(fig)



# Splitting the "close" column Data into Training and Testing.
# For Data Training, DataFrame for close column and staring from the 0 index we calculate 70% of the total values.
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) # Starting from the 0th index of "close" column

# For Data Testing and it will start from the 70% value and it will be going till the complete length.
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# Here the data is splitted in to such a way that 70% data is used for training and 30% data is used for testing.
# Now we scale down the data between 0 and 1 using sklearn.preprocessing.
# Also install scikit-learn using the pip installer.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))


# Now fitting the Training data first.
data_training_array = scaler.fit_transform(data_training) # Scaled down the data using MinMaxScaler.



# Splitting the data into x_train and y_train.
# x_train = []
# y_train = []

# for i in range (100, data_training_array.shape[0]):
#   x_train.append(data_training_array[i - 100: i])
#   y_train.append(data_training_array[i, 0])
# x_train , y_train = np.array(x_train), np.array(y_train)





# Load the pretrined model for the prediction.
model = load_model('keras_model_stock.keras')

# Testing part.
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index = True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range (100, input_data.shape[0]):
  x_test.append(input_data[i - 100: i])
  y_test.append(input_data[i , 0])

# Converting the x_test and y_test into numpy arrays.
x_test, y_test = np.array(x_test), np.array(y_test)


# Now for makking predictions.
y_predicted = model.predict(x_test)

# Now scaling the values up again.
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Now plotting the graph for predicted values.
# The pridected graph will not be 100% accurate but will give approximate predicted trends for the stock.
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc = 'upper left')
st.pyplot(fig2)

