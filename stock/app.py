import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock trend Prediction')
user_input = st.text_input('Enter stock Ticker','AAPL')

df = data.DataReader(user_input,'yahoo',start,end)
##describing data
st.subheader('Data from 2010-2019')
st.write(df.describe())
##visualisation
st.subheader("Closing Price vs Time chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time chart with 100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time chart with 100MA & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

###spliting data into training and testing
## trainign 70 % of total data
##testing 30% of total data...working only on closing column
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])


###for lstm model we will have to scale down data between 0 and 1
###processing data  between 0 and 1 using sklearn preprocessing MinMaxScaller
from sklearn.preprocessing import MinMaxScaler

###defining object of MInMaxscaler and feature range between 0 and 1 
scaler = MinMaxScaler(feature_range=(0,1))
##fit training data into min_max_sacaler and return as an array
data_training_array = scaler.fit_transform(data_training)

###x train for getting y_train   y train is dependent on first 100 for our case.
###..last one eleemtn of x_train remove and first eleemnt of y_train add each time
# x_train = []
# y_train = []

# for i in range(100,data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100:i])
#     y_train.append(data_training_array[i,0])
# ### converting x_train list and y_train list into array
# # x_train
# x_train = np.array(x_train)
# y_train = np.array(y_train)

##Load my model
model = load_model('keras_model.h5')
##Testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test = np.array(x_test)
y_test = np.array(y_test)

##making prediction
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler

y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

##final graph
st.subheader('Prediction vs Original')

fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)