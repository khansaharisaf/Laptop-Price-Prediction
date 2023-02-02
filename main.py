import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

#import the model
pipe = pickle.load(open('pipe.pkl','rb'))
# df = pickle.load(open('df.pkl','rb'))
df = pd.read_csv("data.csv")

ohe = OneHotEncoder(handle_unknown="ignore")
X = df.drop(columns=['Price_euros'])
y = df['Price_euros']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

X_OHE = ohe.fit_transform(X_train)

X_test_ohe = ohe.transform(X_test)

rf = RandomForestRegressor()

rf.fit(X_OHE, y_train)

st.title("Laptop Price Predictor")

#Brand Laptop
company = st.selectbox('Brand',df['Company'].unique())

# Type Laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram 
ram = st.selectbox('RAM (in GB)',[2,4,6,8,12,16,24,32,64])

# Weight
weight = st.number_input('Weight')

#Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# Screen Size
screen_size = st.number_input('Screen_Size')

# resolution
resolution = st.selectbox('Screen Resolution',['1366x768','1600x900','1920x1080','2304x1440','2560x1440','2560x1600'])

#CPU
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

#hdd
hdd = st.selectbox('HDD (in GB)',[0,8,128,256,512,1024])

#ssd
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

#gpu
gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

# Os
os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    #query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
    
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    
#split nilai ppi dan diubah ke integer

X_res = int(resolution.split('x')[0])
Y_res = int(resolution.split('x')[0])
#ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
# query = np.array([company,type,ram,touchscreen,ips,weight,cpu,hdd,ssd,gpu,os])
# st.table(df) 

temp = pd.DataFrame([company,type,ram,touchscreen,ips,weight,cpu,hdd,ssd,gpu,os]).T
A = ohe.transform(temp)
st.table(A)
st.write(A.shape)
#Prediksi Harga Laptop
# query = query.reshape(1,11)
st.title("Prediction : $ " + str(rf.predict(A)[0]))
# st.title("Predicted price in Dollar : $ " + str(int(np.exp(rf.predict(X)[0]))))
