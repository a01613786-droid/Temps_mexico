import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor


st.write(''' # Predicción de temperatura en varias ciudades de México ''')
st.image("mexico.jpg", caption="Predicción de la temperatura en Acapulco:0, Acuña:1 y Aguascalientes:2")

st.header('Datos de la ciudad y fecha')

def user_input_features():
  # Entrada
  City = st.number_input('City:', min_value=0, max_value=2, value = 0, step = 1)
  year = st.number_input('year', min_value=1, max_value=3000, value = 1, step = 1)
  month = st.number_input('month',min_value=1, max_value=12, value = 1, step = 1)
  day = st.number_input('day:', min_value=1, max_value=31, value = 1, step = 1)

  user_input_data = {'City': City,
                     'year': year,
                     'month': month,
                     'day': day}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

temp =  pd.read_csv('Temp_mexico.csv', encoding='latin-1')
X = temp.drop(columns='AverageTemperature')
y = temp['AverageTemperature']


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613786)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['City'] + b1[1]*df['year'] + b1[2]*df['month'] + b1[3]*df['day']

st.subheader('Predicción de temperatura')
st.write('La temperatura en la ciudad seleccionada será de: ', prediccion)
