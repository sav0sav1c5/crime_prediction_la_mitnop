# %% Importi

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import folium
import os
from folium.plugins import HeatMap

# %% Pripremanje podataka

def priprema_podataka(data):
    '''Funkcija za pripremu podataka za neuronsku mrežu'''
    # Odabir relevantnih kolona
    relevant_columns = ['YEAR OCC', 'MONTH OCC', 'HOUR OCC', 'AREA', 'Rpt Dist No', 'Crm Cd', 'Vict Age', 'LAT', 'LON']
    data = data[relevant_columns]
    
    # Enkodiranje kategorijalnih podataka
    label_encoder = LabelEncoder()
    data['AREA'] = label_encoder.fit_transform(data['AREA'])
    data['Crm Cd'] = label_encoder.fit_transform(data['Crm Cd'])
    
    return data

# %% Kreiranje modela

def kreiraj_i_treniraj_model(data):
    X = data.drop(columns=['LAT', 'LON'])
    y = data[['LAT', 'LON']]
    
    # Skaliranje podataka
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    # Podela podataka na trening i test setove
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # Kreiranje modela
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='linear'))
    
    # Kompajliranje modela
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    # Treniranje modela
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
    
    return model, scaler_X, scaler_y, X_test, y_test, history

def predikcija(model, scaler_y, X_test, y_test):
    # Predikcija na test setu
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler_y.inverse_transform(y_pred)
    y_test_rescaled = scaler_y.inverse_transform(y_test)
    
    # Prikaz rezultata
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_rescaled[:, 0], y_test_rescaled[:, 1], color='blue', label='Actual', alpha=0.2)
    plt.scatter(y_pred_rescaled[:, 0], y_pred_rescaled[:, 1], color='red', label='Predicted', alpha=0.1)
    plt.xlabel('LAT')
    plt.ylabel('LON')
    plt.title('Actual vs Predicted Locations')
    plt.legend()
    plt.show()
    
    # Ispis rezultata
    print('Mean Absolute Error:', mean_absolute_error(y_test_rescaled, y_pred_rescaled))
    print('Mean Squared Error:', mean_squared_error(y_test_rescaled, y_pred_rescaled))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled)))

    return y_pred_rescaled, y_test_rescaled
