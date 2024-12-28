#%% Importi

import Eksplorativna_analiza as ea
import Neuronska_mreza as nm
import folium
from folium.plugins import HeatMap
from tensorflow.keras.models import save_model

#%% Main funkcija - poziva zasebne fajlove od kojih je svaki vezan za jedan deo istraživanja
if __name__ == '__main__':

    data = ea.izvrsi_eksplorativnu_analizu()
#%% 
    # Generisanje heat mape
    output_file_name = 'heatmap_of_LA.html'
    ea.generisi_heat_mapu(data=data, top_n=20, output_file=output_file_name)
#%%  
    # Priprema podataka
    pripremljeni_podaci = nm.priprema_podataka(data)
#%%     
    # Kreiranje i treniranje modela
    model, scaler_X, scaler_y, X_test, y_test, history = nm.kreiraj_i_treniraj_model(pripremljeni_podaci)
#%%    
    # Čuvanje modela
    save_model(model, 'crime_prediction_model.keras')
#%%    
    # Predikcija i evaluacija
    y_pred_rescaled, y_test_rescaled = nm.predikcija(model, scaler_y, X_test, y_test)
    
    # Kreiranje mape za prikaz predviđenih zločina
    m = folium.Map([34.0522, -118.2437], zoom_start=12)  # Koordinate Los Angelesa

    # Dodavanje sloja koji prikazuje celu površinu Los Angelesa
    folium.Choropleth(
        geo_data='los-angeles.geojson',  # GeoJSON koji sadrži granice Los Angelesa
        fill_opacity=0.1,
        line_opacity=0.3,
    ).add_to(m)

    # Pretvaranje predviđenih koordinata u format pogodan za HeatMap
    predikcije = []
    for pred in y_pred_rescaled:
        predikcije.append([pred[0], pred[1]])

    # Generisanje HeatMap za predviđene zločine
    heat_map = HeatMap(predikcije, radius=10, blur=8, max_zoom=1, min_opacity=0.45, name='Predicted Crimes')
    heat_map.add_to(m)

    # Čuvanje mape kao HTML
    m.save('predicted_crime_locations.html')
