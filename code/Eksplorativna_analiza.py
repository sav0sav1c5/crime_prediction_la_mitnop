#%% Importi

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium as folium
from folium.plugins import HeatMap

def izvrsi_eksplorativnu_analizu(plot=False):
    '''Vraca dataframe sa kolonama koje ce se koristiti pri obuci neuronskih mreza'''  
    
#%% Čitanje podataka iz .CSV fajla
    
    data = pd.read_csv('Crime_Data_from_2020_to_Present.csv')
    
    # Eksplorativna analiza
    print("Prvih nekoliko redova skupa podataka:")
    print(data.head())
    
    print("Osnovna statistika numeričkih kolona:")
    print(data.describe())
    
    print("Osnovne informacije o skupu podataka:")
    print(data.info())
    
#%% Provera postojanja nedostajućih vrednosti u skupu
    
    missing_values = data.isna().sum()
    
    print(missing_values)
    
#%% Uklanjanje uočenih nedostajućih vrednosti
    
    columns_for_drop = ['Mocodes', 'Vict Sex', 'Vict Descent',
                        'Premis Cd', 'Premis Desc', 'Weapon Used Cd',
                        'Weapon Desc', 'Crm Cd 1', 'Crm Cd 2',
                        'Crm Cd 3', 'Crm Cd 4', 'Cross Street']
    
    data = data.drop(columns = columns_for_drop)
    
    missing_values = data.isna().sum()
    
    print(missing_values)
    
    print(data)
    
#%% Brisanje redova gdje su koordinate ne postojece 
    
    data = data.drop(data.loc[(data['LAT'] == data['LON']) & (data['LAT'] == 0)].index)
    
    print(data)
    
#%% Parsiranje datuma i vremena u skupu podataka
    
    try:
        data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p')
    except ValueError:
        data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], errors='coerce')
    
    data['YEAR OCC'] = data['DATE OCC'].dt.year
    data['MONTH OCC'] = data['DATE OCC'].dt.month
    data['HOUR OCC'] = data['TIME OCC'].apply(lambda x: int(str(x).zfill(4)[:2]))
    data['QUARTER OCC'] = data['DATE OCC'].dt.to_period('Q')
    
#%% Identifikacija ključnih parametara za kreiranje istarživanja
    
    # Provera broja različitih vrednosti za kolonu 'AREA'
    area_unique_count = data['AREA'].nunique()
    print(f"Broj različitih vrednosti u koloni 'AREA': {area_unique_count}")
    
    # Provera broja različitih vrednosti za kolonu 'LOCATION'
    location_unique_count = data['LOCATION'].nunique()
    print(f"Broj različitih vrednosti u koloni 'LOCATION': {location_unique_count}")
    
    # Provera broja različitih vrednosti za kolonu 'Rpt Dist No'
    rpt_dist_unique_count = data['Rpt Dist No'].nunique()
    print(f"Broj različitih vrednosti u koloni 'Rpt Dist No': {rpt_dist_unique_count}")
    
    # Provera broja različitih vrednosti za kolonu 'Crm Cd'
    crmcd_dist_unique_count = data['Crm Cd'].nunique()
    print(f"Broj različitih vrednosti u koloni 'Crm Cd': {crmcd_dist_unique_count}")
    
#%% Distribucija zločina prema tipu zločina #1
    
    # Provera broja različitih vrednosti za kolonu 'Crm Cd Desc'
    crime_type_counts = data['Crm Cd Desc'].value_counts()
    print(crime_type_counts.head(15))
    
    if(plot):
        plt.figure(figsize=(15, 8))
        crime_type_counts.plot(kind='bar')
        plt.title('Distribucija zločina prema tipu zločina')
        plt.xlabel('Tip krivičnog dela')
        plt.ylabel('Broj krivičnih dela')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
#%% Distribucija zločina prema tipu zločina #2
    
    # Nakon prikazivanja podataka uočeno je da je spektar tipova 
    # zločina u odnosu na frekvenciju njihovog desavanja u gradu 
    # tolike teritorije preveliki pa je posmatranje distribucije
    # zločina prema tipu bazirano na one bitne i najučestalije
    
    # Izaberite nekoliko najčešćih vrsta krivičnih dela za plotovanje
    top_crime_types = crime_type_counts.head(20)
    data_top_crimes = data[data['Crm Cd Desc'].isin(top_crime_types.index)]
    
    if(plot):
        plt.figure(figsize=(15, 10))
        top_crime_types.plot(kind='bar')
        plt.title('Distribucija zločina prema tipu zločina')
        plt.xlabel('Tip krivičnog dela')
        plt.ylabel('Broj krivičnih dela')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
#%% Distribucija zločina prema oblasti u kojoj su se desili

    if(plot):
        plt.figure(figsize=(12, 8))
        data['AREA NAME'].value_counts().plot(kind='bar')
        plt.title('Distribucija zločina prema oblasti u kojoj su se desili')
        plt.xlabel('Oblast dešavanja')
        plt.ylabel('Broj krivičnih dela')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    
#%% Distribucija zločina prema godini dešavanja
    
    if(plot):
        plt.figure(figsize=(10, 6))
        data_top_crimes = data[data['Crm Cd Desc'].isin(top_crime_types.index)]
        data_top_crimes['YEAR OCC'].value_counts().sort_index().plot(kind='bar')
        plt.title('Distribucija zločina prema godini dešavanja (Top 20)')
        plt.xlabel('Godina dešavanja')
        plt.ylabel('Broj krivičnih dela')
        plt.show()
    
#%% Distribucija zločina prema mesecu dešavanja
    
    if(plot):
        plt.figure(figsize=(10, 6))
        data_top_crimes['MONTH OCC'].value_counts().sort_index().plot(kind='bar')
        plt.title('Distribucija zločina prema mesecu dešavanja (Top 20)')
        plt.xlabel('Mesec dešavanja')
        plt.ylabel('Broj krivičnih dela')
        plt.show()
    
#%% Distribucija zločina prema vremena dešavanja u toku dana
    
    if(plot):
        plt.figure(figsize=(10, 6))
        hour_counts_top = data_top_crimes['HOUR OCC'].value_counts().sort_index()
        hour_labels_top = [f"{hour:02}:00" for hour in hour_counts_top.index]
        hour_counts_top.plot(kind='bar')
        plt.title('Distribucija zločina prema vremenu dešavanja u toku dana (Top 20)')
        plt.xlabel('Sat dešavanja')
        plt.ylabel('Broj krivičnih dela')
        plt.xticks(range(len(hour_labels_top)), hour_labels_top, rotation=45)
        plt.tight_layout()
        plt.show()

    return data_top_crimes

# %% Dodatna obrada skupa podataka i generisanje heat mape

def generisi_heat_mapu(data, top_n, output_file):
    top_crime_types = data['Crm Cd'].value_counts().head(top_n).index
    data = data[data['Crm Cd'].isin(top_crime_types)]
    
    mapa = folium.Map(location=[34.0522, -118.2437], zoom_start=10)
    
    heat_data = [[row['LAT'], row['LON']] for index, row in data.iterrows()]
    
    gradient = {
        0.2: 'blue',
        0.4: 'lime',
        0.6: 'yellow',
        0.8: 'orange',
        1.0: 'red'
    }

    HeatMap(heat_data, radius=10, blur=5, max_zoom=1, min_opacity=0.3, gradient=gradient).add_to(mapa)
    
    
    absolute_path = os.path.join(os.getcwd(), output_file)
    
    mapa.save(absolute_path)
    print(f"Heat map saved to {absolute_path}")