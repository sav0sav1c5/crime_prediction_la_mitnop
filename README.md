# Projekat iz Metode i tehnike nauke o podacima (MITNOP) - Predikcija Zločina

## Opis

Ovaj repozitorijum sadrži kompletan materijal vezan za analizu i predikciju zločina u Los Anđelesu. Projekat se fokusira na identifikaciju obrazaca korišćenjem neuronskih mreža za predikciju lokacija i vremena budućih zločina. Uključuje eksplorativnu analizu podataka, implementaciju različitih neuronskih mreža i vizualizaciju rezultata.

## Specifikacije projekta

1. **Specifikacija Projekta**
   - Analiza podataka o zločinima na osnovu javno dostupnih setova podataka.
   - Implementacija modela za predikciju uz pomoć mašinskog učenja.

2. **Eksplorativna Analiza Podataka**
   - Vizualizacija geografskih i vremenskih raspodela zločina.
   - Priprema podataka za modelovanje.

3. **Implementacija Neuronskih Mreža**
   - Jednostavna neuronska mreža (SLNN).
   - Višeslojna neuronska mreža (MLNN).
   - Rekurentne neuronske mreže sa GRU jedinicama.
   - Evaluacija modela na osnovu tačnosti i pouzdanosti.

4. **Interaktivne Vizualizacije**
   - Generisanje toplotnih mapa i interaktivnih geografskih prikaza.

## Struktura Projekta

- `data/` - Sadrži ulazne skupove podataka i rezultate pretprocesiranja.
- `models/` - Implementacije neuronskih mreža i trenirani modeli:
  - `single_layer_NN.py` - Jednostavna neuronska mreža.
  - `MaskedRNN.py` - Napredne RNN implementacije.
  - `MultiInputRNN.py` - Model sa više ulaza za RNN.
  - `crime_prediction_model.h5` - Sačuvani trenirani model.
- `notebooks/` - Jupyter beležnice za eksplorativnu analizu i vizualizaciju.
- `visualizations/` - Izlazni fajlovi vizualizacija:
  - `crime_geographical_distribution.png` - Raspodela zločina.
  - `heatmap_of_LA.html` - Interaktivna toplotna mapa.
- `scripts/` - Glavne skripte za pokretanje projekta:
  - `main.py` - Glavna skripta za analizu i treniranje modela.
  - `RNN_data_preprocessing.py` - Pretprocesiranje podataka.
- `requirements.txt` - Lista potrebnih biblioteka za projekat.

## Pokretanje Projekta

1. Klonirajte repozitorijum:
   ```bash
   git clone https://github.com/korisnik/projekat.git
   ```
2. Instalirajte potrebne biblioteke:
   pip install -r requirements.txt
3. Pokrenite aplikaciju:
   python main.py

## Rezultati

Vizualizacija podataka: Interaktivne mape i grafikoni koji prikazuju vremensku i geografsku distribuciju zločina.
Evaluacija modela: Komparativna analiza performansi različitih neuronskih mreža.
Predikcija: Generisanje predikcija na osnovu novih podataka o zločinima.

## Autori
**Duško Radić IN39/2021**
**Savo Savić IN50/2021**

Projekat izrađen kao deo zadatka na MITNOP kursu.

