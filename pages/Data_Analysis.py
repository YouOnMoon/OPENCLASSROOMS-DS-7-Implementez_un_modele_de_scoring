'''Cette premère page nous sert à explorer relativement simplememnt nos données initiales, en les visualisant dans
l'application.
Ici, nous nous concentrons sur les données telles qu'elles sont disponibles au départ.'''

#Importation des librairies utiles

#Streamlit
import streamlit as st

#Manipulation des données
import pandas as pd
import numpy as np
import json

#Requests pour envoyer les requêtes au serveur via l'API réalisée avec FastAPI
import requests

#Présentation de la page
st.title('**Données à notre disposition pour notre étude!**')

'''De manière à fournir une meilleure expérience utilisateur, nous utilisons le système de chache de streamlit
de manière à ne pas recharger toute la page lors des manipulations de l'utilisateur.
Certaines fonctions peuvent prendre du temps, et le fait de conserver le résultat d'une fonction en l'absence de chagement
nous est fortement utile.'''

#Mise en cache de la requête POST nous permettant de charger un jeu de données sélectionné par l'utilisateur
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def import_dataset(name):
    data_import_input = {'name' : name, 'column' : 'any'}
    data_import_request = requests.post(url = "http://ocds7ey.herokuapp.com/initial_data_select/to_df", data = json.dumps(data_import_input))
    data_import_answer = pd.DataFrame(data_import_request.json())
    #de manière à pouvoir recevoir les données, nous devont remplacer les valeurs np.nan (sinon, impossible de les recevoir)
    #Pour cela, nous les remplacons par une valeur abérante côté serveur, et nous les remplacons de nouveau ici
    data_import_answer = data_import_answer.replace(-15000, np.nan)
    return data_import_answer

#Texte explicatif du premier expander
st.write("Cette partie est dédiée à la visualisation des données initiales à notre disposition.")
st.write("Pour ce faire, nous choisissons un des fichiers de départ, et nous pouvons ensuite sélectionner une colonne pour en afficher la distribution.")
st.write(' ')
st.write("Dans le premier onglet, nous avons à notre disposition un sélecteur nous permettant de sélectionner un des fichiers, et d'afficher un échantillon de sa population.")

#Premier expander - permet de sélectionner un jeu de données, et d'afficher le dataframe
with st.expander("**Sélection des données**"):
    #première requête GET permettant de recevoir la liste des fichiers csv disponibles dans le reprértoire ../inputs
    res = requests.get(url = "http://ocds7ey.herokuapp.com/initial_data_select")
    expander_options = res.json().keys()
    option = st.selectbox('Veuillez sélectionner un fichier', expander_options)

    #Requête pour recevoir le jeu de données sélectionné
    df = import_dataset(str(option))
    st.subheader("Affichage d'un échantillon de 500 individus du fichier : **{}**".format(option))
    st.dataframe(df)
    cols = df.columns

#Texte explicatif du second expander pour la visualisation des données
st.write("Dans le second onglet, nous avons un sélecteur à notre disposition nous permettant de choisir une colonne du jeu de données considéré, et en afficher la distribution.")
st.write("Un boutton est également disponible pour afficher la distribution des coefficients de Pearson des variables quantitatives.")

#Le second expander nous permet de sélectionner une colonne dans le jeu de données sélectionné, et d'en afficher la distribution.
#Pour cela, nous envoyons une requête POST au serveur, qui nous renverra une image, dont les signaux sont séparés, 
#et reconstitués ici de manière à être JSON serializable. L'image contiendra un plot, de type piechart pour les 
#variables qualitatives, et histplot pour les variables quantitatives.
with st.expander("**Visualisation des données**"):
    #Sélection de la colonnes
    data_file = st.selectbox('Veuillez sélectionner une variable:', cols)
    column_select = {'name' : str(option), 'column' : str(data_file)}

    #Requête d'affichage du plot
    h1 = requests.post('http://ocds7ey.herokuapp.com/initial_data_select/to_figure', data = json.dumps(column_select))
    full_channels = h1.json()

    #Reconstitution de l'image
    red = pd.DataFrame(full_channels['R']).values
    green = pd.DataFrame(full_channels['G']).values
    blue = pd.DataFrame(full_channels['B']).values
    last = pd.DataFrame(full_channels['A']).values
    rgb = np.dstack((red,green,blue, last))
    st.subheader("Affichage de la distribution de la variable sélectionnée.")
    #Affichage
    st.image(rgb, use_column_width = True)

    #Enfin, nous pouvons afficher, en cliquant sur un bouton, la distribution des coefficients de pearson de nos 
    #colonnes, pour le dataframe sélectionné au premier expander de la même manière
    if st.button('Distribution des coeffcients de corrélation'):
        df_to_corr = {"name" : str(option), "column" : "any"}
        h2 = requests.post('http://ocds7ey.herokuapp.com/initial_data_select/corelations_plotter', data = json.dumps(df_to_corr))
        full_channels2 = h2.json()
        red2 = pd.DataFrame(full_channels2['R']).values
        green2 = pd.DataFrame(full_channels2['G']).values
        blue2 = pd.DataFrame(full_channels2['B']).values
        last2 = pd.DataFrame(full_channels2['A']).values
        rgb2 = np.dstack((red2,green2,blue2, last2))
        st.image(rgb2)