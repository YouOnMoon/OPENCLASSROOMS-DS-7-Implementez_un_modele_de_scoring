'''Ce fichier permet de mettre en place notre dashboard à destination des décisionnaires. Les différentes requêtes permettent d'accéder aux informations du jeu de données de production, et à effectuer des prédictions.'''

#Importation des librairies utiles

#Streamlit
import streamlit as st
import streamlit.components.v1 as components #Affichage de données HTML

#Manipulation des données
import pandas as pd
import numpy as np
import json

#Requêtes
import requests

#Visualisation des données
import altair as alt

#Requête permettant d'importer le jeu de donées sélectionné en production à partir de l'expérience et la run choisie par l'utilisateur.
#L'utilisateur peut importer le ju de données X_trai d'entraînement, X_test de test, de de Production.'''
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def import_dataset(exp_name, run_name, dataset):
    data_import_input = {'exp_name' : exp_name, 'run_name' : run_name, 'dataset' : dataset}
    data_import_request = requests.post(url = 'https://ocds7ey.herokuapp.com/production/import_data', data = json.dumps(data_import_input))
    data_import_answer = pd.DataFrame(data_import_request.json())
    return data_import_answer

#Requête permettant d'importer le jeu de données projetté sur un individu et des variables sélectionnés par l'utilisateur.'''
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def import_dataset_filter(exp_name, run_name, dataset, observation, features):
    data_import_input = {'exp_name' : exp_name, 'run_name' : run_name, 'dataset' : dataset, 'observation' : float(observation), 'features' : features}
    data_import_request = requests.post(url = 'https://ocds7ey.herokuapp.com/production/import_data/features_select', data = json.dumps(data_import_input))
    data_import_answer = pd.DataFrame(data_import_request.json())
    return data_import_answer

#'''Requête permettant d'importer la distribution d'une colonne sélectonnée par l'utilisateur. L visualisation est réalisée via la librairie altair pour une 
#meilleure lisibilité.'''
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def import_dataset_distribution(exp_name, run_name, dataset, features):
    data_import_input = {'exp_name' : exp_name, 'run_name' : run_name, 'dataset' : dataset, 'features' : features}
    data_import_request = requests.post(url = 'https://ocds7ey.herokuapp.com/production/import_data/features_select/distribution', data = json.dumps(data_import_input))
    data_import_answer = pd.DataFrame(data_import_request.json())
    return data_import_answer

#'''Requête permettant d'utiliser un modèle chargé via MLFlow Models en inférence sur un individu sélectionné par l'utilisateur.'''
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def prediction(exp_name, run_name, input_data):
    prediction_input = {'exp_name' : exp_name, 'run_name' : run_name, 'data' : input_data}
    prediction_request = requests.post(url = 'https://ocds7ey.herokuapp.com/production/inference', data = json.dumps(prediction_input))
    prediction_answer = prediction_request.json()
    predicted_class = prediction_answer['Prediction']
    probability = prediction_answer['Probability']
    threshold = prediction_answer['Threshold']
    return predicted_class, probability, threshold

#'''Requête permettant d'afficher l'interprétabilité locale du modèle sur l'individu sélectionné grâce à la libraire SHAP_VALUES côté serveur.
#Le serveur ne renvoi que les features pour lesquels on a une importance significative pour une meilleure lisibilité.'''
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def local_explainer_no_plot(exp_name, run_name, input_data):
    prediction_input = {'exp_name' : exp_name, 'run_name' : run_name, 'data' : input_data}
    prediction_request = requests.post(url = 'https://ocds7ey.herokuapp.com/production/inference/shap_explainer/no_plot', data = json.dumps(prediction_input))
    prediction_answer = prediction_request.json()['A']
    return prediction_answer

#'''A l'utilisation d'un modèle en inférence, MLFlow log les prédictions et les individus dans le repértoire artifacts/predictions de la run considérée.
#Cette requête demande au serveur de concaténer les inputs quele modèle a vue, de dropper les valeurs dupliquées si le modèle a réalisé plusieurs prédictions
#sur le même individu, et effectuer une analyse de data drift entre le jeu de données d'entraînement et les individus utilisé en inférence.
#Cette analyse de DataDrift est réalisée grâce à la librairie evidently, et le DataDriftPreset.

#Attention, cette analyse n'est fiable que si le modèle a été utilisé sur un nombre conséquent d'individus (sinon nous auront de nombreuses colonnes faussement
#datadriftées).

#L'analyse du datadrift est transmise au format .json, et seulement pour les colonnes pour lesquelles le datadrift est détécté.'''
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def model_performance_tracker(exp_name, run_name):
    prediction_input = {'exp_name' : exp_name, 'run_name' : run_name}
    prediction_request = requests.post(url = 'https://ocds7ey.herokuapp.com/prodction/model_performance', data = json.dumps(prediction_input))
    prediction_answer = pd.DataFrame(prediction_request.json())
    return prediction_answer

st.title('**Dashboard de Production**')

st.write("Bienvenue sur notre tableau de bord de Production!")

#Chargement de l'expérience de référence, de la run correspondant à la regression logistique, et du jeu de données de production
exp_name = 'PROJECT_7_OC_DS_EXPERIENCE_REFERENCE'
run_name = 'Log_Reg_Class_Weight'
dataset = 'Production'

#Importation du jeu de données de production
dataset_df = import_dataset(exp_name, run_name, dataset).drop(['index'], axis = 1)

#Mise en forme de la page
st.write("Sur ce tableau de bord, nous reprennons l'ensemble des demandes de crédit en cours, et nous effectuons des prédictions sur la capacité d'un demandeur de crédit à le rembourser.")
st.write("Le premier onglet liste l'ensemble des demandes de crédit, et fournit les informations clients associées à chaque demande.")

#Création de trois colonnes d'affichage des données
with st.expander("**Liste de toutes les demandes de crédit en cours:**"):
    st.subheader("**Jeu de données de Production complet!**")
    st.write("Dans ce tableau, chaque ligne identifiée par son numéro correspond à une demande de crédit, et chaque colonne correspond à une information client particulière.")
    st.dataframe(dataset_df, use_container_width=True)

col1, col2= st.columns([3,2], gap = 'large')

#Colonne d'affichage de la distribution d'une variable du jeu de données de production
with col1:
    st.subheader("**Affichage de la répartition d'une information client particulière.**")
    st.write("*Ici, nous pouvons sélectionner une information client, et en visualiser la distribution.*")
    feature_distribution = st.selectbox("**Choix de l'information client à afficher:**", dataset_df.columns, index = 185)
    area_chart_data = import_dataset_distribution(exp_name, run_name, dataset, feature_distribution)
    area_chart_data['index'] = area_chart_data.index.astype(float)
    altair_chart_distribution = alt.Chart(area_chart_data).mark_bar().encode(
        x = alt.X('index', axis=alt.Axis(labelAngle=0, labelLimit=500, title = feature_distribution, labelColor = 'black', titleFontSize = 20, titleFontWeight = 'bold', titleColor = 'black')), 
        y = alt.Y(feature_distribution, axis=alt.Axis(labelAngle=0, labelLimit=500, title = 'Décompte', labelColor = 'black', titleFontSize = 20, titleFontWeight = 'bold', titleColor = 'black')))
    st.altair_chart(altair_chart_distribution, use_container_width=True)

#Colonne d'affichage des informations d'une demande de crédit en particulier par sélection des features et de l'individu sélectionné
with col2:
    st.subheader("**Affichage des informations d'une demande de crédit choisie!**")
    st.write("*Ici, nous pouvons afficher différentes informations d'un client en particulier, en entrer le numéro de la demande de crédit considérée.*")
    displayed_features = st.multiselect('**Choix des informations à afficher:**', dataset_df.columns, default = ['PAYMENT_RATE', 'AMT_CREDIT', 'AMT_INCOME_TOTAL'])
    observation_display = st.number_input('**Veuillez choisir une demande de crédit en entrant son numéro:**', min_value=0, max_value=int(np.max(dataset_df.index)), step = 10)
    selected_data = import_dataset_filter(exp_name, run_name, dataset, observation_display, displayed_features)
    st.dataframe(selected_data)

st.write("*Enfin, nous pouvons effectuer une prédiction de probabilité de défault de paiement d'un demandeur de crédit en entrant le numéro de la demande de crédit ci-dessous.*")
st.write("*Un gestionnaire d'alertes est disponible dans la partie de droite, de manière à valider la qualité des prédictions.*")
col3, col4 = st.columns([3, 2])

#Colonne permettant d'effectuer une prédiction sur un individu
with col3:
    st.subheader('**Prédictions des demandes de crédit en cours:**')
    observation = st.number_input("**Veuillez sélectionner le numéro d'une demande de crédit pour effectuer une prédiction:**", min_value=0, max_value=int(np.max(dataset_df.index)), step = 10)
    pred_input = list(dataset_df.iloc[int(observation)].astype(float).values)
    predicted_class, probability, thresh = prediction(exp_name, run_name, pred_input)

    st.metric(label = "**Probabilité de défaut de paiement:**", value = str(probability*100)[:5] + "%")
    st.metric(label = "**Seuil de décision à ne pas dépasser pour accord de crédit:**", value = str(thresh*100)[:2] + "%")

    if predicted_class == 0:
        st.metric(label = "**Statut de la demande de crédit:**", value = "Accordé", delta = float(str(thresh*100 - probability*100)[:2]), delta_color='normal')
    else:
        st.metric(label = "**Statut de la demande de crédit:**", value = "Refusé", delta = float(str(thresh*100 - probability*100)[:2]), delta_color='normal')

#Colonne permettant de vérifier la viabilité du modèle en production en comptant le nombre de colonnes sujettes au data drift
with col4:
    st.subheader('**Gestion des alertes:**')
    st.info("**Nous proposons 3 niveaux d'alerte pour rendre compte de la viabilité des prédictions.**")
    st.success("**Moins de 5 alertes: performances optimales!**")
    st.warning("**Moins de 15 alertes: performances acceptables!**")
    st.error("**Plus de 15 alertes: action urgente requise.**")
    if st.button("**Lancer une analyse de performance des prédictions!**"):
        performance_df = model_performance_tracker(exp_name, run_name)
        performance_chart_data = performance_df[['drift_score', 'threshold']]
        if len(performance_df) < 5:
            st.success("**ATTETION:** {] alertes détéctées, les prédictions sont considérées comme viables, aucune action n'est requise!".format(len(performance_df)))
        elif len(performance_df) < 15:
            st.warning("**ATTENTION:** {} alertes détéctées, les prédictions sont toujours viables mais veuillez tenir le Data Scientist informé de la situation.".format(len(performance_df)))
        else:
            st.error("**ATTENTION:** {} alertes détéctées, veuillez contacter le Data Scientist d'urgence!".format(len(performance_df)))

#Partie permettant l'interprétation locale du modèle pour chaque prédiction, en utilisant la librairie shap côté serveur, nous mettons en avant les features les plus importants pour la prédiction
st.write("*Pour chaque prédiction, nous mettons en avant les caractéristiques clients favorables et défavorables ci-dessous, de manière à pouvoir justifier de manière transparente chaque décisions aux clients.*")
st.subheader('**Informations clients décisives pour la prédiction:**')
chart_data = pd.DataFrame({"shap_values" : local_explainer_no_plot(exp_name, run_name, pred_input), "features" : dataset_df.columns})
chart_data['abs'] = chart_data['shap_values'].abs()
chat_data_sorted = chart_data.sort_values(by='abs', ascending = False)
chat_data_loc = chat_data_sorted.iloc[:10]
chat_data_loc['color'] = chat_data_loc['shap_values'] > 0
chat_data_loc['Légende'] = np.nan
chat_data_loc.loc[chat_data_loc['color'] == False, 'Légende'] = "Facteur défavorable à l'accord du crédit"
chat_data_loc.loc[chat_data_loc['color'] == True, 'Légende'] = "Facteur favorable à l'accord du crédit"
altair_chart = alt.Chart(chat_data_loc).mark_bar().encode(
    y = alt.Y('features', axis=alt.Axis(labelAngle=0, labelLimit=500, title = 'Informations client décisives', labelColor = 'black', titlePadding=150, titleFontSize = 20, titleFontWeight = 'bold', titleColor = 'black'), sort = '-x'), 
    x = alt.X("shap_values", axis=alt.Axis(labelAngle=0, labelLimit=500, title = 'Importance des informations client', labelColor = 'black', titleFontSize = 20, titleFontWeight = 'bold', titleColor = 'black')), 
    color = alt.Color('Légende', scale = alt.Scale(range = ['red', 'blue']), legend=alt.Legend(orient="top", labelLimit=500, 
                                                                                               columns = 1, fillColor = 'cornsilk', labelColor = 'black', 
                                                                                               titleColor = 'black', titleFontSize = 20, titleFontWeight = 'bold'))).configure_axis(labelFontSize=12).properties(height=800)
st.altair_chart(altair_chart, use_container_width=True)