'''Enfin, cette page nous permet d'envoyer des requêtes au serveur, afin d'utiliser les modèles en inférence.
Pour cela, nous sélectionnons une expérience, puis un modèle (via la run), et nous loadons le modèle côté serveur, via le module MLFlow Models.
Les inputs sont envoyés au serveur, et le serveur renvoie un prédiction, et l'interprétabilité locale du modèle.

Comme nous le verrons par la suite, nous incluons également des outils de Data Visualisation, et de suivi des performances des modèles.'''

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
    data_import_request = requests.post(url = 'http://ocds7ey.herokuapp.com/production/import_data', data = json.dumps(data_import_input))
    data_import_answer = pd.DataFrame(data_import_request.json())
    return data_import_answer

#Requête permettant d'importer le jeu de données projetté sur un individu et des variables sélectionnés par l'utilisateur.'''
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def import_dataset_filter(exp_name, run_name, dataset, observation, features):
    data_import_input = {'exp_name' : exp_name, 'run_name' : run_name, 'dataset' : dataset, 'observation' : float(observation), 'features' : features}
    data_import_request = requests.post(url = 'http://ocds7ey.herokuapp.com/production/import_data/features_select', data = json.dumps(data_import_input))
    data_import_answer = pd.DataFrame(data_import_request.json())
    return data_import_answer

#'''Requête permettant d'importer la distribution d'une colonne sélectonnée par l'utilisateur. L visualisation est réalisée via la librairie altair pour une 
#meilleure lisibilité.'''
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def import_dataset_distribution(exp_name, run_name, dataset, features):
    data_import_input = {'exp_name' : exp_name, 'run_name' : run_name, 'dataset' : dataset, 'features' : features}
    data_import_request = requests.post(url = 'http://ocds7ey.herokuapp.com/production/import_data/features_select/distribution', data = json.dumps(data_import_input))
    data_import_answer = pd.DataFrame(data_import_request.json())
    return data_import_answer

#'''Requête permettant d'utiliser un modèle chargé via MLFlow Models en inférence sur un individu sélectionné par l'utilisateur.'''
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def prediction(exp_name, run_name, input_data):
    prediction_input = {'exp_name' : exp_name, 'run_name' : run_name, 'data' : input_data}
    prediction_request = requests.post(url = 'http://ocds7ey.herokuapp.com/production/inference', data = json.dumps(prediction_input))
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
    prediction_request = requests.post(url = 'http://ocds7ey.herokuapp.com/production/inference/shap_explainer/no_plot', data = json.dumps(prediction_input))
    prediction_answer = prediction_request.json()['A']
    return prediction_answer

#'''Requête permettant de faire une prédiction sur 1000 individus aléatoires du jeu de données, et de retourner l'interprétation locale moyenne 
#sur les 1000 individus.'''
#@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def sample_50_predict(exp_name, run_name, dataset):
    prediction_input = {'exp_name' : exp_name, 'run_name' : run_name, 'dataset' : dataset}
    prediction_request = requests.post(url = 'http://ocds7ey.herokuapp.com/production/inference/sample1000', data = json.dumps(prediction_input))
    prediction_answer = prediction_request.json()
    prediction = prediction_answer['Prediction']
    probability = prediction_answer['Probability']
    index = prediction_answer['Index']
    shap_values = prediction_answer['shap_values']
    df = pd.DataFrame({'Prediction' : prediction, 'Probability' : probability, 'Index' : index})
    df['Index'] = df['Index'].astype(int)
    return df, shap_values

#'''Requête permettant d'effectuer la même opération que la fonction précédente, mais sur l'ensemble du jeu de données sélectionné.'''
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def sample_full_predict(exp_name, run_name, dataset):
    prediction_input = {'exp_name' : exp_name, 'run_name' : run_name, 'dataset' : dataset}
    prediction_request = requests.post(url = 'http://ocds7ey.herokuapp.com/production/inference/full', data = json.dumps(prediction_input))
    prediction_answer = prediction_request.json()
    prediction = prediction_answer['Prediction']
    probability = prediction_answer['Probability']
    index = prediction_answer['Index']
    shap_values = prediction_answer['shap_values']
    df = pd.DataFrame({'Prediction' : prediction, 'Probability' : probability, 'Index' : index})
    df['Index'] = df['Index'].astype(int)
    return df, shap_values

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
    prediction_request = requests.post(url = 'http://ocds7ey.herokuapp.com/prodction/model_performance', data = json.dumps(prediction_input))
    prediction_answer = pd.DataFrame(prediction_request.json())
    return prediction_answer

#'''Ici, nous sélectionons une requête pour une analyse de datadrift entre la colonne du jeu de données de production, et celle du jeu de données d'entraînement.
#L'output est un fichier html mis sous format string, et directement display dans l'application.'''
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def single_col_data_drift(exp_name, run_name, dataset, feature):
    prediction_input = {'exp_name' : exp_name, 'run_name' : run_name, 'dataset' : dataset, 'features' : feature}
    prediction_request = requests.post(url = 'http://ocds7ey.herokuapp.com/production/datadrift/single_column', data = json.dumps(prediction_input))
    prediction_answer = prediction_request.json()
    return prediction_answer

#'''De la même manière que pour la fonction précédente, nous récupérons le tableau html d'analyse du datadrift, mais cette fois ci sur l'ensemble des features 
#de notre jeu de données.'''
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def full_col_data_drift(exp_name, run_name, dataset):
    prediction_input = {'exp_name' : exp_name, 'run_name' : run_name, 'dataset' : dataset}
    prediction_request = requests.post(url = 'http://ocds7ey.herokuapp.com/production/datadrift/full_column', data = json.dumps(prediction_input))
    prediction_answer = prediction_request.json()['HTML']
    return prediction_answer

st.title('**Mise en production des modèles!**')

st.write("Bienvenue sur notre tableau de bord de Production!")

nb_exp_request = requests.get(url='http://ocds7ey.herokuapp.com/experiment_start')
names = nb_exp_request.json()['Names']
exp = st.sidebar.selectbox('**Veuillez Sélectionner une éxperience :**', names)
run_list_input = {'name' : 'any', 'column' : 'any', 'exp_name' : str(exp)}
run_list_request = requests.post(url = 'http://ocds7ey.herokuapp.com/experiment_start/select/runs', data = json.dumps(run_list_input))
runs = run_list_request.json().keys()
run = st.sidebar.selectbox('**Veuillez Sélectionner un modèle:**', runs)
dataset = st.sidebar.selectbox('**Veuillez sélectionner un jeu de données:**', ['Production', 'Test', 'Entraînement'])
dataset_df = import_dataset(exp, run, dataset).drop(['index'], axis = 1)

st.write("Dans le premier onglet, le jeu de données de production est disponible sous la forme d'un tableau.")
with st.expander('**Affichage du jeu de données complet:**'):
    st.subheader('**Jeu de données de {}**'.format(dataset))
    st.dataframe(dataset_df, use_container_width=True)

st.write("Dans le deuxième onglet, nous pouvons visualiser les informations relatives à une demande de crédit en particulier, en sélectionnant les colonnes à afficher.")
with st.expander('**Focus sur un individu et sélection des features à afficher:**'):
    displayed_features = st.multiselect('**Choix des features à afficher:**', dataset_df.columns)
    observation_display = st.number_input('**Veuillez choisir un individu:**', min_value=0, max_value=int(np.max(dataset_df.index)), step = 10)
    selected_data = import_dataset_filter(exp, run, dataset, observation_display, displayed_features)
    st.dataframe(selected_data)

st.write("Dans ce troisième onglet, nous pouvons visualiser la distribution d'une variable en particulier.")
with st.expander("**Affichage de la distribution d'une colonne:**"):
    feature_distribution = st.selectbox('**Choix des features à afficher:**', dataset_df.columns)
    area_chart_data = import_dataset_distribution(exp, run, dataset, feature_distribution)
    st.subheader("**Distribution de la variable : {}**".format(feature_distribution))
    st.bar_chart(area_chart_data)

st.subheader('**Mise à disposition du modèle : {}**'.format(run))
observation = st.number_input("**Lancer une prédiction:**", min_value=0, max_value=int(np.max(dataset_df.index)), step = 10)
pred_input = list(dataset_df.iloc[int(observation)].astype(float).values)
predicted_class, probability, thresh = prediction(exp, run, pred_input)
if predicted_class == 0:
    tmp = 'accordé'
else:
    tmp = 'refusé'
st.write("Crédit {} : {}% de chances de défaut de paiement!".format(tmp, str(probability*100)[:5]))
st.write("Seuil de décision : {}%".format(str(thresh*100)[:2]))

with st.container():
    st.subheader('**Interprétation locale de la prédiction:**')
    chart_data = pd.DataFrame({"shap_values" : local_explainer_no_plot(exp, run, pred_input), "features" : dataset_df.columns})
    chart_data['abs'] = chart_data['shap_values'].abs()
    chat_data_sorted = chart_data.sort_values(by='abs', ascending = False)
    chat_data_loc = chat_data_sorted.iloc[:10]
    altair_chart = alt.Chart(chat_data_loc).mark_bar().encode(
        x = alt.X('features', axis=alt.Axis(labelAngle=80, labelLimit=500, title = ' '), sort = '-y'), 
        y = "shap_values", 
        color = alt.condition(alt.datum.shap_values > 0, alt.value("blue"), alt.value("red"))).configure_axis(labelFontSize=12).properties(height=600)
    st.altair_chart(altair_chart, use_container_width=True)

st.write(" ")
st.write("Dans l'onglet suivant, nous pouvons effectuer des prédictions groupées sur plusieurs individus.")
with st.expander("**Prédictions multiples:**"):
    mutliple_inference = st.radio("**Veuillez sélectionner une option:**", ('10 Observations', 'Full Dataset'), horizontal = True)
    if mutliple_inference == 'Full Dataset':
        st.warning("**ATTENTION** : Cette option permet de faire des prédictions sur l'ensemble du jeu de données. Cette opération peut prendre beaucoup de temps!")
    else:
        st.info("Cette option permet de réaliser des prédictions sur 10 observations aléatoires de notre jeu de données!")
    if st.button("**Inférence**"):
        with st.spinner("**Inférence en cours! En attente du serveur ...**"):
            if mutliple_inference == '10 Observations':
                predictions_df, shap_values_1000 = sample_50_predict(exp, run, dataset)
                st.dataframe(predictions_df.transpose())
                chart_data_1000 = pd.DataFrame({"shap_values" : shap_values_1000, "features" : dataset_df.columns})
                chart_data_1000['abs'] = chart_data_1000['shap_values'].abs()
                chart_data_sorted_1000 = chart_data_1000.sort_values(by='abs', ascending = False)
                chart_data_1000_loc = chart_data_sorted_1000.iloc[:10]
                altair_chart_1000 = alt.Chart(chart_data_1000_loc).mark_bar().encode(
                    x = alt.X('features', axis=alt.Axis(labelAngle=80, labelLimit=500, title = ' '), sort = '-y'), 
                    y = "shap_values", 
                    color = alt.condition(alt.datum.shap_values > 0, alt.value("blue"), alt.value("red"))).configure_axis(labelFontSize=12).properties(height=600)
                st.altair_chart(altair_chart_1000, use_container_width=True)
            else:
                #Option désactivée dans la version light de l'application
                #full_predictions_df, shap_values_full = sample_full_predict(exp, run, dataset)
                #chart_data_full = pd.DataFrame({"shap_values" : shap_values_full, "features" : dataset_df.columns})
                #chart_data_full['abs'] = chart_data_full['shap_values'].abs()
                #chart_data_sorted_full = chart_data_full.sort_values(by='abs', ascending = False)
                #chart_data_full_loc = chart_data_sorted_full.iloc[:10]
                #altair_chart_full = alt.Chart(chart_data_full_loc).mark_bar().encode(
                    #x = alt.X('features', axis=alt.Axis(labelAngle=80, labelLimit=500, title = ' '), sort = '-y'), 
                    #y = "shap_values", 
                    #color = alt.condition(alt.datum.shap_values > 0, alt.value("blue"), alt.value("red"))).configure_axis(labelFontSize=12).properties(height=600)
                #st.altair_chart(altair_chart_full, use_container_width=True)    
                st.warning("**Cette option est désactivée dans la version light de l'application pour économiser des ressources computationnelles, merci de sélectionner l'option d'inférence sur 10 observations.**")

st.write(" ")
st.write("Chaque prédiction effectuée par le modèle est centralisée via MLFlow Tracking, et les performances du modèle sont suivis par une analyse de Data Drift.")
st.write("En ce qui concerne cette analyse, si nous détections des colonnes pour lesquelles les distribution sont très différentes du jeu de données d'entraînement, une alarte est affichée.")
with st.expander("**Suivi des performances du modèle**"):
    st.warning("**ATTENTION**: Veuillez vous assurer que le modèle a effetcué au minimum 1000 prédictions uniques pour que cette analyse soit pertinente!")
    if st.button("**Analyse de Data Drift sur les prédictons**"):
        performance_df = model_performance_tracker(exp, run)
        performance_chart_data = performance_df[['drift_score', 'threshold']]
        st.subheader("**Data Drift entre jeu de données d'entraînement et prédictions:**")
        if len(performance_df) > 0:
            st.warning("**ATTENTION:** {} variables sont sujettes au data drift entre le jeu de données d'entraînement et les données de production!".format(len(performance_df)))
        st.bar_chart(performance_chart_data)
        st.dataframe(performance_df, use_container_width=True)

st.write(" ")
st.write("Dans l'onglet suivant, il est possible de sélectionner une colonne pour lancer une analyse de Data Drift entre le jeu de données d'entraînement et de production.")
with st.expander("**Data Drift Report:**"):
    drift_column_select = st.selectbox("**Veuillez choisir une colonne:**", dataset_df.columns)
    data_drift_result = single_col_data_drift(exp, run, dataset, drift_column_select)
    json_data_drift = data_drift_result['JSON']
    html_data_drift = data_drift_result['HTML']
    components.html(html_data_drift, scrolling = True, height = 1250)

st.write(" ")
st.write("Dans l'onglet suivant, il est possible lancer une analyse de Data Drift sur l'ensemble des colonnes du jeu de données de production et d'entraîenement.")
with st.expander("**Data Drift Report All Columns**"):
    if st.button('**Analyse de DataDrift sur toutes les colonnes**'):
        #Option désactivée sur la version light de l'application
        #data_drift_report_all_columns = full_col_data_drift(exp, run, dataset)
        #components.html(data_drift_report_all_columns, scrolling = True, height = 1250)
        st.warning("**Cette option est désactivée dans la version light de l'application pour des raisons de ressources computationnelles limitées.**")
