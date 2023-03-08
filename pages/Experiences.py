'''Cette page nous permet de comparer les expériences, les runs, et par extension, les modèles entraînés précédemment.
Elle nous permet également, via MLFlow, de supprimer les runs et expériences, si celles-ci sont considérées obsolètres.
'''

#Importation des librairies utiles

#Streamlit
import streamlit as st

#Manipulation des données
import pandas as pd
import numpy as np
import json

#Requests
import requests

st.title('**Gestion des expériences!**')

st.write("Sur cette page, nous comparons les modèles au sein des différentes expériences via les métriques enregistrée grâce à MLFlow à l'entraînement.")

st.write("Le premier onglet est dédié au comparatif des métriques et la gestion des expériences et des runs.")
st.write("Il vous est possible d'afficher les métriques des runs, et de les comparer via un système de visualisation des données.")
st.write("Si vous le souhaitez, vous pouvez également supprimer des runs ou des expériences, si celles-ci ne correpsondent pas aux critères de performance attendus en production.")
#Mise en cache des résultats de certaines requêtes pour amélioration 
#de l'expérience utilisateur.'''

#requête demandant au serveur de supprimer une expérience - utilisation de la fonction del_experiement de ../src/utils.py
@st.cache(suppress_st_warning=True)
def del_experiences(exp_name):
    deletion_request_input = {'name' : 'any', 'column' : 'any', 'exp_name' : exp_name}
    deletion_request = requests.post(url='http://ocds7ey.herokuapp.com/experiment_start/delete', data = json.dumps(deletion_request_input))
    deletion_answer = deletion_request.json()
    deleted_exp_name = deletion_answer['exp']
    return deleted_exp_name

#affichage du tableau des métriques des runs d'une expérience sous forme de dataframe pour comparaison des modèles
@st.cache(suppress_st_warning=True)
def metrics_displayer(exp_name, SCORE, ACC, REC, F1, AUC, BAL_ACC):
    metrics_request_innput = {'exp_name' : exp_name, 'SCORE' : SCORE, 'ACC' : ACC, 'REC': REC, 'F1' : F1, 'AUC' : AUC, 'BAL_ACC' : BAL_ACC}
    metrics_request = requests.post(url='http://ocds7ey.herokuapp.com/experiments/runs/metrics', data = json.dumps(metrics_request_innput))
    metrics_answer = metrics_request.json()
    metrics_answer_df = pd.DataFrame(metrics_answer)
    metrics_answer_df = metrics_answer_df.replace(-1, np.nan)
    return metrics_answer_df

#requête demandant au serveur de supprimer une run - utilisation de la fonction del_run de ../src/utils.py
@st.cache(suppress_st_warning=True)
def del_run(exp_name, run_name):
    deletion_request_input = {'exp_name' : exp_name, 'run_name' : run_name}
    deletion_request = requests.post(url='http://ocds7ey.herokuapp.com/experiments/runs/delete', data = json.dumps(deletion_request_input))
    deletion_answer = deletion_request.json()
    deleted_run_name = deletion_answer['run']
    return deleted_run_name

#affichage d'un plot comparatif d'une métrique choisie par l'utilisateur, de la pire à la meilleur pour les modèles
@st.cache(suppress_st_warning=True)
def metrics_plotter(df, metric):
    df_dict = df.to_dict()
    metric_plot_request_input = {'metric' : metric, 'data' : df_dict}
    metric_plot_request = requests.post(url = 'http://ocds7ey.herokuapp.com/experiments/runs/metrics/compare', data = json.dumps(metric_plot_request_input))
    metric_plot_answer = metric_plot_request.json()
    red = pd.DataFrame(metric_plot_answer['R']).values
    green = pd.DataFrame(metric_plot_answer['G']).values
    blue = pd.DataFrame(metric_plot_answer['B']).values
    last = pd.DataFrame(metric_plot_answer['A']).values
    rgb = np.dstack((red,green,blue,last))
    return rgb

#Récupération des roc_curves et confusion matrix logguées via mlflow pour une run données et affichage
@st.cache(suppress_st_warning=True)
def saved_figures_plotter(exp_name, run_name):
    figures_request_input = {'exp_name' : exp_name, 'run_name' : run_name}
    train_roc_request = requests.post(url='http://ocds7ey.herokuapp.com/experiments/runs/metrics/train_roc_curve', data = json.dumps(figures_request_input))
    test_roc_request = requests.post(url='http://ocds7ey.herokuapp.com/experiments/runs/metrics/test_roc_curve', data = json.dumps(figures_request_input))
    train_cm_request = requests.post(url='http://ocds7ey.herokuapp.com/experiments/runs/metrics/train_cm', data = json.dumps(figures_request_input))
    test_cm_request = requests.post(url='http://ocds7ey.herokuapp.com/experiments/runs/metrics/test_cm', data = json.dumps(figures_request_input))
    train_roc_answer = train_roc_request.json()
    test_roc_answer = test_roc_request.json()
    train_cm_answer = train_cm_request.json()
    test_cm_answer = test_cm_request.json()

    red1 = pd.DataFrame(train_roc_answer['R']).values
    green1 = pd.DataFrame(train_roc_answer['G']).values
    blue1 = pd.DataFrame(train_roc_answer['B']).values
    last1 = pd.DataFrame(train_roc_answer['A']).values
    train_roc = np.dstack((red1,green1,blue1,last1))

    red1 = pd.DataFrame(test_roc_answer['R']).values
    green1 = pd.DataFrame(test_roc_answer['G']).values
    blue1 = pd.DataFrame(test_roc_answer['B']).values
    last1 = pd.DataFrame(test_roc_answer['A']).values
    test_roc = np.dstack((red1,green1,blue1,last1))

    red1 = pd.DataFrame(train_cm_answer['R']).values
    green1 = pd.DataFrame(train_cm_answer['G']).values
    blue1 = pd.DataFrame(train_cm_answer['B']).values
    last1 = pd.DataFrame(train_cm_answer['A']).values
    train_cm = np.dstack((red1,green1,blue1,last1))    

    red1 = pd.DataFrame(test_cm_answer['R']).values
    green1 = pd.DataFrame(test_cm_answer['G']).values
    blue1 = pd.DataFrame(test_cm_answer['B']).values                                                                                                  
    last1 = pd.DataFrame(test_cm_answer['A']).values                
    test_cm = np.dstack((red1,green1,blue1,last1))
    return train_roc, test_roc, train_cm, test_cm

#Récupération des featues importances loggées pour une run en particulier (intérpetation globale du modèle)
@st.cache(suppress_st_warning=True)
def feature_importance(exp_name, run_name):
    importance_request_input = {'exp_name' : exp_name, 'run_name' : run_name}
    feature_importance_request = requests.post(url='http://ocds7ey.herokuapp.com/experiments/runs/metrics/feature_importance', data = json.dumps(importance_request_input))
    feature_importance_answer = feature_importance_request.json()
    red1 = pd.DataFrame(feature_importance_answer['R']).values
    green1 = pd.DataFrame(feature_importance_answer['G']).values
    blue1 = pd.DataFrame(feature_importance_answer['B']).values
    last1 = pd.DataFrame(feature_importance_answer['A']).values
    image = np.dstack((red1,green1,blue1,last1))
    return image

#Le premier expander est dédié à la comparaison des modèles au sein d'une expérience.
#Pour cela, l'utilisateur peut directement sélectionner une expérience, et peut sélectionner des métriques
#pour comparer les modèles au sein des différentes runs, ainsi que visualiser ces métriques.'''
with st.expander("**Liste des epxériences!**"):
    #Sélection d'une experience - requête de la liste des expériences et sélecteur
    nb_exp_request = requests.get(url='http://ocds7ey.herokuapp.com/experiment_start')
    names = nb_exp_request.json()['Names']
    exp = st.selectbox('**Veuillez Sélectionner une éxperience :**', names)
    st.write("Vous avez sélectionné l'expérience: **{}**".format(exp))

    #Boutton de suppression définitive d'une expérience
    if st.button("**Supprimer l'expérience séléctionnée!**"):
        st.warning("L'option de suppression des expériences est mise en suspend le temps de la soutenance pour conserver tous les résultats.")
        #deleted_exp = del_experiences(exp)
        #st.success("Expérience {} supprimée avec succès.".format(exp))

    #Check box pour le choix des métriques à afficher
    st.write('Veuillez sélectionner les métriques à afficher!')
    SCORE = st.checkbox('Custom-score', value=True)
    ACC = st.checkbox('Accuracy', value=True)
    REC = st.checkbox('Recall', value=True)
    F1 = st.checkbox('F1-Score', value=True)
    AUC = st.checkbox('ROC-AUC', value=True)
    BAL_ACC = st.checkbox('Balanced-Accuracy', value=True)
    metrics_df = metrics_displayer(exp, SCORE, ACC, REC, F1, AUC, BAL_ACC)
    st.dataframe(metrics_df) 

    #Métriques d'une run en particulier - sélection d'une run par l'utilisateur
    run = st.selectbox("**Veuillez choisir une run!**", np.setdiff1d(np.array(metrics_df.index), np.array(['DummyClassifier'])))
    st.write("Vous avez sélectionné la run: **{}**".format(run))

    #Possibilité de supprimer la run
    if st.button("**Supprimer la run séléctionnée!**"):
        st.warning("L'option de suppression des runs est mise en suspend le temps de la soutenance pour conserver tous les résultats.")
        #deleted_run = del_run(exp, run)
        #st.success('Run {} supprimée avec succès!'.format(deleted_run))

    #Affichage des confusion matrix et des roc_curves
    train_roc, test_roc, train_cm, test_cm = saved_figures_plotter(exp, run)
    col1, col2 = st.columns(2)
    with col1:
        st.image(train_roc)
    with col2:
        st.image(test_roc)
    col3, col4 = st.columns(2)
    with col3:
        st.image(train_cm)
    with col4:
        st.image(test_cm)

    #Comparaison des runs en fonction de la métrique définie par l'utilisateur (visualisation)
    st.markdown('**Sélection de la métrique:**')
    metric = st.selectbox('**Sélection de la métrique à comparer**', metrics_df.columns)
    full_metrics = metrics_displayer(exp, True, True, True, True, True, True)
    fig = metrics_plotter(full_metrics, metric)
    st.image(fig, use_column_width = 'always')

#Fonction permettant de retourner les métriques et hyperparamètres d'un modèle
#Pour cela, nous importons les résultats de la recherche sur grille logguée via mlflow
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def split_columns(df):
    dict_df = df.to_dict()
    param_request_input = {'metric' : 'any', 'data' : dict_df}
    param_request = requests.post(url='http://ocds7ey.herokuapp.com/experiments/runs/cv_results/param_selector', data = json.dumps(param_request_input))
    param_answer = param_request.json()
    metrics = param_answer['metrics']
    params = param_answer['params']
    return metrics, params

#Fonction permettant d'afficher le plot des résultats de la recherche sur grille pour l'optimisation
#des hyperparamètres. L'utilisateur sélectionne les hyperparamètres, et la métrique désirée, 
#et un plot permet de mesurer l'influence des hyperparamètres sur les métriques
@st.cache(suppress_st_warning=True)
def param_metric_plotter(df, param, metric, param_2):
    df_dict = df.to_dict()
    metric_param_input = {'data' : df_dict, 'param' : param, 'metric' : metric, 'param_2' : param_2}
    param_metric_request = requests.post(url='http://ocds7ey.herokuapp.com/experiments/runs/cv_results/param_plotter', data = json.dumps(metric_param_input))
    param_metric_answer = param_metric_request.json()
    red = pd.DataFrame(param_metric_answer['R']).values
    green = pd.DataFrame(param_metric_answer['G']).values
    blue = pd.DataFrame(param_metric_answer['B']).values
    last = pd.DataFrame(param_metric_answer['A']).values
    rgb = np.dstack((red,green,blue,last))
    return rgb

st.write(' ')
st.write("Dans le deuxième onglet, vous avez accès aux différentes visualisations permettant de comparer les valeurs des différents hyperparamètres.")
st.write("En sélectionnant des hyperparamètres, vous pouvez comparer leurs influences sur les mériques disponibles.")

#Cette expander se sert des fonctions précédentes afficher les plots comparatifs des hyperparamètres et des métriques.
#Pour cela, nous importons les résultats de la recherche sur grille par validation croisée, et nous
#plottons les valeurs des métriques en fonction de l'hyperparamètre choisi par l'utilisateur.

#Nous affichons également les résultats d'interpétation globale des modèles, via les features importances
#loggés via mlflow.'''
with st.expander('**Analyse des hyper_paramètres:**'):
    st.info("**ATTENTION** : Pour des raisons de ressources computationnelles limitées côté serveur, nous n'affichons pas les nuages de points des métriques en fonction des hyperparamètres dans cette version, ni l'interprétation globale des modèles.")
    #Requête de réception des résultats du gridsearch et affichage du tableau
    #cv_request_input = {'exp_name' : exp, 'run_name' : run}
    #cv_request_request = requests.post(url='http://ocds7ey.herokuapp.com/experiments/runs/cv_results', data = json.dumps(cv_request_input))
    #cv_answer = cv_request_request.json()
    #cv_df = pd.DataFrame(cv_answer)
    #st.dataframe(cv_df)

    
    #Pour des raisons de limitation des ressources computatinnelles côté serveur, nous n'affichons pas les features importanes et les plots ci-dessous sur la versoin light

    #Sélection des métriques et hyperparamètres, et affichage du plot
    #metrics, params = split_columns(cv_df)
    #X_param = st.selectbox("**Sélection du premier paramètre:**", params)
    #Y_param = st.selectbox("**Sélection du second paramètre:**", params)
    #metric_select = st.selectbox("**Sélection de la métrique:**", metrics)
    #param_metric_fig = param_metric_plotter(cv_df, X_param, metric_select, Y_param)
    #st.image(param_metric_fig, use_column_width=True)

    #Affichage des features importances
    #st.subheader('**Features importance du modèle séléctionné:**')
    #feature_importance_plot = feature_importance(exp, run)
    #st.image(feature_importance_plot)

