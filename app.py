'''Ce fichier contient le code de l'API de notre application.
Dans notre cas, nous avons fait le choix de réaliser cette API via la librairie FastAPI, à partir de laquelle nous allons crée les endpoints et URI
auquels le client addrssera les requêtes pour mettre en oeuvre l'application.'''

#Importation des librairies utiles

#Librairies relatives à l'API
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

#Librairies de manipulation des données
import pandas as pd
import numpy as np
import codecs #permet de d'importer des fichier html sous la forme de données textuelles - utile pour le DataDrift

#Libraires de gestion système
import os
import warnings
import sys
import gc
import mlflow
import json

#Librairies de visualisation des données
import matplotlib.pyplot as plt
from PIL import Image


#Librairies relatives aux modèles (pour accès à la liste de leurs hyperparamètres dans notre cas, et shap pour interprétation locale)
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import shap


#evidently
from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import *
from evidently.tests import *

#Mise en mémoire du chemin absolu du fichier actuel
script_path = os.getcwd()

#Ajout du folder src pour l'import des fonctions de nos différents fichiers
sys.path.append('src')
sys.path.append('input')
warnings.filterwarnings("ignore")
os.chdir('src')

#Importation des fonctions des fichiers src/train.py et src/utils.py
from src.utils import correlation_lister
from src.utils import Quantitative_Distribution_Plotter
from src.utils import Qualitative_Distribution_Plotter
from src.train import import_data
from src.train import pkl_import
from src.train import hyperparam_tuning
from src.train import smote_hyperparam_tuning
from src.utils import experiments_lister
from src.utils import run_lister
from src.utils import run_metrics
from src.utils import del_run
from src.utils import del_experiment
from src.utils import runs_plotter
from src.utils import cv_results_returner
os.chdir('..')

'''Dans cette partie, nous allons définir plusieurs classes pydatic de manière à fournir des inputs avec les bon types pour nos différentes fonctions.'''
class df_input(BaseModel):
    name : str
    column : str

class exp_input(BaseModel):
    name : str
    na_values : float
    na_strategy : str
    pearson_select : float
    description : str

class exp_col_input(BaseModel):
    name : str
    column : str
    exp_name : str

class model_type(BaseModel):
    type : str

class double_dict(BaseModel):
    dict_1 : dict
    dict_2 : dict

class int_grid(BaseModel):
    mini : int
    maxi : int
    step : int
    is_exp : bool

class float_grid(BaseModel):
    mini : float
    maxi : float
    step : float
    is_exp : bool

class hyper_params_tuning(BaseModel):
    train : dict
    target : dict
    halving_state : bool
    json_export : bool
    standard_state : bool
    pca_state : bool
    params : dict
    grid_params : dict
    model_type : str
    run_name : str
    exp_name : str

class display_metrics(BaseModel):
    exp_name : str
    SCORE : bool
    ACC : bool
    REC : bool
    F1 : bool
    AUC : bool
    BAL_ACC : bool

class exp_run(BaseModel):
    exp_name : str
    run_name : str

class run_metrics_data(BaseModel):
    metric : str
    data : dict

class param_metric_plot(BaseModel):
    data : dict
    param : str
    metric : str
    param_2 : str

class exp_run_data(BaseModel):
    exp_name : str
    run_name : str
    dataset : str

class exp_run_predict(BaseModel):
    exp_name : str
    run_name : str
    data : list

class exp_run_features(BaseModel):
    exp_name : str
    run_name : str
    dataset : str
    observation : float
    features : list

class exp_run_single_feature(BaseModel):
    exp_name : str
    run_name : str
    dataset : str
    features : str

'''Enfin, nous pouvons mettre en pklace notre API via FastAPI.'''
app = FastAPI()

#Fonction de test initial de l'API
@app.post('/')
def get():
    return {'message' : 'OpenClassrooms - Projet 7 - Creéz un modèle de scoring!'}

'''Fonction permettant d'importer les noms des données initiales via la méthode GET.
Pour cela, nous nous rendont dans le répertoire input, et nous listaons les fichiers .CSV présents.'''
@app.get('/initial_data_select')
def data_selection():
    os.chdir(script_path)
    expander_options = np.array([w[:-4] for w in os.listdir('input')])
    options_dict = {}
    for i in expander_options:
        options_dict[i] = i
    del expander_options
    gc.collect()
    return options_dict

'''Chargement et echantillonnage d'un jeu de donné sélectionné - 500 individus pour gagner en temps d'affihage côté client.'''
@app.post('/initial_data_select/to_df')
def df_display(input:df_input):
    os.chdir(script_path)
    #Récupération du fichier séléctionné côté client
    tmp = 'input/' + str(input.dict()['name']) + '.csv'
    df = pd.read_csv(tmp)
    #Echentillonnage
    df_samp = df.sample(500).fillna(-15000)
    #Mise sous forme dictionnaire pour envoie au client
    df_dict = df_samp.to_dict()
    del tmp, df_samp, df
    gc.collect()
    return df_dict

'''Création d'un plot représentant la distribution d'une variable d'un jeu de données initial.'''
@app.post('/initial_data_select/to_figure')
def figure_maker(input:df_input):
    os.chdir(script_path)
    data = input.dict()

    #Lecture du dataframe
    tmp = 'input/' + str(data['name']) + '.csv'
    df = pd.read_csv(tmp)
    #Séléction de la colonne
    col = data['column']

    #Création du plot en fonction du type de la colonne (type object = qualitative - type float = quantitative)
    if df[col].dtype == 'object':
        fig = Qualitative_Distribution_Plotter(df, col)
    else:
        fig = Quantitative_Distribution_Plotter(df, col)

    #Enregistrement du plot sous forme d'image et chargement de l'image
    image = fig.savefig('fig.png', bbox_inches='tight')
    image2 = Image.open('fig.png')
    im_arr = np.asarray(image2)
    os.remove('fig.png')

    #Séparation des signaux RGBA de l'image pour transfert au client sous forme d'un dictionnaire
    h1 = pd.DataFrame(im_arr[:,:,0]).to_dict()
    h2 = pd.DataFrame(im_arr[:,:,1]).to_dict()
    h3 = pd.DataFrame(im_arr[:,:,2]).to_dict()
    h4 = pd.DataFrame(im_arr[:,:,3]).to_dict()
    final_dict = {'R':h1, 'G':h2, 'B':h3, 'A':h4}
    del h1, h2, h3, h4, im_arr, image2, image, fig, col, df, tmp, data
    gc.collect()
    return final_dict

'''Mise en place de la distribution des coeficients de Pearson au sein d'un jeu de données'''
@app.post('/initial_data_select/corelations_plotter')
def correlation_maker(input:df_input):
    os.chdir(script_path)
    data = input.dict(include = {'name'})

    #Lecture du dataframe
    tmp = 'input/' + str(data['name']) + '.csv'
    df = pd.read_csv(tmp)

    #Création du plot et mise sous forme d'image au format.png
    fig, corr = correlation_lister(df)
    image = fig.savefig('fig.png', bbox_inches='tight')
    image2 = Image.open('fig.png')
    im_arr = np.asarray(image2)
    os.remove('fig.png')

    #Séparation des 4 signaux RGBA pour envoi au client
    h1 = pd.DataFrame(im_arr[:,:,0]).to_dict()
    h2 = pd.DataFrame(im_arr[:,:,1]).to_dict()
    h3 = pd.DataFrame(im_arr[:,:,2]).to_dict()
    h4 = pd.DataFrame(im_arr[:,:,3]).to_dict()
    final_dict = {'R':h1, 'G':h2, 'B':h3, 'A':h4}
    del h1, h2, h3, h4, im_arr, image2, image, fig, df, tmp, data
    gc.collect()
    return final_dict

'''Liste des expériences déjà crées via une requête de type GET'''
@app.get('/experiment_start')
def experiment_starer():
    os.chdir(script_path)
    #Liste des experiences crées
    exp_dict = experiments_lister()

    #Récupération des noms des ancienes expériences
    names = list(exp_dict.keys())
    result = {'Count' : int(len(names)), 'Names' : names}
    del exp_dict, names
    gc.collect()
    return result

'''Création d'une nouvelle expérience MLFlow en générant un nouveau jeu de données (Train, Target et Production -- ici nommé test).
Pour cela, nous faison appel à la fonction import_data de src/train.py, nous permettant de créer des jeux de données customisés en fonction des seuils de valeurs manquantes, 
de corrélation, et des stratégies de complétion des valeurs manquantes.
Un Dummy Classifieur est automatiquement loggué via MLFlow dans la toute première run de l'expérience, et les métriques sot mesurées dessus.'''
@app.post('/experiment_start/create')
def experiment_starter(input:exp_input):
    os.chdir(script_path)
    data = input.dict()

    #Récupération du nom de l'expérience et des seuils de valeurs manquantes, corrélation et stratégie de complétion des valeurs manquantes
    #Ainsi que la description de l'expérience, logguée en tag.
    name = data['name']
    na_values = data['na_values']
    na_strategy = data['na_strategy']
    pearson_select = data['pearson_select']
    description = data['description']

    #Utilisation de la fonction import_data
    train, test, target, exp_id = import_data(True, na_values, pearson_select, na_strategy, name, description)

    #Chargeent des jeux de données crées et passage en dictionnaires
    train, test, target, exp_id, exp_name = pkl_import()   
    final_dict = {'id' : exp_id, 'name' : exp_name}
    del data, name, na_values, na_strategy, pearson_select, description, train, test, target, exp_id, exp_name
    gc.collect()
    return final_dict

'''Fonction permettant d'afficher les jeux de données echantillonnés côté client.
Séléction de l'expérience, et lecture des jeux de données pour les afficher.'''
@app.post('/experiment_start/select')
def experiment_selector(input_name:exp_input):
    os.chdir(script_path)
    data = input_name.dict()
    #Lecture du nom de l'expérience sélectionnée et récupération de son ID
    exp_name = data['name']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]

    #Boucle sur les répertoires de l'expérience, jusqu'à trouver les 3 fichiers avec succès
    for i in os.listdir('../mlruns/' + str(exp_id)):
        if i != 'meta.yaml':
            try:
                os.chdir('../mlruns/' + str(exp_id) + '/' + str(i) + '/artifacts/preprocessed')
                if len(os.listdir()) == 3:
                    break
            except:
                pass
    train = pd.read_pickle('train.pkl')
    test = pd.read_pickle('test.pkl')
    target = pd.read_pickle('target.pkl')
    os.chdir('../../../../../src/preprocessed')

    #Export des fichiers dans repértoire tampon et sample - les valeurs manquantes éventuelles sont remplacées par une valeur abérante
    #et ensuite re-remplacées par npn.nan côté client pour permettre le passage du serveur au client
    for i in os.listdir():
        os.remove(i)
    train.to_pickle('train.pkl')
    test.to_pickle('test.pkl')
    target.to_pickle('target.pkl')
    samp = np.min([len(train), len(test), 500])
    train = train.sample(samp)
    target = target.sample(samp)
    test = test.sample(samp)
    os.chdir('..')
    EXP_DICT = {'ID': str(exp_dict[exp_name]), 'NAME' : str(exp_name)}
    with open('mlflow_exp/exp_id.json', 'w') as fp:
            json.dump(EXP_DICT, fp)
    os.chdir('..')
    mlflow.set_experiment(exp_name)

    #Transformation des jeux de données en dictionnaires pour passage au client
    train = train.to_dict()
    test = test.to_dict()
    target = target.to_dict()
    final_dict = {'train' : train, 'test' : test, 'target' : target, 'id' : exp_id, 'name' : exp_name}
    os.chdir(script_path)
    del data, exp_name, exp_dict, exp_id, train, test, target, EXP_DICT, samp
    gc.collect()
    return final_dict

'''Visualisation de la distribution d'une colonne d'un jeu de données sélectionné, avec variantes en fonction du type de colonne.'''
@app.post('/experiment_start/select/to_figure')
def figure_maker(input:exp_col_input):
    os.chdir(script_path)
    data = input.dict()
    #Chargement des jeux de données (cf. fonction précédente) et lecture de la colonnes demandée par le client
    df_name = data['name']
    col = data['column']
    exp_name = data['exp_name']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('../mlruns/' + str(exp_id))
    for i in os.listdir():
        if i != 'meta.yaml':
            try:
                os.chdir(str(i) + '/artifacts/preprocessed')
                if len(os.listdir()) == 3:
                    break
            except:
                pass
    if df_name == 'train':
        df = pd.read_pickle('train.pkl')
    elif df_name == 'test':
        df = pd.read_pickle('test.pkl')
    else:
        df = pd.read_pickle('target.pkl')

    #Création des plots en fonction du type de la variable considérée (PieChart pour variable qualitative - HistPlot pour quantitative)
    if len(df[col].unique()) < 3:
        fig = Qualitative_Distribution_Plotter(df, col)
    else:
        fig = Quantitative_Distribution_Plotter(df, col)

    #Sauvegarde du plot en image et réimport avec séparation des 4 signaux RGBA (et suppression de l'image enregistrée)
    image = fig.savefig('fig.png', bbox_inches='tight')
    image2 = Image.open('fig.png')
    im_arr = np.asarray(image2)
    os.remove('fig.png')
    h1 = pd.DataFrame(im_arr[:,:,0]).to_dict()
    h2 = pd.DataFrame(im_arr[:,:,1]).to_dict()
    h3 = pd.DataFrame(im_arr[:,:,2]).to_dict()
    h4 = pd.DataFrame(im_arr[:,:,3]).to_dict()

    #Envoi des 4 signaux dans un dictionnaire (JSON serializable)
    final_dict = {'R':h1, 'G':h2, 'B':h3, 'A':h4}
    del h1, h2, h3, h4, im_arr, image2, image, fig, df, data, df_name, col, exp_name, exp_dict, exp_id
    gc.collect()
    return final_dict


'''Liste des runs d'une expérience donnée, et renvoi de la liste des runs au client'''
@app.post('/experiment_start/select/runs')
def runs_displayer(input:exp_col_input):
    os.chdir(script_path)
    data = input.dict()
    exp_name = data['exp_name']
    exp_dict = experiments_lister()
    os.chdir('..')
    runs = run_lister(exp_dict[exp_name])
    del data, exp_name, exp_dict
    gc.collect()
    return runs

'''Côté client, pour entraîner un modèle, nous sélectionnons un classifieur.
Côté serveur, en fonction du classifieur demandé, créons un classifieur pour extraire la liste des hyperparamètres, et les renvoyer au client.
De cette manière, le client a une meilleur visibilité sur la gestion des hyperparamètres qu'il veut mettre en place.'''
@app.post('/experiment_start/select/runs/model_selection')
def model_selector(input:model_type):
    os.chdir(script_path)
    data = input.dict()
    #Lecture du type de modèle et création d'un modèle sans entraînement
    model_type = data['type']
    if model_type == 'GradientBooster':
        model = GradientBoostingClassifier()
    elif model_type == 'HistGradientBooster':
        model = HistGradientBoostingClassifier()
    elif model_type == 'LightGBM':
        model = LGBMClassifier()
    else:
        model = LogisticRegression()

    #Récupération de la liste des hyperparamètres et leur valeurs par défaut puis renvoi au client
    dict_params = model.get_params()
    del data, model_type, model
    gc.collect()
    return dict_params

'''Le client a le choix d'entrer des hyperparamètres manuellement pour son classifieur.
Ici, ces hyperparamètres sont récupérés via les differents input_displayers côté client, et regroupés dans un dictionaire.'''
@app.post('/experiment_start/select/runs/model_selection/manual_params')
def model_manual_params(input:double_dict):
    os.chdir(script_path)
    data = input.dict()
    dict_1 = data['dict_1']
    dict_2 = data['dict_2']
    tmp_dict = dict_1.copy()
    for i in tmp_dict.keys():
        if i in dict_2.keys():
            del dict_1[i]
    del dict_2, data, tmp_dict
    gc.collect()
    return dict_1

'''Pour l'optimisation des hyperparamètres par validation croisée, le client viens séléctionner les valeurs maximales et
minimales, ainsi que le nombre de steps et le type d'espacement (linspace ou logspace entre valeurs mi et max).
Ici, nous lisons ces valeurs, et nous créeons la liste de valeurs  à tester pour les hyperparamètres à optilmiser.'''
@app.post('/experiment_start/select/runs/model_selection/int_params')
def int_grid_maker(input:float_grid):
    os.chdir(script_path)
    #Lecture des valeurs min, max et le nombre de valeurs
    data = input.dict()
    mini = data['mini']
    maxi = data['maxi']
    step = data['step']
    is_exp = data['is_exp']

    #Création de la grille de valeurs en fonction du type de scaling
    if is_exp == True:
        grid = list(np.logspace(mini, maxi, num = int(step)) - 1)
    else:
        grid = list(np.linspace(mini, maxi, num = int(step)))

    #Création du dictionnaire avec les valeurs de la grille
    final_dict = {'grid' : grid}
    del data, mini, maxi, step, is_exp, grid
    gc.collect()
    return final_dict


'''Ici, nous réalisons la même opération que précédemment, mais pour les hyperparamètres de type float.
La distinction est faite via les deux URI différentes, car la gestion des hyperparamètres de type integer est plus compliquée.
N'étant pas JSON serializable, les hyperparmètres de type integer sont convertis en float côté client,et reconvertis en integers 
pour l'entraînement côté serveur.'''
@app.post('/experiment_start/select/runs/model_selection/float_params')
def float_grid_maker(input:float_grid):
    os.chdir(script_path)
    data = input.dict()
    mini = data['mini']
    maxi = data['maxi']
    step = data['step']
    is_exp = data['is_exp']
    if is_exp == True:
        grid = list(np.logspace(mini, maxi, num = int(step)) - 1)
    else:
        grid = list(np.linspace(mini, maxi, num = int(step)))
    final_dict = {'grid' : grid}
    del data, mini, maxi, step, is_exp, grid
    gc.collect()
    return final_dict

'''Ici, nous mergeons l'ensemble des grilles précédemment crée en un seul dictionnaire. Cela permettra par la suite de créer
un seul dictionnaire à passer en argument de la fonction GridSearchCV du module model_selection de scikit-learn.'''
@app.post('/experiment_start/select/runs/model_selection/merge_params')
def full_grid_maker(input:double_dict):
    data = input.dict()
    dict_1 = data['dict_1']
    dict_2 = data['dict_2']
    tmp_dict = dict_1.copy()
    for i in tmp_dict.keys():
        if i in dict_2.keys():
            dict_1[i] = dict_2[i][0]
    del data, dict_2, tmp_dict
    gc.collect()
    return dict_1

'''Enfin, cette dernière fonction liée à l'URI d'optimisation des hyperparamètres permet d'entraîner des modèles en fonction
des choix du client du point de vue:
- De la gestion du déséquilibre des classes de la target.
- Des choix de préprocesseurs
- Du classifieur choisis
- De la liste des hyperparamètres entrés manuellement, et des grilles d'hyperparamètres à optimiser.
Cela appelera la fonction hyper_param_tuning ou smote_hyper_param_tuning du fichier src/train.py'''
@app.post('/experiment_start/select/runs/model_selection/hyper_param_tuning')
def hyper_param_tuning(input:hyper_params_tuning):
    os.chdir(script_path)
    data = input.dict()
    #Choix des jeux de données - non utilisé ici
    train = data['train']
    target = data['target']
    #Choix du type de gestion du déséquilibre des classes (anciennement halving state pour choix du type de recherche sur grille)
    halving_state = data['halving_state']
    json_export = data['json_export']
    #Choix du type de préprocesseur
    standard_state = data['standard_state']
    pca_state = data['pca_state']
    #Choix des hyperparamètres choisis
    params = data['params']
    #Grilles d'hyperparamètres à optimiser
    grid_params = data['grid_params']
    #Type de classifieur
    model_type = data['model_type']
    #Nom de la run et de l'expérience
    run_name = data['run_name']
    exp_name = data['exp_name']
    #Chargement des jeux de données train et target
    exp_dict = experiments_lister()
    os.chdir('..')
    exp_id = exp_dict[exp_name]
    for i in os.listdir('mlruns/' + exp_id):
        try:
            train_df = pd.read_pickle('mlruns/' + exp_id + '/' + i + '/artifacts/preprocessed/train.pkl')
            target_df = pd.read_pickle('mlruns/' + exp_id + '/' + i + '/artifacts/preprocessed/target.pkl')
            break
        except:
            pass
    #train_df = pd.DataFrame(train) -- ancienne méthode par envoi par le client
    #target_df = pd.DataFrame(target)
    #Gestion des types des colonnes pour distinction colonnes catégoriques et numériques (seuls les colonnes catégoriques ont moins de deux valeurs uniques
    # à cause du One-Hot-Encoder)
    os.chdir('src')
    for i in train_df.columns:
        if len(train_df[i].unique()) < 3:
            train_df[i] = train_df[i].astype(object)
    #Appel de l'une ou l'autre des fonctions d'entraînement en fonction de la gestion du déséquilibre des classes choisie par l'utilisateur
    if halving_state:
        best_params, run_name = hyperparam_tuning(train_df, target_df, json_export, standard_state, pca_state, params, grid_params, model_type, run_name, exp_name, exp_id)
    else:
        best_params, run_name = smote_hyperparam_tuning(train_df, target_df, json_export, standard_state, pca_state, params, grid_params, model_type, run_name, exp_name, exp_id)
    #Envoi du nom de la run et des meilleurs hyperparamètres au client
    final_dict = {'params' : best_params, 'run_name' : run_name}
    del exp_dict, data, train, target, train_df, target_df, halving_state, json_export, standard_state, pca_state, params, grid_params, model_type, run_name, exp_name, exp_id
    gc.collect()
    return final_dict

'''Permet de supprimer une expérience via un sélecteur de nom d'expérience, et un boutton côté client. (cf src/utils.py
pour fonctionnement des fonctions experiments_lister et del_experiment)'''
@app.post('/experiment_start/delete')
def experiment_deleter(input:exp_col_input):
    os.chdir(script_path)
    mlflow.end_run()
    data = input.dict()
    exp_name = data['exp_name']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    del_experiment(exp_id)
    final_dict = {}
    final_dict['exp'] = exp_name
    del data, exp_name, exp_dict, exp_id
    gc.collect()
    return final_dict

'''Permet d'envoyer au client les métriques de toutes les runs d'une expérience, en fonction des métriques demandées.
Les demandes de métriques se font côté client via des checkboxes, renvoyant une valeurs booléenne.
Pour chaque métrique demandée, nous chargeons les métriques des runs en train et test, et retournons un dataframe converti en dictionnaire.'''    
@app.post('/experiments/runs/metrics')
def run_selector(input:display_metrics):
    os.chdir(script_path)
    data = input.dict()
    #Lecture du nom de l'expérience demandée, et des valeurs booléennes des métriques demandées.
    exp_name = data['exp_name']
    SCORE = data['SCORE']
    ACC = data['ACC']
    REC = data['REC']
    F1 = data['F1']
    AUC = data['AUC']
    BAL_ACC = data['BAL_ACC']

    #Liste des runs de l'expérience
    mlflow.set_experiment(exp_name)
    exp_dict = experiments_lister()
    os.chdir('..')
    run_dict = run_lister(exp_dict[exp_name])
    run_names = list(run_dict.keys())
    os.chdir(script_path)
    all_metrics_df = pd.DataFrame()

    #Utilisation de la fonction run_metrics pour récupérer les métriques de chaque run
    for run in run_names:
        tmp = run_metrics(exp_dict[exp_name], run_dict[str(run)], str(run))
        all_metrics_df = pd.concat([all_metrics_df, tmp])
    final_metrics_df = pd.DataFrame()

    #Filtre sur les métriques demandées
    if SCORE:
        final_metrics_df = pd.concat([final_metrics_df, all_metrics_df[['Test_custom_score', 'Train_custom_score']]], axis = 1)
    if ACC:
        final_metrics_df = pd.concat([final_metrics_df, all_metrics_df[['Test_ACC', 'Train_ACC']]], axis = 1)
    if REC:
        final_metrics_df = pd.concat([final_metrics_df, all_metrics_df[['Test_REC', 'Train_REC']]], axis = 1)
    if F1:
        final_metrics_df = pd.concat([final_metrics_df, all_metrics_df[['Test_F1', 'Train_F1']]], axis = 1)
    if AUC:
        final_metrics_df = pd.concat([final_metrics_df, all_metrics_df[['Test_AUC', 'Train_AUC']]], axis = 1)
    if BAL_ACC:
        final_metrics_df = pd.concat([final_metrics_df, all_metrics_df[['Test_balanced_ACC', 'Train_balanced_ACC']]], axis = 1)
    final_metrics_dict = final_metrics_df.fillna(-1).to_dict()
    del tmp, all_metrics_df, final_metrics_df, data, exp_name, SCORE, ACC, REC, F1, AUC, BAL_ACC, exp_dict, run_dict, run_names
    gc.collect()
    return final_metrics_dict

'''Permet de supprimer une run d'une expérience à la demande du client via un sélecteur de run et un bouton'''
@app.post('/experiments/runs/delete')
def experiment_deleter(input:exp_run):
    os.chdir(script_path)
    mlflow.end_run()
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_dict[exp_name])
    run_id = run_dict[run_name]
    del_run(exp_id, run_id)
    final_dict = {}
    final_dict['run'] = run_name
    del data, exp_name, exp_dict, exp_id
    gc.collect()
    return final_dict

'''Permet d'envoyer au client un plot comparatif d'une métrique en particulier pour les différentes runs d'une expérience'''
@app.post('/experiments/runs/metrics/compare')
def metrics_plotter(input:run_metrics_data):
    os.chdir(script_path)
    data = input.dict()
    #Sélection de la métrique
    metric = data['metric']
    #Lecture des métriques
    data_base = data['data']
    data_df = pd.DataFrame(data_base)
    #Création du plot comparatif - passage en image.png et décomposition des signaux
    fig = runs_plotter(data_df, metric)
    image = fig.savefig('fig.png', bbox_inches='tight')
    image2 = Image.open('fig.png')
    im_arr = np.asarray(image2)
    os.remove('fig.png')
    h1 = pd.DataFrame(im_arr[:,:,0]).to_dict()
    h2 = pd.DataFrame(im_arr[:,:,1]).to_dict()
    h3 = pd.DataFrame(im_arr[:,:,2]).to_dict()
    h4 = pd.DataFrame(im_arr[:,:,3]).to_dict()
    #Envoi des signaux de l'image sous forme d'un dictionnaire
    final_dict = {'R':h1, 'G':h2, 'B':h3, 'A':h4}
    del data, metric, data_base, data_df, fig, image, image2, im_arr, h1, h2, h3, h4
    gc.collect()
    return final_dict

'''Envoi des résultats de l'optimisation des hyperparamètres par recherche sur grille d'une run enparticulier'''
@app.post('/experiments/runs/cv_results')
def cv_results(input:exp_run):
    os.chdir(script_path)
    data = input.dict()
    #Lecture de la run et de l'expérience demandée
    exp_name = data['exp_name']
    run_name = data['run_name']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    #Récupération des résultats de la recherche sur grille via la fonction cv_results_returner de src/utils.py et suppression des colonnes non pertinantes
    df = cv_results_returner(exp_id, run_id)
    df = df.drop(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params'], axis = 1)
    for i in df.columns:
        if 'split' in i:
            df = df.drop([i], axis = 1)
        if 'rank' in i:
            df = df.drop([i], axis = 1)
        if 'std_' in i:
            df = df.drop([i], axis = 1)
    #Envoi au client
    df_dict = df.to_dict()
    del data, exp_name, run_name, exp_dict, exp_id, run_dict, run_id, df
    gc.collect()
    return df_dict

'''Permet de sélectionner des hyperparamètres et des métriques pour visualiser l'influence des hyperparamètres sur les métriques plus tard'''
@app.post('/experiments/runs/cv_results/param_selector')
def param_selector(input:run_metrics_data):
    os.chdir(script_path)
    data = input.dict()
    columns = data['data']
    param_columns = []
    metric_columns = []
    for i in columns:
        if 'param' in i:
            param_columns.append(i)
        else:
            metric_columns.append(i)
    final_dict = {'params' : param_columns, 'metrics' : metric_columns}
    del data, columns, param_columns, metric_columns
    gc.collect()
    return final_dict

'''Permet de visualiser l'importance des hyperparamètres sur les métriques via un plot'''
@app.post('/experiments/runs/cv_results/param_plotter')
def param_plotter(input:param_metric_plot):
    os.chdir(script_path)
    #Choix de l'hyperparamètre et de la métrique à comparer
    data = input.dict()
    df = pd.DataFrame(data['data'])
    X = data['param']
    Y = data['metric']
    hue = data['param_2']

    #Création du plot à partir des résultats d'ptimisation par validation croisée
    fig = plt.figure(figsize = (10,8))
    for i in df[hue].unique():
        tmp = df.loc[df[hue] == i]
        label_str = hue + '_=_' + str(i)
        plt.scatter(tmp[X].values, tmp[Y].values, label = label_str)
    plt.title('Scatterplot de la métrique {} \n en fonction du paramètre \n{}'.format(Y, X),
              size = 20, weight = 'bold')
    plt.legend(bbox_to_anchor=(1.4, 0.5), shadow=True)
    plt.xlabel('{}'.format(X), size = 13, weight = 'bold')
    plt.ylabel('{}'.format(Y), size = 13, weight = 'bold')
    plt.grid(True, ls = '--')

    #Renvoi du plot sous la forme d'une image
    image = fig.savefig('fig.png', bbox_inches='tight')
    image2 = Image.open('fig.png')
    im_arr = np.asarray(image2)
    os.remove('fig.png')
    h1 = pd.DataFrame(im_arr[:,:,0]).to_dict()
    h2 = pd.DataFrame(im_arr[:,:,1]).to_dict()
    h3 = pd.DataFrame(im_arr[:,:,2]).to_dict()
    h4 = pd.DataFrame(im_arr[:,:,3]).to_dict()
    final_dict = {'R':h1, 'G':h2, 'B':h3, 'A':h4}
    del h1, h2, h3, h4, data, df, X, Y, hue, fig, tmp, label_str, image, image2, im_arr
    gc.collect()
    return final_dict


'''Charge la train confusion matrix d'une run pour envoi au client'''
@app.post('/experiments/runs/metrics/train_cm')
def train_confusion_matrix(input:exp_run):
    os.chdir(script_path)
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    path = '../mlruns/' + exp_id + '/' + run_id + '/artifacts/metrics_figures/train_conf_matrix.png'
    image = Image.open(path)
    im_arr = np.asarray(image)
    h1 = pd.DataFrame(im_arr[:,:,0]).to_dict()
    h2 = pd.DataFrame(im_arr[:,:,1]).to_dict()
    h3 = pd.DataFrame(im_arr[:,:,2]).to_dict()
    h4 = pd.DataFrame(im_arr[:,:,3]).to_dict()
    final_dict = {'R':h1, 'G':h2, 'B':h3, 'A':h4}
    del h1, h2, h3, h4, data, exp_name, run_name, exp_dict, exp_id, run_dict, run_id, path, image, im_arr
    gc.collect()
    return final_dict

'''Charge la test confusion matrix d'une run pour envoi au client'''
@app.post('/experiments/runs/metrics/test_cm')
def test_confusion_matrix(input:exp_run):
    os.chdir(script_path)
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    path = '../mlruns/' + exp_id + '/' + run_id + '/artifacts/metrics_figures/test_conf_matrix.png'
    image = Image.open(path)
    im_arr = np.asarray(image)
    h1 = pd.DataFrame(im_arr[:,:,0]).to_dict()
    h2 = pd.DataFrame(im_arr[:,:,1]).to_dict()
    h3 = pd.DataFrame(im_arr[:,:,2]).to_dict()
    h4 = pd.DataFrame(im_arr[:,:,3]).to_dict()
    final_dict = {'R':h1, 'G':h2, 'B':h3, 'A':h4}
    del h1, h2, h3, h4, data, exp_name, run_name, exp_dict, exp_id, run_dict, run_id, path, image, im_arr
    gc.collect()
    return final_dict

'''Charge la train roc_curve d'une run pour envoi au client'''
@app.post('/experiments/runs/metrics/train_roc_curve')
def train_roc_curve(input:exp_run):
    os.chdir(script_path)
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    path = '../mlruns/' + exp_id + '/' + run_id + '/artifacts/metrics_figures/train_roc_curve.png'
    image = Image.open(path)
    im_arr = np.asarray(image)
    h1 = pd.DataFrame(im_arr[:,:,0]).to_dict()
    h2 = pd.DataFrame(im_arr[:,:,1]).to_dict()
    h3 = pd.DataFrame(im_arr[:,:,2]).to_dict()
    h4 = pd.DataFrame(im_arr[:,:,3]).to_dict()
    final_dict = {'R':h1, 'G':h2, 'B':h3, 'A':h4}
    del h1, h2, h3, h4, data, exp_name, run_name, exp_dict, exp_id, run_dict, run_id, path, image, im_arr
    gc.collect()
    return final_dict

'''Charge la test roc_curve d'une run pour envoi au client'''
@app.post('/experiments/runs/metrics/test_roc_curve')
def test_roc_curve(input:exp_run):
    os.chdir(script_path)
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    path = '../mlruns/' + exp_id + '/' + run_id + '/artifacts/metrics_figures/test_roc_curve.png'
    image = Image.open(path)
    im_arr = np.asarray(image)
    h1 = pd.DataFrame(im_arr[:,:,0]).to_dict()
    h2 = pd.DataFrame(im_arr[:,:,1]).to_dict()
    h3 = pd.DataFrame(im_arr[:,:,2]).to_dict()
    h4 = pd.DataFrame(im_arr[:,:,3]).to_dict()
    final_dict = {'R':h1, 'G':h2, 'B':h3, 'A':h4}
    del h1, h2, h3, h4, data, exp_name, run_name, exp_dict, exp_id, run_dict, run_id, path, image, im_arr
    gc.collect()
    return final_dict

'''Charge les feature importance d'une run d'une expérience en particulier pour envoi au client.'''
@app.post('/experiments/runs/metrics/feature_importance')
def feature_importance(input:exp_run):
    os.chdir(script_path)
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    path = '../mlruns/' + exp_id + '/' + run_id + '/artifacts/feature_importance/feature_importances.png'
    image = Image.open(path)
    im_arr = np.asarray(image)
    h1 = pd.DataFrame(im_arr[:,:,0]).to_dict()
    h2 = pd.DataFrame(im_arr[:,:,1]).to_dict()
    h3 = pd.DataFrame(im_arr[:,:,2]).to_dict()
    h4 = pd.DataFrame(im_arr[:,:,3]).to_dict()
    final_dict = {'R':h1, 'G':h2, 'B':h3, 'A':h4}
    del h1, h2, h3, h4, data, exp_name, run_name, exp_dict, exp_id, run_dict, run_id, path, image, im_arr
    gc.collect()
    return final_dict

'''Importe un jeu de donnée d'une run, d'une expérience pour affichage en production, nous pouvons charger le jeu de donnée de train, de tets ou de production (sans targets)'''
@app.post('/production/import_data')
def import_data_prod(input:exp_run_data):
    os.chdir(script_path)
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    dataset = data['dataset']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    os.chdir('..')
    mlflow.set_experiment(exp_name)
    if dataset == 'Production':
        path = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/preprocessed/test.pkl'
    elif dataset == 'Test':
        path = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/train_test/X_test.pkl'
    else:
        path = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/train_test/X_train.pkl'
    df = pd.read_pickle(path).reset_index()
    df_dict = df.to_dict()
    del data, exp_name, run_name, exp_dict, exp_id, run_dict, run_id, path, df, dataset
    gc.collect()
    return df_dict

'''Cette fonction permet de se concentrer sur un indvidu et des variables choisis en production.'''
@app.post('/production/import_data/features_select')
def import_data_feature(input:exp_run_features):
    os.chdir(script_path)
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    dataset = data['dataset']
    features = data['features']
    observation = int(data['observation'])
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    os.chdir('..')
    mlflow.set_experiment(exp_name)
    if dataset == 'Production':
        path = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/preprocessed/test.pkl'
    elif dataset == 'Test':
        path = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/train_test/X_test.pkl'
    else:
        path = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/train_test/X_train.pkl'
    
    #Projection sur l'individu choisi, et les features choisis côté client
    df = pd.read_pickle(path).reset_index()
    df = df.loc[df.index == observation, features]
    df_dict = df.to_dict()
    del data, exp_name, run_name, exp_dict, exp_id, run_dict, run_id, path, df, dataset, features, observation
    gc.collect()
    return df_dict


'''Renvoi la distribution d'une colonne en particulir en production pour affichage côté client.'''
@app.post('/production/import_data/features_select/distribution')
def feature_distribution(input:exp_run_single_feature):
    os.chdir(script_path)
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    dataset = data['dataset']
    features = data['features']
    exp_dict = experiments_lister()
    os.chdir('..')
    exp_id = exp_dict[exp_name]
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    os.chdir('..')
    mlflow.set_experiment(exp_name)
    if dataset == 'Production':
        path = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/preprocessed/test.pkl'
    elif dataset == 'Test':
        path = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/train_test/X_test.pkl'
    else:
        path = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/train_test/X_train.pkl'
    df = pd.read_pickle(path).reset_index()
    value_counts = pd.DataFrame(df[features].value_counts()).to_dict()
    del data, exp_name, run_name, dataset, features, exp_dict, exp_id, run_dict, run_id, path, df
    gc.collect()
    return value_counts

'''Chargement d'un modèle lié à une run, et utilisation du modèle en inférence. Tous les modèles sont loggés comme artéfact dans un répertoire portant le nom exp_name + run_name + classifier.
Utilisation de la méthode load_model de MLFlow Model pour cela. L'input est envoyé par le client au serveur. Le serveur renvoie les résultats.'''
@app.post('/production/inference')
def model_server(input:exp_run_predict):
    os.chdir(script_path)
    #Sélection de l'expérience et la run
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    prediction_input = np.array(data['data']).reshape(1, -1)
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    os.chdir('..')

    #Définition de l'expéience mlflow et chemin vers le modèle
    mlflow.set_experiment(exp_name)
    path = "mlruns/" + exp_id + '/' + run_id + '/artifacts/' + exp_name + '_' + run_name + '_Classifier'
    try:
        model = mlflow.sklearn.load_model(path)
    except:
        model = mlflow.lightgbm.load_model(path)

    #Sélection de l'individu à prédire et prédiction avec les méthode predict et predict_roba (seuil à 0.5)
    features = pd.read_pickle("mlruns/" + exp_id + '/' + run_id + '/artifacts/preprocessed/train.pkl').columns
    best_thresh = open("mlruns/" + exp_id + '/' + run_id + '/params/classifier__threshold').read()
    best_thresh = float(best_thresh)
    score = model.predict_proba(prediction_input)[0][1]
    prediction = (score > best_thresh).astype(float)
    for i in os.listdir('predictions_tmp'):
        os.remove('predictions_tmp/' + i)
    tmp_df = pd.DataFrame(prediction_input, columns = features)

    #Enregistrement de la prédiction via mlflow pour mise en place de suivi des performances du modèle
    tmp_df['prediction'] = prediction
    tmp_df['score'] = score
    tmp_df.to_csv('predictions_tmp/prediction.csv', index = False)
    try:
        n = len(os.listdir("mlruns/" + exp_id + '/' + run_id + '/artifacts/predictions'))
    except:
        n = 0
    with mlflow.start_run(run_id=run_id, experiment_id=exp_id):
        mlflow.log_artifacts('predictions_tmp', artifact_path='predictions/' + 'pred_' + str(n))
    mlflow.end_run()

    #renvoi des résultat au client
    final_dict = {'Prediction' : prediction, 'Probability' : score, 'Threshold' : best_thresh}
    os.chdir(script_path)
    del data, exp_name, run_name, exp_dict, exp_id, run_dict, run_id, path, model, features, prediction, score, tmp_df, n
    gc.collect()
    return final_dict

'''Interprétaton locale de la prédiction avec utilisation de shap explainer. Le modèle prédit la probabilité d'appartenir à la classe positive
et le kernel explainer met en lummière les features ayant l'impact le plus élevé sur les prédiction.
Les features ayants un impact faible sont supprimés de manière à avoir une meilleure visibilité à la visualisation.'''
@app.post('/production/inference/shap_explainer/no_plot')
def model_server_explainer(input:exp_run_predict):
    os.chdir(script_path)
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    prediction_input = np.array(data['data'])
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    os.chdir('..')
    mlflow.set_experiment(exp_name)
    path = "mlruns/" + exp_id + '/' + run_id + '/artifacts/' + exp_name + '_' + run_name + '_Classifier'
    try:
        model = mlflow.sklearn.load_model(path)
    except:
        model = mlflow.lightgbm.load_model(path)
    path2 = "mlruns/" + exp_id + '/' + run_id + '/artifacts/preprocessed/train.pkl'
    train_data = pd.read_pickle(path2)
    shap.initjs()
    shap_explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(train_data.sample(500), 3))
    shap_values = list(shap_explainer.shap_values(prediction_input)[0])
    final_dict = {'A' : shap_values}
    del data, exp_name, run_name, exp_dict, prediction_input, exp_id, run_dict, run_id, path, model, path2, train_data, shap_explainer, shap_values
    gc.collect()
    return final_dict

'''A cette URI, nous recevons une requête pour effectuer une inférence sur 1000 individus aléatoirement. 
Les prédictions sont renvoyés au client, et nous utilisons shap pour renvoyer une intérpetation moyenne sur l'ensemble des 1000
observations. Pour cela, nous effectuons une interprétation locale individuelle, et nous prennons la moyenne des interpétations par 
features.
Par soucis de visibilité, seuls les features ayant les importances les plus élevées sont conservées.'''
@app.post('/production/inference/sample1000')
def model_server_sample(input:exp_run_data):
    os.chdir(script_path)
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    dataset = data['dataset']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    os.chdir('..')
    mlflow.set_experiment(exp_name)
    path = "mlruns/" + exp_id + '/' + run_id + '/artifacts/' + exp_name + '_' + run_name + '_Classifier'
    try:
        model = mlflow.sklearn.load_model(path)
    except:
        model = mlflow.lightgbm.load_model(path)
    if dataset == 'Production':
        path_data = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/preprocessed/test.pkl'
    elif dataset == 'Test':
        path_data = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/train_test/X_test.pkl'
    else:
        path_data = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/train_test/X_train.pkl'
    df = pd.read_pickle(path_data).reset_index(drop = True)
    sample = df.sample(10)
    index = list(sample.index)
    best_thresh = open("mlruns/" + exp_id + '/' + run_id + '/params/classifier__threshold').read()
    best_thresh = float(best_thresh)
    score_full = list(model.predict_proba(sample))
    score = []
    for i in range(len(score_full)):
        score.append(score_full[i][1])
    prediction = list((np.array(score)>best_thresh).astype(float))
    train_data = pd.read_pickle("mlruns/" + exp_id + '/' + run_id + '/artifacts/preprocessed/train.pkl')
    shap_explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(train_data.sample(300), 3))
    shap_values = list(np.ndarray.mean(np.array(shap_explainer.shap_values(sample))[0], axis = 0))
    for i in os.listdir('predictions_tmp'):
        os.remove('predictions_tmp/' + i)
    tmp_df = sample.copy()
    tmp_df['prediction'] = prediction
    tmp_df['score'] = score
    tmp_df.to_csv('predictions_tmp/prediction.csv', index = False)
    try:
        n = len(os.listdir("mlruns/" + exp_id + '/' + run_id + '/artifacts/predictions'))
    except:
        n = 0
    with mlflow.start_run(run_id=run_id, experiment_id=exp_id):
        mlflow.log_artifacts('predictions_tmp', artifact_path='predictions/' + 'pred_' + str(n))
    mlflow.end_run()
    final_dict = {'Prediction' : prediction, 'Probability' : score, 'Index' : index, 'shap_values' : shap_values}
    del data, exp_name, run_name, exp_dict, dataset, exp_id, run_dict, run_id, path, model, path_data, df, index, sample, prediction
    gc.collect()
    del score, score_full, train_data, shap_explainer, shap_values, tmp_df, n
    gc.collect()
    return final_dict

'''De la même manière que précédemment, nous effectuons une prédiction groupée, mais cette fois sur l'ensemble du jeu de données de 
production.'''
@app.post('/production/inference/full')
def model_server_full(input:exp_run_data):
    os.chdir(script_path)
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    dataset = data['dataset']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    os.chdir('..')
    mlflow.set_experiment(exp_name)
    path = "mlruns/" + exp_id + '/' + run_id + '/artifacts/' + exp_name + '_' + run_name + '_Classifier'
    try:
        model = mlflow.sklearn.load_model(path)
    except:
        model = mlflow.lightgbm.load_model(path)
    if dataset == 'Production':
        path_data = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/preprocessed/test.pkl'
    elif dataset == 'Test':
        path_data = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/train_test/X_test.pkl'
    else:
        path_data = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/train_test/X_train.pkl'
    df = pd.read_pickle(path_data).reset_index(drop = True)
    index = list(df.index)
    best_thresh = open("mlruns/" + exp_id + '/' + run_id + '/params/classifier__threshold').read()
    best_thresh = float(best_thresh)
    score = list(model.predict_proba(df)[:,1])
    prediction = list((np.array(score)>best_thresh).astype(float))
    train_data = pd.read_pickle("mlruns/" + exp_id + '/' + run_id + '/artifacts/preprocessed/train.pkl')
    shap_explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(train_data.sample(300), 3))
    shap_values = list(np.ndarray.mean(np.array(shap_explainer.shap_values(df))[0], axis = 0))
    for i in os.listdir('predictions_tmp'):
        os.remove('predictions_tmp/' + i)
    tmp_df = df.copy()
    tmp_df['prediction'] = prediction
    tmp_df['score'] = score
    tmp_df.to_csv('predictions_tmp/prediction.csv', index = False)
    try:
        n = len(os.listdir("mlruns/" + exp_id + '/' + run_id + '/artifacts/predictions'))
    except:
        n = 0
    with mlflow.start_run(run_id=run_id, experiment_id=exp_id):
        mlflow.log_artifacts('predictions_tmp', artifact_path='predictions/' + 'pred_' + str(n))
    mlflow.end_run()
    final_dict = {'Prediction' : prediction, 'Probability' : score, 'Index' : index, 'shap_values' : shap_values}
    return final_dict

'''Cette fonction permet de retourner l'analyse de datadrift d'une colonne en particulier entre le jeu de données sélectionné
et le jeu de données d'entraînement du modèle sélectionné.'''
@app.post('/production/datadrift/single_column')
def datadrift_single_col(input:exp_run_single_feature):
    os.chdir(script_path)
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    dataset = data['dataset']
    feature = data['features']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    os.chdir('..')
    if dataset == 'Production':
        path_data = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/preprocessed/test.pkl'
    elif dataset == 'Test':
        path_data = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/train_test/X_test.pkl'
    else:
        path_data = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/train_test/X_train.pkl'  
    current_data = pd.DataFrame(pd.read_pickle(path_data)[feature])
    reference = pd.DataFrame(pd.read_pickle('mlruns/' + exp_id + '/' + run_id + '/artifacts/preprocessed/train.pkl')[feature])
    report = Report(metrics=[DataDriftPreset(),])
    report.run(reference_data=reference, current_data=current_data)
    report.save_json('drift_report.json')
    report.save_html('drift_report.html')
    with open('drift_report.json', 'r') as file:
        json_drift = json.loads(file.read())
    with codecs.open('drift_report.html', 'r', encoding='utf8') as html_f:
        html_drift = html_f.read()
    os.remove('drift_report.json')
    os.remove('drift_report.html')
    final_dict = {'HTML' : html_drift, 'JSON' : json_drift}
    return final_dict

@app.post('/production/datadrift/full_column')
def datadrift_all_cols(input:exp_run_data):
    os.chdir(script_path)
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    dataset = data['dataset']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    os.chdir('..')
    if dataset == 'Production':
        path_data = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/preprocessed/test.pkl'
    elif dataset == 'Test':
        path_data = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/train_test/X_test.pkl'
    else:
        path_data = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/train_test/X_train.pkl'  
    current_data = pd.read_pickle(path_data)
    reference = pd.read_pickle('mlruns/' + exp_id + '/' + run_id + '/artifacts/preprocessed/train.pkl')
    report = Report(metrics=[DataDriftPreset(),])
    report.run(reference_data=reference, current_data=current_data)
    report.save_html('drift_report.html')
    with codecs.open('drift_report.html', 'r', encoding='utf8') as html_f:
        html_drift = html_f.read()
    os.remove('drift_report.html')
    final_dict = {'HTML' : html_drift}
    return final_dict

@app.post('/prodction/model_performance')
def performance_tracking(input:exp_run):
    os.chdir(script_path)
    data = input.dict()
    exp_name = data['exp_name']
    run_name = data['run_name']
    exp_dict = experiments_lister()
    exp_id = exp_dict[exp_name]
    os.chdir('..')
    run_dict = run_lister(exp_id)
    run_id = run_dict[run_name]
    os.chdir('..')
    df = pd.DataFrame()
    path = 'mlruns/' + exp_id + '/' + run_id + '/artifacts/predictions'
    for i in os.listdir(path):
        tmp = pd.read_csv(path + '/' + i + '/prediction.csv')
        df = pd.concat([df, tmp], axis = 0)
    inference_data = df[df.columns[:-2]].drop_duplicates(keep = 'first')
    scores = df['score']
    predictions = df['prediction']
    #réalisé sur un échantillon du jeu de données pour accélérer les calculs
    reference = pd.read_pickle('mlruns/' + exp_id + '/' + run_id + '/artifacts/preprocessed/train.pkl').sample(1000)
    report = Report(metrics=[DataDriftPreset(),])
    report.run(reference_data=reference, current_data=inference_data)
    report.save_json('drift_report.json')
    with open('drift_report.json', 'r') as f:
        drift = json.loads(f.read())['metrics'][1]['result']['drift_by_columns']
    os.remove('drift_report.json')
    final_dict = {"column_name" : [], "column_type" : [], "stattest_name" : [], "drift_score" : [], "threshold" : []}
    for i in drift.keys():
        if drift[i]['drift_detected'] == True:
            final_dict["column_name"].append(drift[i]["column_name"])
            final_dict["column_type"].append(drift[i]["column_type"])
            final_dict["stattest_name"].append(drift[i]["stattest_name"])
            final_dict["drift_score"].append(drift[i]["drift_score"])
            final_dict["threshold"].append(drift[i]["threshold"])
    return final_dict

#Définition de l'hôte et du port à l'appel du fichier
if __name__=='__main__':
    port = os.environ['PORT']
    uvicorn.run(app, host = '0.0.0.0', port = port)

#python -m uvicorn app:app --reload
#Ligne de commande pour servir l'application
#uvicorn.run(app, host = '127.0.0.1', port = 8000)
