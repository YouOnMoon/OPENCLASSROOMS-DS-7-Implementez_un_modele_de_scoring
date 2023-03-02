'''Dans ce fichier, nous allons définir plusieurs fonctions permettant de crée nos expériences et nos runs via mlflow.

Dans ce cadre, nous définirons des fonctions permettant de crée une épxerience, entraînent un premier traitement des données
initailes à notre disposition, de manière à créer plusieurs jeux de données, pour l'entraînement, la mise en production, et les targets.

Par la suite au sein de chaque expérience, nous allons mettre en place des runs au sein desquels nous mettons en places des fonctions permattant
d'entraîner des modèles, avec une recherche sur grille des meilleurs hyper-paramètres, la mesure des performances des modèles en utilisat plusieurs métriques, 
et la centralisation des modèles et de différents artéfacts de manière centralisée en utilisant mlflow tracking.'''

#Impotration des libriries utiles

#Librairies de destion système
import gc
import os
import mlflow
import mlflow.sklearn
import json
import time
import sys

sys.path.append('..')
os.chdir('..')
#fonction de feature engineering issue du kernel Kaggle
from src.FeatureEngineering import main

#Librairies de manipulation des données
import numpy as np
import pandas as pd

#Librairies de visualisation des données
import matplotlib.pyplot as plt
import seaborn as sns

#Librairires de pré-processing des données
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from sklearn.compose import make_column_selector
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer

#Librairies d'importation des modèles
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklego.meta import Thresholder #non utilisée ici, mais a servie au développement

#Librairies de mesure de performances (métriques)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import make_scorer #mise en place d'un score métier 
from sklearn.inspection import permutation_importance

#Fonctions définies dans le fichier utils.py 
from src.utils import nan_dropper #suppression features avec des valeurs manquantes à partir d'un certain seuil
from src.utils import Correlation_filter
from src.utils import nan_filler
from src.utils import chi2_dropper
from src.utils import eta_squared_dropper
from src.utils import std_dropper
from src.utils import chi2_selector
from src.utils import eta_squared_selector
os.chdir('src')
#######################################################################################################################################

#Définiton de notre loss function pour séléctionner les meilleurs modèles en terme de coût métier
def fn_fp_scorer(y_true, y_pred):
    '''Dans notre cas, la classe positive correspond au fait qu'un crédit s'est retrouvé en défaut de paiement.
    C'est également la classe minoritaire de notre jeu de données.
    Nous estimerons ici (ce qui est à préciser pour un cas d'application réel), qu'un client en défaut de paiement
    coûte 10 fois plus que ce qu'aurait rapporté un crédit refusé qui ne se serait pas retrouvé en défaut de paiement.
    Nous avons donc une fonction coût qui pénalise 10 fois plus les faux négatifs que les faux positifs.
    Nous pondéront cette loss à la taille de la population pour consérver des valeurs comparables en test, validation et à 
    l'entraînement.'''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    score = -(10*fn + fp)/len(y_true)
    return score

#Définition du score à partir de la loss function
model_score = make_scorer(fn_fp_scorer, greater_is_better=True)

#######################################################################################################################################

'''La fonction suivante a plusieurs objectifs:
    - Créer une nouvelle expérince mlflow
    - Au sein de cette expérince, nous effectuons les préprocessings de base de nos jeu de données dans le répertoire ../inputs à partir du kernel kaggle
     du fichier FeatureEngineering.py de manière à ce que chaque individu de la population corresponde à un crédit.
    - Nous séparons les jeux de données train et target, ainsi que le jeu de données "test" qui correspond à nos données de production.
    - Nous supprimons les variables trop corrélées à la target s'il y en a de manière à éviter le data leakage.
    - Nous supprimer certaines variables en fonction du taux de valeurs manquantes via un seuil.
    - Nous séléctionnons certaines variables trop corrélée entre elles via un seuil.
    - Nous complétons les valeurs manquantes des features quantitatives via une stratégie rédéfinie (mean, median, min, max, none, ...)
    - Nous placons les jeux de données dans un dosseir temporaire pour les logger plus tard.
    - Nous initions un Dummy Classifer avec la stratégie most-frequent de manière à comparer nos futurs modèles à une approche naïve adaptée à un jeu de données
    avec des targets déséquilibrés.
    - Nous définissons plusieurs jeux de données X_train, X_test, y_train et y_test pour entraîenement et test.
    - Nous mesurons plusieurs métriques sur ces jeux de données.
    - Nous créeons une première run mlflow au sein de la nouvelle expérience.
    - Nos loggons les jeux de données, le modèle DummyClassifier et les métriques (ainsi que la description de l'expérience en tag)'''

def import_data(pickle_export, nan_thresh, corr_thresh, nan_strategy, exp_name, exp_desc):
    #Définition de l'URI de tracking mlflow
    mlflow.set_tracking_uri('mlruns')

    #Définition du nom de l'expérience par l'utilisatuuer et création de l'expérience
    EXPERIMENT_NAME = exp_name
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME, tags = {'Description' : exp_desc})
    EXP_DICT = {'ID': str(EXPERIMENT_ID), 'NAME' : EXPERIMENT_NAME}
    os.chdir('src')

    #Appel de la fonction main pour effectuer notre feature engineering à partir du kernel kaggle - debugg = false pour l'intégralité du jeu de données
    df = main(debug = False)

    #Séparation des jeux de données train, de production (test), et les targets
    train_index = df.loc[~df['TARGET'].isna()].index
    train = df.loc[~df['TARGET'].isna()]
    test = df.loc[df['TARGET'].isna()]
    target = pd.DataFrame(train['TARGET'])
    #Suppression des colonnes inutiles
    #Suppression des colonnes inutiles - et redéfinition des jeux de données train et production
    df = df.drop(['TARGET', 'index', 'SK_ID_CURR'], axis = 1)
    train = df.loc[df.index.isin(train_index)]
    test = df.loc[~df.index.isin(train_index)]

    #Filtre des features via les seuils sur les valeurs manqauntes, les coefficients de pearson et complétion des valeurs manquantes
    train = nan_dropper(train, nan_thresh)
    test = test[train.columns]
    train = nan_filler(train, nan_strategy)

    for i in train.columns:
        if nan_strategy == 'median':
            test[i] = test[i].fillna(train[i].median())
        elif nan_strategy == 'mean':
            test[i] = test[i].fillna(train[i].mean())
        elif nan_strategy == 'min':
            test[i] = test[i].fillna(train[i].min())
        elif nan_strategy == 'max':
            test[i] = test[i].fillna(train[i].max())
        else:
            pass

    #Filtre des features via les seuils sur les valeurs manqauntes, les coefficients de pearson et complétion des valeurs manquantes
    train = std_dropper(train, 0.01)
    test = test[train.columns]

    #On place les colonnes qualitatives (one-hot-encodées dans le kernel kaggle - moins de 3 valeurs uniques) en type object
    for i in train.columns:
        if len(train[i].unique()) < 3:
            train[i] = train[i].astype(object)
            test[i] = test[i].astype(object)

    #Suppression des colonnes catégoriques trop corrélées à la target
    train = chi2_dropper(train, target, 0.5)
    test = test[train.columns]

    #Nous réalisons la même opération avec les variables quantitatives, en se servant de la métrique eta_squared
    train = eta_squared_dropper(train, target, 0.5)
    test = test[train.columns]

    #Masks sur les différents types de colonnes
    cat_cols = train.dtypes.loc[train.dtypes == 'object'].index
    num_cols = train.dtypes.loc[train.dtypes != 'object'].index

    #Projectionsur les colonnes catégoriques
    train_cat = train[cat_cols]
    train_num = train[num_cols]

    #Sélection des features pour limiter les corrélations trop élevées
    train_num_cleaned = Correlation_filter(train_num, corr_thresh)
    #Suppession des colonnes droppés de train et test
    inti_cols = train_num.columns
    end_cols = train_num_cleaned.columns
    diff = inti_cols.drop(end_cols)
    train = train.drop(diff, axis = 1)
    test = test[train.columns]

    #Sélection des features qualitatifs pour le critère du chi2
    train = chi2_selector(train, 0.9)

    #Suppression des features qualitatifs trop corrélés aux features quantitatifs
    #train = eta_squared_selector(train, corr_thresh) - non réalisé ici car très long sans résultats intéréssant
    test = test[train.columns]

    os.chdir('..')

    #export dans un répertoire temporaire des jeux de donnés au formt .pkl en attendant d'être loggués en artefact et des exp_id au format .json
    if pickle_export == True:
        os.chdir('src/preprocessed')
        for i in os.listdir():
            os.remove(i)
        os.chdir('..')
        train.to_pickle('preprocessed/train.pkl')
        test.to_pickle('preprocessed/test.pkl')
        target.to_pickle('preprocessed/target.pkl')
        with open('mlflow_exp/exp_id.json', 'w') as fp:
            json.dump(EXP_DICT, fp)
        os.chdir('..')

    #on se place dans la nouvelle expérience mlflow et séparation des données en train et test - startification sur la target - test size à 0.2
    mlflow.set_experiment(exp_name)
    Rand_State = 11
    Test_Size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(train, target, stratify = target, random_state = Rand_State, test_size = Test_Size)

    #export dans un répertoire temporaire des jeux de donnés d'entraînement det de test en attendant log mlflow
    for i in os.listdir('src/train_test'):
        os.remove('src/train_test/'+i)
    X_train.to_pickle('src/train_test/X_train.pkl')
    X_test.to_pickle('src/train_test/X_test.pkl')
    y_train.to_pickle('src/train_test/y_train.pkl')
    y_test.to_pickle('src/train_test/y_test.pkl')

    #Mise en place Dummy Classifier - strategy most frequent pour coller à un jeu de données à targets déséquilibrées (Bonnes accuracy, mauvais recall)
    Dummy = DummyClassifier(strategy='most_frequent')
    train_time_start = time.time()
    Dummy.fit(X_train, y_train)
    train_time = time.time() - train_time_start
    inference_time_start = train_time_start = time.time()
    Dummy.predict(X_test.iloc[0].values.reshape(1,-1))
    inference_time = time.time() - inference_time_start
    y_train_pred = Dummy.predict(X_train)
    y_test_pred = Dummy.predict(X_test)

    #Mesure des métriqus classiques sur jeux de données train et test
    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_rec = recall_score(y_train, y_train_pred)
    test_rec = recall_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    train_balanced_acc = balanced_accuracy_score(y_train, y_train_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)

    #Mesure du score défini
    train_score = fn_fp_scorer(y_train, y_train_pred)
    test_score = fn_fp_scorer(y_test, y_test_pred)

    '''Les confusion matrix et les rocs curves sont enregistrés dans un repértoire tempon en attendant d'étre logguées via mlflow'''
    #Création confusion matrix test et export dans le repértoire metrics figure
    for i in os.listdir('src/metrics_figures'):
        os.remove('src/metrics_figures/'+i)
    test_cm = confusion_matrix(y_test, y_test_pred, labels=Dummy.classes_)
    test_cm_df = pd.DataFrame(test_cm, index = Dummy.classes_, columns = Dummy.classes_)
    fig_conf_test = sns.heatmap(test_cm_df, annot=True, cmap = 'Blues')
    plt.title('Test confusion matrix', size = 20, weight = 'bold')
    plt.xlabel('predictions', size = 12, weight = 'bold')
    plt.ylabel('truth', size = 12, weight = 'bold')
    fig_conf_image_test = fig_conf_test.figure.savefig('src/metrics_figures/test_conf_matrix.png')
    del test_cm, test_cm_df, fig_conf_test
    gc.collect()

    #Création confusion matrix train
    train_cm = confusion_matrix(y_train, y_train_pred, labels=Dummy.classes_)
    train_cm_df = pd.DataFrame(train_cm, index = Dummy.classes_, columns = Dummy.classes_)
    fig_conf_train = sns.heatmap(train_cm_df, annot=True, cmap = 'Blues')
    plt.title('Train confusion matrix', size = 20, weight = 'bold')
    plt.xlabel('predictions', size = 12, weight = 'bold')
    plt.ylabel('truth', size = 12, weight = 'bold')
    fig_conf_image_train = fig_conf_train.figure.savefig('src/metrics_figures/train_conf_matrix.png')
    del train_cm, train_cm_df, fig_conf_train
    gc.collect()

    #roc_curve jeu de données train
    train_roc_curve = RocCurveDisplay.from_estimator(Dummy, X_train, y_train)
    train_roc_curve_plot = train_roc_curve.plot().figure_
    plt.title('Train ROC_CURVE', size = 20, weight = 'bold')
    train_roc_image = train_roc_curve_plot.savefig('src/metrics_figures/train_roc_curve.png')

    #roc_curve jeu de données test
    test_roc_curve = RocCurveDisplay.from_estimator(Dummy, X_test, y_test)
    test_roc_curve_plot = test_roc_curve.plot().figure_
    plt.title('Test ROC_CURVE', size = 20, weight = 'bold')
    test_roc_image = test_roc_curve_plot.savefig('src/metrics_figures/test_roc_curve.png')

    #Création de la run mlflow et nom de la run 'DummyClassifier'
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name= 'DummyClassifier', nested=False) as run:
        #Mise en tag du random state du train_test_split et du test size
        mlflow.set_tag('Train_Test_Split_Random_State', str(Rand_State))
        mlflow.set_tag('Test_Size', str(Test_Size))

        #log des jeux de données et des plots roc_curve et confusion_matrix
        mlflow.log_artifacts("src/preprocessed", artifact_path = 'preprocessed')
        mlflow.log_artifacts("src/train_test", artifact_path = 'train_test')
        mlflow.log_artifacts("src/metrics_figures", artifact_path = 'metrics_figures')

        #log des métriques
        mlflow.log_metric('Inference_time', inference_time)
        mlflow.log_metric('Training_time', train_time)
        mlflow.log_metric('Train_AUC', train_auc)
        mlflow.log_metric('Test_AUC', test_auc)
        mlflow.log_metric('Train_ACC', train_acc)
        mlflow.log_metric('Test_ACC', test_acc)
        mlflow.log_metric('Train_REC', train_rec)
        mlflow.log_metric('Test_REC', test_rec)
        mlflow.log_metric('Train_F1', train_f1)
        mlflow.log_metric('Test_F1', test_f1)
        mlflow.log_metric('Train_balanced_ACC', train_balanced_acc)
        mlflow.log_metric('Test_balanced_ACC', test_balanced_acc)
        mlflow.log_metric('Test_custom_score', test_score)
        mlflow.log_metric('Train_custom_score', train_score)

        #log du modèle et récupération de l'id de la run
        mlflow.sklearn.log_model(Dummy, str(EXPERIMENT_NAME) + '_DummyClassifier')
        run_id = run.info.run_id

    #Désallocation des variables placées en mémoire à l'utilisation de la fonction
    del EXPERIMENT_NAME, EXP_DICT, train_index, df, pickle_export
    gc.collect()
    del Rand_State, Test_Size, X_train, X_test, y_train, y_test, Dummy, y_train_pred, y_test_pred, train_auc, test_auc, train_acc
    gc.collect()
    del test_acc, train_rec, test_rec, train_f1, test_f1, train_balanced_acc, test_balanced_acc, train_score, test_score, train_roc_curve
    gc.collect()
    del train_roc_curve_plot, train_roc_image, test_roc_curve, test_roc_curve_plot, test_roc_image
    gc.collect()

    #Retourne les jeux de données train, test, target et l'ID de l'éxperience crée
    return train, test, target, EXPERIMENT_ID


#######################################################################################################################################


'''La fonction suivante permet de charger les données placées temporairement dans le  répertoire src/preprocessed correpsondant 
aux jeux de données auquals on apliqué notre feature engineering'''
def pkl_import():
    train = pd.read_pickle('src/preprocessed/train.pkl')
    test = pd.read_pickle('src/preprocessed/test.pkl')
    target = pd.read_pickle('src/preprocessed/target.pkl')
    exp_id = json.load(open('src/mlflow_exp/exp_id.json'))['ID']
    exp_name = json.load(open('src/mlflow_exp/exp_id.json'))['NAME']
    return train, test, target, exp_id, exp_name


#######################################################################################################################################


'''La fonction suivante permet de créer de nouvelles runs mlflow au sein d'une expérience données, de manière à pouvoir entraîner de noueaux modèles via une optimisation des hyper-paramètres
par recherche sur grille par validation croisée. La gestion du désequilibre des target se fait via SMOTE, création synthétique de nouvelles observations de la classe minoritaire en fonction
du nombre de plus proches voisins.
Cela se passe en plusieurs étapes:
- Split du jeu de données initila en test et train (test size à 0.2, stratification sur la target)
- Initialisation de SMOTE ENCODER pour la prise en compte des variables qualitatives, nombreuses dans notre jeu de données
- Innitialisation du standard Scaler, obligatoire avec SMOTE dans le cas de la création synthetic de nouvelles variables, pour les features quantitatives.
- Nous avons l'option de créer une PCA et conserver 95% de la variance expliquée de notre jeu de données. Cette option perd néanmoins sont intérêt ici car nous séléectionnons les features en
fonction du seuil de coefficient de pearson, et de nombreuses colonnes sont très sparses, ce qui pousse à utiliser un pré-processeur composite avec PCA sur les features quantitiatives, et
truncatedSVD sur les features qualitatives.
- Création du pipeline avec un classifieur au choix.
- Définition des hyper-paramètres du classifieur choisis.
- Import des hyper-paramètres sur lesquels effectuer la recherche sur grille.
- Recherche sur grille par validation croisée, en incluant une liste de k_neighbors pour smote.
- Entraînement d'un modèle sur le jeu de données X_train avec les meilleurs hyper-paramètres.
- Mesure des performances du modeèle, enregistrement des résultats dans des répertoires temporaires en attendant log mlflow.
- Création de la nouvelle run mlflow et enregistrement des métriques, du modèle, des jeux de données, et des plots.'''

def smote_hyperparam_tuning(train, target, json_export, scaler, pca_state, params, grid_params, model_type, new_run_name, experiment_name, experiment_id):
#Check type des colonnes après importation
    for i in train.columns:
        if len(train[i].unique())<3:
            train[i] = train[i].astype(object)
        else:
            train[i] = train[i].astype(float)

#Masks colonnes catégoriques et numériques
    num_mask = (train.dtypes != object).values
    cat_mask = (train.dtypes == object).values

#Split des jeux de données de train et test
    Rand_State = 11
    Test_Size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(train, target, stratify = target, random_state = Rand_State, test_size = Test_Size)

#Initialisation de SMOTENCODER, avec mask sur les features qualitatifs, du standard scaler, et de la pca (non obligatoire)
    over_sampling_ratio = 2*target['TARGET'].value_counts().min()/len(target)
    smote = SMOTENC(random_state=11, categorical_features=cat_mask, sampling_strategy=over_sampling_ratio)
    std = make_column_transformer((StandardScaler(), num_mask), (MinMaxScaler(), cat_mask))
    pca = make_column_transformer((PCA(), num_mask), (TruncatedSVD(), cat_mask))

#Choix du classifieur par l'utilisateur et défiintion des hyper-paramètres séléctionnés par l'utilisateur
    if model_type == 'GradientBooster':
        if len(params.keys()) == 0:
            classifier = Thresholder(GradientBoostingClassifier(), threshold= 0.5)
        else:
            classifier = Thresholder(GradientBoostingClassifier(**params), threshold= 0.5)
    elif model_type == 'HistGradientBooster':
        if len(params.keys()) == 0:
            classifier = Thresholder(HistGradientBoostingClassifier(), threshold= 0.5)
        else:
            classifier = Thresholder(HistGradientBoostingClassifier(**params), threshold= 0.5)
    elif model_type == 'LightGBM':
        if len(params.keys()) == 0:
            classifier = Thresholder(LGBMClassifier(), threshold= 0.5)
        else:
            classifier = Thresholder(LGBMClassifier(**params), threshold= 0.5)
    else:
        if len(params.keys()) == 0:
            classifier = Thresholder(LogisticRegression(), threshold= 0.5)
        else:
            classifier = Thresholder(LogisticRegression(**params), threshold= 0.5)

#Choix du pipeline
    if (scaler == False) and (pca_state == False):
        pipe = Pipeline(steps = [('smote', smote), ('classifier', classifier)])
        X_train = X_train.values
    elif (scaler == True) and (pca_state == False):
        pipe = Pipeline(steps = [('std', std), ('smote', smote), ('classifier', classifier)])
    elif (scaler == False) and (pca_state == True):
        pipe = Pipeline(steps = [('smote', smote), ('pca', pca), ('classifier', classifier)])
    else:
        pipe = Pipeline(steps = [('std', std), ('smote', smote), ('pca', pca), ('classifier', classifier)])

#Changement du type des hyper-paramètres de type intéger (géré de la sorte car besoin de passer les paramètre en float pour pasage du client au serveur)
    for i in grid_params.keys():
        if i in classifier.get_params().keys():
            if type(classifier.get_params()[i]) == int:
                grid_params[i] = np.array(grid_params[i]).astype(int)

#Définition de la grille pour l'optimisation eds hyper-paramètres, et grille des plus proches voisins SMOTE
    param_keys= grid_params.copy().keys()
    for i in param_keys:
        grid_params['classifier__model__'+i] = grid_params[i]

    for i in param_keys:
        del grid_params[i]
    grid_params['smote__k_neighbors'] = np.linspace(3, 9, 4).astype(int)
    grid_params['classifier__threshold'] = np.linspace(0.1, 0.5, 5)

#Définition des métriques, et recherche sur grille avec refit sur le score défini plus haut (loss function basée sur les faux négatifs et faux positifs) 
    score = {'Self_Score' : model_score, 'F1' : 'f1', 'Recall' : 'recall', 'Accuracy' : 'accuracy', 'ROC_AUC' : 'roc_auc', 'Balanced_Accuracy' : 'balanced_accuracy'}
    grid = GridSearchCV(pipe, grid_params, cv = 5, n_jobs= 1, verbose = 1, scoring = score, refit = 'Self_Score')
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    results = pd.DataFrame(grid.cv_results_)

#Enregistrement des résultats de la recherche sur grille en attendant log mlflow, et des meilleurs hyper-paramètres
    if os.path.exists('../src/cv_results/results.csv'):
        os.remove('../src/cv_results/results.csv')
    else:
        pass
    results.to_csv('../src/cv_results/results.csv', index = False)
    param_df = pd.DataFrame(np.array(list(best_params.values())), index = list(best_params.keys())).transpose()
    params_float = {}

#Convertion des dtypes en float pour export au format .json - communication client-serveur
    for i in best_params.keys():
        try:
            params_float[i] = float(best_params[i])
        except:
            params_float[i] = best_params[i]
    if json_export == True:
        os.chdir('../src/hyper_params')
        for i in os.listdir():
            os.remove(i)
        os.chdir('../..')
        with open('src/hyper_params/hyperparams.json', 'w') as fp:
            json.dump(params_float, fp)
#Entraînement nouveau modèle - sans thresholder
    if model_type == 'GradientBooster':
        if len(params.keys()) == 0:
            classifier = GradientBoostingClassifier()
        else:
            classifier = GradientBoostingClassifier(**params)
    elif model_type == 'HistGradientBooster':
        if len(params.keys()) == 0:
            classifier = HistGradientBoostingClassifier()
        else:
            classifier = HistGradientBoostingClassifier(**params)
    elif model_type == 'LightGBM':
        if len(params.keys()) == 0:
            classifier = LGBMClassifier()
        else:
            classifier = LGBMClassifier(**params)
    else:
        if len(params.keys()) == 0:
            classifier = LogisticRegression()
        else:
            classifier = LogisticRegression(**params)

    #Mise en place du pipeline pour couvir les cas d'usages avec le classifier
    if (scaler == False) and (pca_state == False):
        pipe = Pipeline(steps = [('smote', smote), ('model', classifier)])
    elif (scaler == True) and (pca_state == False):
        pipe = Pipeline(steps = [('std', std), ('smote', smote), ('model', classifier)])
    elif (scaler == False) and (pca_state == True):
        pipe = Pipeline(steps = [('pca', pca), ('smote', smote), ('model', classifier)])
    else:
        pipe = Pipeline(steps = [('std', std), ('smote', smote), ('pca', pca), ('model', classifier)])

    #Définition des paramètres du model
    new_params = {}
    for i in best_params.keys():
        if 'classifier' in i:
            new_params[i[12:]] = best_params[i]
        elif 'threshold' in i:
            best_threshold = best_params[i]
        else:
            new_params[i] = best_params[i]
    best_threshold = new_params['threshold']
    del new_params['threshold']

    #Mise en place des nouveaux hyperparamètre
    pipe.set_params(**new_params)

    #Entraînement nouveau modèle et mesure des temps d'entraînement et d'inférence
    train_time_start = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - train_time_start

    inference_time_start = train_time_start = time.time()
    pipe.predict(X_test.iloc[0].values.reshape(1,-1))
    inference_time = time.time() - inference_time_start

    #Mesure des métriques classiques et de notre score custom
    train_proba_predict = pipe.predict_proba(X_train)[:, 1]
    try:
        test_proba_predict = pipe.predict_proba(X_test)[:,1]
    except:
        test_proba_predict = pipe.predict_proba(X_test.values)[:,1]
    y_train_pred = (train_proba_predict > best_threshold).astype(int)
    y_test_pred = (test_proba_predict > best_threshold).astype(int)

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_rec = recall_score(y_train, y_train_pred)
    test_rec = recall_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    train_balanced_acc = balanced_accuracy_score(y_train, y_train_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    train_score = fn_fp_scorer(y_train, y_train_pred)
    test_score = fn_fp_scorer(y_test, y_test_pred)

#Enegistremet des jeux de donénes train et test pour log mlflow
    for i in os.listdir('src/train_test'):
        os.remove('src/train_test/'+i)
    try:
        X_train.to_pickle('src/train_test/X_train.pkl')
    except:
        pd.DataFrame(X_train, columns = X_test.columns).to_pickle('src/train_test/X_train.pkl')
    X_test.to_pickle('src/train_test/X_test.pkl')
    y_train.to_pickle('src/train_test/y_train.pkl')
    y_test.to_pickle('src/train_test/y_test.pkl')

#Enregistrement des confusion matrix et des roc_curves pour log mlflow
    for i in os.listdir('src/metrics_figures'):
        os.remove('src/metrics_figures/'+i)

    #confusion matrix
    test_cm = confusion_matrix(y_test, y_test_pred, labels=np.unique(target['TARGET'].values))
    train_cm = confusion_matrix(y_train, y_train_pred, labels=np.unique(target['TARGET'].values))
    test_cm_df = pd.DataFrame(test_cm, index = np.unique(target['TARGET'].values), columns = np.unique(target['TARGET'].values))
    fig_conf_test = sns.heatmap(test_cm_df, annot=True, cmap = 'Blues')
    plt.title('Test confusion matrix', size = 20, weight = 'bold')
    plt.xlabel('predictions', size = 12, weight = 'bold')
    plt.ylabel('truth', size = 12, weight = 'bold')
    fig_conf_image_test = fig_conf_test.figure.savefig('src/metrics_figures/test_conf_matrix.png')
    train_cm_df = pd.DataFrame(train_cm, index = np.unique(target['TARGET'].values), columns = np.unique(target['TARGET'].values))
    fig_conf_train = sns.heatmap(train_cm_df, annot=True, cmap = 'Blues')
    plt.title('Train confusion matrix', size = 20, weight = 'bold')
    plt.xlabel('predictions', size = 12, weight = 'bold')
    plt.ylabel('truth', size = 12, weight = 'bold')
    fig_conf_image_train = fig_conf_train.figure.savefig('src/metrics_figures/train_conf_matrix.png')

    #roc_curve
    try:
        train_roc_curve = RocCurveDisplay.from_estimator(pipe, X_train, y_train)
    except:
        train_roc_curve = RocCurveDisplay.from_estimator(pipe, X_train.values, y_train)
    train_roc_curve_plot = train_roc_curve.plot().figure_
    plt.title('Train ROC_CURVE', size = 20, weight = 'bold')
    train_roc_image = train_roc_curve_plot.savefig('src/metrics_figures/train_roc_curve.png')

    try:
        test_roc_curve = RocCurveDisplay.from_estimator(pipe, X_test, y_test)
    except:
        test_roc_curve = RocCurveDisplay.from_estimator(pipe, X_test.values, y_test)
    test_roc_curve_plot = test_roc_curve.plot().figure_
    plt.title('Test ROC_CURVE', size = 20, weight = 'bold')
    test_roc_image = test_roc_curve_plot.savefig('src/metrics_figures/test_roc_curve.png')

#Calcul et plot des features importances en fonction des types de modèle -- coefficients de la régression logistique -- fetaure importance pour LGMClasifier
#Pas de feature importnace disponible avec le HistGradientBoostingClassifier -- utilisation de la méthode permutation importance de scikit-learn
    if model_type == 'HistGradientBooster':
        feature_importance = permutation_importance(pipe, X_train, y_train, n_repeats=5, random_state=11, n_jobs=1)['importances_mean']
    else:
        try:
            feature_importance = pipe['model'].coef_[0]
        except:
            feature_importance = pipe['model'].feature_importances_
    features = train.columns
    feature_importance = list(feature_importance)
    while len(feature_importance) < len(features):
        feature_importance.append(0)
    importance = pd.Series(feature_importance, index=features).sort_values(ascending=True)
    importance_df = pd.DataFrame({'features' : features, 'importance' : feature_importance, 'abs_importance' : np.abs(feature_importance)})

    #On se place sur les 15 features avec l'importance la plus forte
    importance_df_loc = importance_df.sort_values(by = 'abs_importance', ascending = True).iloc[-15:]
    importance_df_loc = importance_df_loc.sort_values(by = 'importance', ascending = True)

    #Affichage des features importances les plus significatives
    importance_fig = plt.figure(figsize = (15,15))
    plt.rcParams.update({'font.size' : 12})
    plt.title('Features importances - Interprétation globale du modèle', size = 30, weight = 'bold')
    plt.barh(importance_df_loc['features'].values, importance_df_loc['importance'].values)
    plt.grid(True, ls = '--')
    plt.xlabel('Feature Importance', size = 20, weight = 'bold')
    rot = plt.yticks(rotation = 15)
    for i in os.listdir('src/feature_importance'):
        os.remove('src/feature_importance/' + i)
    importance_fig.savefig('src/feature_importance/feature_importances.png', bbox_inches='tight')
    '''
    #Enregistrement de la permutation importance pour tout type de modèle
    train_perumtation_importance = pd.DataFrame(permutation_importance(pipe, X_train, y_train, n_repeats=5, random_state=11, n_jobs=1)['importances'])
    test_perumtation_importance = pd.DataFrame(permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=11, n_jobs=1)['importances'])
    for i in os.listdir('src/permutation_importance'):
        os.remove('src/permutation_importance/' + i)
    train_perumtation_importance.to_csv('src/permutation_importance/train_importance.csv', index = False)
    test_perumtation_importance.to_csv('src/permutation_importance/test_importance.csv', index = False)'''

    #Initialisation nouvelle run mlflow au sein de l'expérience en cours
    with mlflow.start_run(experiment_id=experiment_id, run_name = new_run_name, nested=False) as run:
        #log des tags
        mlflow.set_tag('Train_Test_Split_Random_State', str(Rand_State))
        mlflow.set_tag('Test_Size', str(Test_Size))

        #log des jeux de données initiales
        mlflow.log_artifacts("src/preprocessed", artifact_path = 'preprocessed')
        run_id = run.info.run_id

        #log des meilleurs hyper-paramètres de la recherhce sur grille
        for i in best_params.keys():
            mlflow.log_param(i, best_params[i])
        
        #log des métriques
        mlflow.log_metric('Val_score', grid.best_score_)
        mlflow.log_metric('Train_AUC', train_auc)
        mlflow.log_metric('Test_AUC', test_auc)
        mlflow.log_metric('Train_ACC', train_acc)
        mlflow.log_metric('Test_ACC', test_acc)
        mlflow.log_metric('Train_REC', train_rec)
        mlflow.log_metric('Test_REC', test_rec)
        mlflow.log_metric('Inference_time', inference_time)
        mlflow.log_metric('Training_time', train_time)
        mlflow.log_metric('Train_F1', train_f1)
        mlflow.log_metric('Test_F1', test_f1)
        mlflow.log_metric('Train_balanced_ACC', train_balanced_acc)
        mlflow.log_metric('Test_balanced_ACC', test_balanced_acc)
        mlflow.log_metric('Test_custom_score', test_score)
        mlflow.log_metric('Train_custom_score', train_score)

        #log des résultats de la recherche sur grille, des hyper-paramètres, des jeux de données X_train, X_test, y_train, y_test
        mlflow.log_artifacts('src/cv_results', artifact_path = "cv_results")
        mlflow.log_artifacts('src/hyper_params', artifact_path = "hyper_params")
        mlflow.log_artifacts("src/train_test", artifact_path = 'train_test')

        #log des plots des confusions matrix, des feature importance, des roc_curves et des permutation importances
        mlflow.log_artifacts("src/metrics_figures", artifact_path = 'metrics_figures')
        #mlflow.log_artifacts("src/permutation_importance", artifact_path = 'permutation_importance')
        mlflow.log_artifacts("src/feature_importance", artifact_path = 'feature_importance')

        #log du model
        if model_type == 'LightGBM':
            mlflow.lightgbm.log_model(pipe, experiment_name + '_' + new_run_name + '_' + 'Classifier')
        else:
            mlflow.sklearn.log_model(pipe, experiment_name + '_' + new_run_name + '_' + 'Classifier')

    #Désallocation des variables mises en mémoire
    del train, num_mask, cat_mask, Rand_State, Test_Size, X_train, X_test, y_train, y_test, smote, std, pca, model_type, classifier, scaler, pca_state, pipe
    gc.collect()
    del grid_params, param_keys, score, grid, best_params, results, param_df, json_export, y_train_pred, y_test_pred, train_auc, test_auc, train_acc, test_acc
    gc.collect()
    del train_rec, test_rec, train_f1, test_f1, train_balanced_acc, test_balanced_acc, train_score, test_score, test_cm, train_cm, test_cm_df, fig_conf_test, fig_conf_image_test
    gc.collect()
    del train_cm_df, fig_conf_train, fig_conf_image_train, train_roc_curve, train_roc_curve_plot, train_roc_image, test_roc_curve, test_roc_curve_plot, test_roc_image, feature_importance
    gc.collect()
    return params_float, new_run_name


#######################################################################################################################################


'''La fonction suivante conserve les mêmes fonctionnalitées que la précédente, à l'exception de la stratégie de gestion du 
déséquilibre de la target. Cette fois ci, nous n'utiliserons pas d'over-sampling, mais nous allons gérer les class-weights de manière
équilibrée, via l'hyper-paramètre class-weight de nos classifeurs. Cela permettra de pénaliser plus fortement les mauvaises prédictions
sur la classe sous représentée, tout en conservant un seuil de décision par la méthode predict_proba à 0.5.'''

def hyperparam_tuning(train, target, json_export, scaler, pca_state, params, grid_params, model_type, new_run_name, experiment_name, experiment_id):
    #Check type des colonnes après importation
    for i in train.columns:
        if len(train[i].unique())<3:
            train[i] = train[i].astype(object)
        else:
            train[i] = train[i].astype(float)
    #Masks colonnes catégoriques et numériques
    num_mask = (train.dtypes != object).values
    cat_mask = (train.dtypes == object).values

    #Split jeu de données train et test
    Rand_State = 11
    Test_Size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(train, target, stratify = target, random_state = Rand_State, test_size = Test_Size)

    #Preprocesseurs composites au choix
    std = make_column_transformer((StandardScaler(), num_mask), (MinMaxScaler(), cat_mask))
    pca = make_column_transformer((PCA(), num_mask), (TruncatedSVD(), cat_mask))

    #Choix du type d'algorithme de classification - pas de class_weight pour le gradient booster - automatiquement remplacé par histgradientbosster
    if model_type == 'GradientBooster':
        if len(params.keys()) == 0:
            classifier = Thresholder(HistGradientBoostingClassifier(), threshold=0.5)
        else:
            classifier = Thresholder(HistGradientBoostingClassifier(**params), threshold=0.5)
    elif model_type == 'HistGradientBooster':
        if len(params.keys()) == 0:
            classifier = Thresholder(HistGradientBoostingClassifier(), threshold=0.5)
        else:
            classifier = Thresholder(HistGradientBoostingClassifier(**params), threshold=0.5)
    elif model_type == 'LightGBM':
        if len(params.keys()) == 0:
            classifier = Thresholder(LGBMClassifier(), threshold=0.5)
        else:
            classifier = Thresholder(LGBMClassifier(**params), threshold=0.5)
    else:
        if len(params.keys()) == 0:
            classifier = Thresholder(LogisticRegression(), threshold=0.5)
        else:
            classifier = Thresholder(LogisticRegression(**params), threshold=0.5)
    
    #Mise en place du pipeline pour couvir les cas d'usages
    if (scaler == False) and (pca_state == False):
        pipe = Pipeline(steps = [('classifier', classifier)])
        X_train = X_train.values
    elif (scaler == True) and (pca_state == False):
        pipe = Pipeline(steps = [('std', std), ('classifier', classifier)])
    elif (scaler == False) and (pca_state == True):
        pipe = Pipeline(steps = [('pca', pca), ('classifier', classifier)])
    else:
        pipe = Pipeline(steps = [('std', std), ('pca', pca), ('classifier', classifier)])

    #Gestion des types des hyperparamètres
    for i in grid_params.keys():
        if i in classifier.get_params().keys():
            if type(classifier.get_params()[i]) == int:
                grid_params[i] = np.array(grid_params[i]).astype(int)

    #Modification des valeurs des clés des grilles d'hyperparamètres
    param_keys= grid_params.copy().keys()
    for i in param_keys:
        grid_params['classifier__model__'+i] = grid_params[i]
    for i in param_keys:
        del grid_params[i]
    
    #Gestion du déséquilibre des classes par pénalisation de la classe minoritaire accrue
    pipe.set_params(**{'classifier__model__class_weight' : 'balanced'})

    #Gestion des seuils de prédiction par recherche sur grille - meilleur threshol autour de 0.5 dans le cas 'balanced' class weight
    grid_params['classifier__threshold'] = np.linspace(0.2, 0.6, 5)
    
    #Recherche sur grille par validation croisée
    score = {'Self_Score' : model_score, 'F1' : 'f1', 'Recall' : 'recall', 'Accuracy' : 'accuracy', 'ROC_AUC' : 'roc_auc', 'Balanced_Accuracy' : 'balanced_accuracy'}
    grid = GridSearchCV(pipe, grid_params, cv = 5, n_jobs= 1, verbose = 1, scoring = score, refit = 'Self_Score')
    grid.fit(X_train, y_train)

    #Enregistrement des meilleurs hyperparamètres et des résultats de la recherche sur grille
    best_params = grid.best_params_
    results = pd.DataFrame(grid.cv_results_)
    if os.path.exists('../src/cv_results/results.csv'):
        os.remove('../src/cv_results/results.csv')
    else:
        pass
    results.to_csv('../src/cv_results/results.csv', index = False)
    param_df = pd.DataFrame(np.array(list(best_params.values())), index = list(best_params.keys())).transpose()
    params_float = {}
    #convertion des dtypes en float pour export au format .json
    for i in best_params.keys():
        if type(best_params[i]) == int:
            params_float[i] = float(best_params[i])
        else:
            params_float[i] = best_params[i]
    if json_export == True:
        os.chdir('../src/hyper_params')
        for i in os.listdir():
            os.remove(i)
        os.chdir('../..')
        with open('src/hyper_params/hyperparams.json', 'w') as fp:
            json.dump(params_float, fp)
    
    #Entraînement nouveau modèle - sans thresholder
    if model_type == 'GradientBooster':
        if len(params.keys()) == 0:
            classifier = HistGradientBoostingClassifier()
        else:
            classifier = HistGradientBoostingClassifier(**params)
    elif model_type == 'HistGradientBooster':
        if len(params.keys()) == 0:
            classifier = HistGradientBoostingClassifier()
        else:
            classifier = HistGradientBoostingClassifier(**params)
    elif model_type == 'LightGBM':
        if len(params.keys()) == 0:
            classifier = LGBMClassifier()
        else:
            classifier = LGBMClassifier(**params)
    else:
        if len(params.keys()) == 0:
            classifier = LogisticRegression()
        else:
            classifier = LogisticRegression(**params)

    #Mise en place du pipeline pour couvir les cas d'usages avec le classifier
    if (scaler == False) and (pca_state == False):
        pipe = Pipeline(steps = [('model', classifier)])
    elif (scaler == True) and (pca_state == False):
        pipe = Pipeline(steps = [('std', std), ('model', classifier)])
    elif (scaler == False) and (pca_state == True):
        pipe = Pipeline(steps = [('pca', pca), ('model', classifier)])
    else:
        pipe = Pipeline(steps = [('std', std), ('pca', pca), ('model', classifier)])

    #Définition des paramètres du model
    new_params = {}
    for i in best_params.keys():
        if 'classifier' in i:
            new_params[i[12:]] = best_params[i]
        elif 'threshold' in i:
            best_threshold = best_params[i]
        else:
            pass
    best_threshold = new_params['threshold']
    del new_params['threshold']

    #Mise en place des nouveaux hyperparamètre
    pipe.set_params(**new_params)
    pipe.set_params(**{'model__class_weight' : 'balanced'})

    #Entraînement nouveau modèle et mesure des temps d'entraînement et d'inférence
    train_time_start = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - train_time_start

    inference_time_start = train_time_start = time.time()
    pipe.predict(X_test.iloc[0].values.reshape(1,-1))
    inference_time = time.time() - inference_time_start

    #Mesure des métriques classiques et de notre score custom
    train_proba_predict = pipe.predict_proba(X_train)[:, 1]
    try:
        test_proba_predict = pipe.predict_proba(X_test)[:,1]
    except:
        test_proba_predict = pipe.predict_proba(X_test.values)[:,1]
    y_train_pred = (train_proba_predict > best_threshold).astype(int)
    y_test_pred = (test_proba_predict > best_threshold).astype(int)

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_rec = recall_score(y_train, y_train_pred)
    test_rec = recall_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    train_balanced_acc = balanced_accuracy_score(y_train, y_train_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    train_score = fn_fp_scorer(y_train, y_train_pred)
    test_score = fn_fp_scorer(y_test, y_test_pred)

    #Enregistrement des ficher train et test pour centralisation via MLFlow
    for i in os.listdir('src/train_test'):
        os.remove('src/train_test/'+i)
    try:
        X_train.to_pickle('src/train_test/X_train.pkl')
    except:
        pd.DataFrame(X_train, columns = X_test.columns).to_pickle('src/train_test/X_train.pkl')
    X_test.to_pickle('src/train_test/X_test.pkl')
    y_train.to_pickle('src/train_test/y_train.pkl')
    y_test.to_pickle('src/train_test/y_test.pkl')

    #Création des confusion matrix et roc_curve
    for i in os.listdir('src/metrics_figures'):
        os.remove('src/metrics_figures/'+i)

    #confusion matrix
    test_cm = confusion_matrix(y_test, y_test_pred, labels=np.unique(target['TARGET'].values))
    train_cm = confusion_matrix(y_train, y_train_pred, labels=np.unique(target['TARGET'].values))
    test_cm_df = pd.DataFrame(test_cm, index = np.unique(target['TARGET'].values), columns = np.unique(target['TARGET'].values))
    fig_conf_test = sns.heatmap(test_cm_df, annot=True, cmap = 'Blues')
    plt.title('Test confusion matrix', size = 20, weight = 'bold')
    plt.xlabel('predictions', size = 12, weight = 'bold')
    plt.ylabel('truth', size = 12, weight = 'bold')
    fig_conf_image_test = fig_conf_test.figure.savefig('src/metrics_figures/test_conf_matrix.png')

    train_cm_df = pd.DataFrame(train_cm, index = np.unique(target['TARGET'].values), columns = np.unique(target['TARGET'].values))
    fig_conf_train = sns.heatmap(train_cm_df, annot=True, cmap = 'Blues')
    plt.title('Train confusion matrix', size = 20, weight = 'bold')
    plt.xlabel('predictions', size = 12, weight = 'bold')
    plt.ylabel('truth', size = 12, weight = 'bold')
    fig_conf_image_train = fig_conf_train.figure.savefig('src/metrics_figures/train_conf_matrix.png')

    #roc_curve
    try:
        train_roc_curve = RocCurveDisplay.from_estimator(pipe, X_train, y_train)
    except:
        train_roc_curve = RocCurveDisplay.from_estimator(pipe, X_train.values, y_train)
    train_roc_curve_plot = train_roc_curve.plot().figure_
    plt.title('Train ROC_CURVE', size = 20, weight = 'bold')
    train_roc_image = train_roc_curve_plot.savefig('src/metrics_figures/train_roc_curve.png')

    try:
        test_roc_curve = RocCurveDisplay.from_estimator(pipe, X_test, y_test)
    except:
        test_roc_curve = RocCurveDisplay.from_estimator(pipe, X_test.values, y_test)
    test_roc_curve_plot = test_roc_curve.plot().figure_
    plt.title('Test ROC_CURVE', size = 20, weight = 'bold')
    test_roc_image = test_roc_curve_plot.savefig('src/metrics_figures/test_roc_curve.png')

    #Interprétation globale du modèle - features importance 
    if model_type == 'HistGradientBooster':
        feature_importance = permutation_importance(pipe, X_train, y_train, n_repeats=5, random_state=11, n_jobs=1)['importances_mean']
    elif model_type == 'GradientBooster':
        feature_importance = permutation_importance(pipe, X_train, y_train, n_repeats=5, random_state=11, n_jobs=1)['importances_mean']
    else:
        try:
            feature_importance = pipe['model'].coef_[0]
        except:
            feature_importance = pipe['model'].feature_importances_
    features = train.columns
    feature_importance = list(feature_importance)
    while len(feature_importance) < len(features):
        feature_importance.append(0)
    importance = pd.Series(feature_importance, index=features).sort_values(ascending=True)
    importance_df = pd.DataFrame({'features' : features, 'importance' : feature_importance, 'abs_importance' : np.abs(feature_importance)})

    #On se place sur les 15 features avec l'importance la plus forte
    importance_df_loc = importance_df.sort_values(by = 'abs_importance', ascending = True).iloc[-15:]
    importance_df_loc = importance_df_loc.sort_values(by = 'importance', ascending = True)

    #Affichage des features importances les plus significatives
    importance_fig = plt.figure(figsize = (15,15))
    plt.rcParams.update({'font.size' : 12})
    plt.title('Features importances - Interprétation globale du modèle', size = 30, weight = 'bold')
    plt.barh(importance_df_loc['features'].values, importance_df_loc['importance'].values)
    plt.grid(True, ls = '--')
    plt.xlabel('Feature Importance', size = 20, weight = 'bold')
    rot = plt.yticks(rotation = 15)
    for i in os.listdir('src/feature_importance'):
        os.remove('src/feature_importance/' + i)
    importance_fig.savefig('src/feature_importance/feature_importances.png', bbox_inches='tight')

    #Les permutation importances ne sont pas conservée par manque de ressources computationnelles, le code est néanmoins disponible (pour une éventuelle version améliorée)
    '''
    train_perumtation_importance = pd.DataFrame(permutation_importance(pipe, X_train, y_train, n_repeats=5, random_state=11, n_jobs=1)['importances'])
    test_perumtation_importance = pd.DataFrame(permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=11, n_jobs=1)['importances'])
    for i in os.listdir('src/permutation_importance'):
        os.remove('src/permutation_importance/' + i)
    train_perumtation_importance.to_csv('src/permutation_importance/train_importance.csv', index = False)
    test_perumtation_importance.to_csv('src/permutation_importance/test_importance.csv', index = False)
    '''

    with mlflow.start_run(experiment_id=experiment_id, run_name = new_run_name, nested=False) as run:
        mlflow.set_tag('Train_Test_Split_Random_State', str(Rand_State))
        mlflow.set_tag('Test_Size', str(Test_Size))
        mlflow.log_artifacts("src/preprocessed", artifact_path = 'preprocessed')
        run_id = run.info.run_id
        for i in best_params.keys():
            mlflow.log_param(i, best_params[i])
        mlflow.log_metric('Val_score', grid.best_score_)
        mlflow.log_metric('Train_AUC', train_auc)
        mlflow.log_metric('Test_AUC', test_auc)
        mlflow.log_metric('Train_ACC', train_acc)
        mlflow.log_metric('Test_ACC', test_acc)
        mlflow.log_metric('Train_REC', train_rec)
        mlflow.log_metric('Test_REC', test_rec)
        mlflow.log_metric('Inference_time', inference_time)
        mlflow.log_metric('Training_time', train_time)
        mlflow.log_metric('Train_F1', train_f1)
        mlflow.log_metric('Test_F1', test_f1)
        mlflow.log_metric('Train_balanced_ACC', train_balanced_acc)
        mlflow.log_metric('Test_balanced_ACC', test_balanced_acc)
        mlflow.log_metric('Test_custom_score', test_score)
        mlflow.log_metric('Train_custom_score', train_score)
        mlflow.log_artifacts('src/cv_results', artifact_path = "cv_results")
        mlflow.log_artifacts('src/hyper_params', artifact_path = "hyper_params")
        mlflow.log_artifacts("src/train_test", artifact_path = 'train_test')
        mlflow.log_artifacts("src/metrics_figures", artifact_path = 'metrics_figures')
        #mlflow.log_artifacts("src/permutation_importance", artifact_path = 'permutation_importance')
        mlflow.log_artifacts("src/feature_importance", artifact_path = 'feature_importance')
        if model_type == 'LightGBM':
            mlflow.lightgbm.log_model(pipe, experiment_name + '_' + new_run_name + '_' + 'Classifier')
        else:
            mlflow.sklearn.log_model(pipe, experiment_name + '_' + new_run_name + '_' + 'Classifier')
    #Désallocation des variables mises en mémoire
    del train, num_mask, cat_mask, Rand_State, Test_Size, X_train, X_test, y_train, y_test, std, pca, model_type, classifier, scaler, pca_state, pipe
    gc.collect()
    del grid_params, param_keys, score, grid, best_params, results, param_df, json_export, y_train_pred, y_test_pred, train_auc, test_auc, train_acc, test_acc
    gc.collect()
    del train_rec, test_rec, train_f1, test_f1, train_balanced_acc, test_balanced_acc, train_score, test_score, test_cm, train_cm, test_cm_df, fig_conf_test, fig_conf_image_test
    gc.collect()
    del train_cm_df, fig_conf_train, fig_conf_image_train, train_roc_curve, train_roc_curve_plot, train_roc_image, test_roc_curve, test_roc_curve_plot, test_roc_image, feature_importance
    gc.collect()
    return params_float, new_run_name