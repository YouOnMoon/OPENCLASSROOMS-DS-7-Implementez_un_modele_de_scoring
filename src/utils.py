'''Ce fichier contient différentes fonctions très utiles, et souvent utilisées dans d'autres fichiers, comme train.py et ../app.py.
Certaines de ces fonctions permettent la visualisation ou encore le traitement des données, ...'''

#Importation des librairies utiles

#Librairies de destion système
import os
import gc
import shutil
import pickle as pkl
import yaml
from yaml.loader import SafeLoader

#librairies de manipulation des données
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

#Librairies de visualisation des données
import matplotlib.pyplot as plt

'''Cette fonction n'est pas utilisée dans notre cas, mais a été utile pour le développement.
Elle crée un générateur qui permet d'importer les jeux de données du repértoire ../input'''
#Fonction d'importation des jeux de données
def data_importer(init_path):
    if '\\' in init_path:
        path = init_path.replace('\\', '/')
    else:
        path = init_path
    os.chdir(path)
    files_list = os.listdir()
    files = []
    for i in files_list:
        tmp = i[:-4]
        files.append(tmp)
        locals()[tmp] = pd.read_csv(i, encoding = 'iso8859_15')
        files.append(locals()[tmp])
    yield(files)

'''Cette fonction permet de sélectionner un fichier à partir du généréteur précédent.
Non utilisée dans notre cas, mais a servi au développement.'''
#Fonction de selection d'un dataframe en particulier
def file_selector(generator, n):
    counter = 0
    for i in generator:
        counter =+1
    if n >= counter:
        print('Entered index higher than number of available files, n is set to : {}'.format(counter - 1))
        n = counter - 1
    for j, k in zip(generator, range(counter)):
        if k == n:
            return j

#Fonction permettant de créer un histogramme représentant la distribution des coefficiets de Pearson au sei d'un jeux de données
#Utilisée à la page DataAnalysis de l'application
def correlation_lister(df):
    for i in df.columns:
        if df[i].dtypes == 'object':
            df = df.drop([i], axis = 1)
    Corr = df.corr()
    Corr = Corr.mask(np.triu(np.ones(Corr.shape)) == 1)
    Correlated_features = []
    Explored_features = []
    for i in Corr.index:
        Explored_features.append(i)
        for j in Corr.columns:
            if (j != i):
                Correlated_features.append([i,j, Corr.loc[i, j]])
    Corr_df = pd.DataFrame(Correlated_features, columns = ['Feature_A', 'Feature_B', 'Pearson'])
    Corr_df = Corr_df.sort_values(by = 'Pearson', ascending = False)
    fig = plt.figure(figsize = (10,8))
    plt.hist(Corr_df['Pearson'], bins = 200)
    plt.grid(True, ls = '--', color = 'navy')
    plt.title('Distribution des coefficients de Pearson de nos variables', size = 20, weight = 'bold')
    plt.xlabel('Coefficient de Pearson', weight = 'bold', size = 15)
    plt.ylabel('Décompte des coeffcients de Pearson', size = 15, weight = 'bold')
    del Corr, Correlated_features, Explored_features
    gc.collect()
    return fig, Corr_df

#Fonction permettant de récupérer le tableau des coefficients de Pearson d'un dataframe, colonne par colonne
def correlation_lister_nomap(df):
    for i in df.columns:
        if df[i].dtypes == 'object':
            df = df.drop([i], axis = 1)
    Corr = df.corr()
    Corr = Corr.mask(np.triu(np.ones(Corr.shape)) == 1)
    Correlated_features = []
    Explored_features = []
    for i in Corr.index:
        Explored_features.append(i)
        for j in Corr.columns:
            if (j != i):
                Correlated_features.append([i,j, Corr.loc[i, j]])
    Corr_df = pd.DataFrame(Correlated_features, columns = ['Feature_A', 'Feature_B', 'Pearson'])
    Corr_df = Corr_df.sort_values(by = 'Pearson', ascending = False)
    del Corr, Correlated_features, Explored_features
    gc.collect()
    return Corr_df

#Fonction permettant de plotter la distribution d'une variable quantitative d'un dataframe donné, sous la forme d'un histplot
def Quantitative_Distribution_Plotter(df, feature):
    fig = plt.figure(figsize = (10,5))
    plt.title('Distribution de la variable : {}'.format(feature), size = 20, weight = 'bold')
    plt.hist(df[feature], bins = np.min(np.array([len(df), 200])), color = 'crimson')
    plt.ylabel("Nombre d'occurences", weight = 'bold', size = 15)
    plt.xticks(rotation = 30)
    plt.grid(True, ls = '--')
    return fig

#Fonction permettant de plotter la distribution d'une variable qualitative d'un dataframe donné, sous la forme d'un pieplot
def Qualitative_Distribution_Plotter(df, feature):
    tmp = df[feature].value_counts()
    labels = tmp.index
    values = tmp.values
    fig = plt.figure(figsize = (10,5))
    plt.title('Distribution de la variable : {}'.format(feature), size = 20, weight = 'bold')
    plt.pie(values, labels = labels, autopct= '%1.0f%%')
    plt.xlabel('Valeurs prises par la variable {}'.format(feature), weight = 'bold', size = 15)
    plt.xticks(rotation = 60)
    plt.grid(True, ls = '--')
    del tmp, labels, values
    gc.collect()
    return fig

'''Cette fonction permet de lister les expériences crée dans le dossier de tracking mlflow.
Nous bouclons sur l'ensemble des répertoires contenus dans le répertoire mlruns, en passant le repértoire .trahs, et pour chaque repértoire, nous 
lisons le nom du repértoire comme ID de l'expérience, et nous cherhcons le nom de l'expérience dans le fichier meta.yaml.
Puis nous créons un dictionnaire dans lequel les clés seront les noms des expériences, et les valeurs les ID des expériences.'''
def experiments_lister():
    os.chdir('mlruns')
    exp_names = []
    exp_ids = []
    all_files = os.listdir()
    filter_1 = [file for file in all_files if not file.isalpha()]
    if '.trash' in filter_1:
        filter_2 = [file for file in filter_1 if file != '.trash']
    else:
        filter_2 = filter_1
    filter_2 = pd.DataFrame(filter_2, columns = ['A']).sort_values(by = 'A')['A'].values
    for i in filter_2:
        if len(filter_2) == 0:
            break
        else:
            os.chdir(i)
            with open('meta.yaml') as f:
                tmp = yaml.load(f, Loader=SafeLoader)
            exp_names.append(tmp['name'])
            exp_ids.append(tmp['experiment_id'])
            os.chdir('..')
    dictionnary = {}
    for i in range(len(exp_names)):
        dictionnary[exp_names[i]] = exp_ids[i]
    del all_files, filter_1, filter_2, f, tmp, exp_ids, exp_names
    gc.collect()
    return dictionnary

'''Cette fonction permet de lister l'ensemble des runs d'une expérience donnée.
Pour cela nous nous placons dans le repéroitre mlruns/exp_id avec l'id de l'expérience, et nous listons les id des runs via les noms de reprétoires présents,
et les noms des runs via les fichiers meta.yaml.
Nous retournons un dictionnaire dans lequelles clés sont les noms des runs, et les valeurs les id des runs.'''
def run_lister(exp_id):
    os.chdir('mlruns')
    os.chdir(exp_id)
    runs_ids = [rep for rep in os.listdir() if rep != 'meta.yaml' and rep != 'tags']
    runs_names = []
    for i in runs_ids:
        os.chdir(i)
        try:
            with open('meta.yaml') as f:
                tmp = yaml.load(f, Loader=SafeLoader)
            runs_names.append(tmp['run_name'])
        except:
            pass
        os.chdir('..')
    os.chdir('..')
    dictionnary = {}
    for i in range(len(runs_names)):
        dictionnary[runs_names[i]] = runs_ids[i]
    del runs_ids, runs_names, f, tmp
    gc.collect()
    return dictionnary

'''Fonction permettant de lister les métriques d'une run, d'une expérience donnée sous forme de DataFrame.'''
def run_metrics(exp_id, run_id, run_name):
    #Chemin vers les métriques loggées
    path = 'mlruns/' + exp_id + '/' + run_id + '/metrics'
    #Initialisation des listes de noms et de valeurs de métriques
    metric = []
    val = []
    #Lecture des noms et valeurs es métriques et ajout aux listes pécédentes
    for i in os.listdir(path):
        tmp = open(path + '/' + i, 'r')
        tmp2 = tmp.read().split(' ')[1]
        metric.append(i)
        val.append(tmp2)
    metrics_dict = {}
    #Création d'un dictionnaire avec les noms en clés et valeurs en valeurs
    for i in range(len(metric)):
        metrics_dict[metric[i]] = [val[i]]
    #Mise sous forme de DataFrame
    metrics_df = pd.DataFrame(metrics_dict, index = [run_name])
    #Désallocation de la mémoire pour les variables inutiles à ce stade
    del metrics_dict, metric, val, tmp, tmp2, path, exp_id, run_id, run_name
    gc.collect()
    return metrics_df

'''Fonction permettnt de supprimer définitivement une run.'''
def del_run(exp_name, run_id):
    os.chdir('mlruns')
    os.chdir(exp_name)
    #Suppression de toute l'arborescence de fichiers de la run
    shutil.rmtree(run_id, ignore_errors=False, onerror=None)
    del exp_name, run_id
    gc.collect()

'''Fonction permettant de supprimer définitivement une expérience en supprimant toute l'arborescence de fichiers du répertoire.
Elle est notamment utilisée à la page Experiences de notre application, utilisable après avoir sélectionnée une expérience et cliqué sur un boutton.'''
def del_experiment(exp_id):
    os.chdir('mlruns')
    for i in os.listdir(exp_id):
        try:
            shutil.rmtree(exp_id + '/' + i)
        except:
            os.remove(exp_id + '/' + i)
        else:
            pass
    os.rmdir(exp_id)
    del exp_id
    gc.collect()

#Fonction créant un plot comparant les valeurs d'une métrique pour toutes les run d'une expérience donnée
#Cette fonction prend comme argument en entrée le résultat d la fonction run_metrics listant les métriques d'une run
#Elle est notamment utilisée à la page Experiences de notre application pour comparer les runs
def runs_plotter(df, metric):
    fig = plt.figure(figsize = (10,5))
    tmp = df.sort_values(by = metric, ascending = True)[metric]
    plt.bar(tmp.index, tmp.values)
    plt.xticks(rotation = 80)
    plt.title('Comparaisons des runs suivant la métrique {}'.format(metric), size = 15,
    weight = 'bold')
    plt.xlabel('Runs', size = 12, weight = 'bold')
    plt.ylabel('Métrique {}'.format(metric), size = 12, weight = 'bold')
    plt.grid(True, ls = '--')
    del df, metric, tmp
    gc.collect()    
    return fig

#Cette fonction permet d'importer un modèle, et les jux de données d'une run pour une expérience donnée
#Elle n'est pas utilisée dans notre cas, mais a servi pour le développement
def import_model(exp_name, run_id):
    os.chdir('mlruns')
    os.chdir(exp_name)
    os.chdir(run_id)
    os.chdir('artifacts/Classifier')
    with open('model.pkl', 'rb') as f:
        model = pkl.load(f)
    os.chdir('../preprocessed')
    test = pd.read_pickle('test.pkl')
    train = pd.read_pickle('train.pkl')
    target = pd.read_pickle('target.pkl')
    del exp_name, run_id
    gc.collect()
    return model, train, test, target

'''Fonction permettant d'utiliser un modèle pour faire des prédictions sur l'ensemble d'un jeu de données.
Elle a servi de base pour l'utilisation des modèles en inférence sur l'ensemble du jeu de données de production.'''
def full_data_predict(df, model):
    #Initialisation des listes de predictions et probabilités d'appartenance à la classe 1
    predictions = []
    predict_proba = []
    #Inférence sur chaque individu du jeu de données
    for i in range(len(df)):
        inputs = df.iloc[i].values.reshape(1,-1)
        predictions.append(model.predict(inputs))
        predict_proba.append(model.predict_proba(inputs))
    predict_proba = np.array(predict_proba)
    #Mise en forme d'un DataFrame concaténant les résultats
    tmp = pd.DataFrame({'Result' : [0, 1], 'Target' : ['Credit Granted', 'Credit Refused']})
    predictions = pd.DataFrame(predictions, columns = ['Result'])
    predictions = predictions.merge(tmp)
    predictions = predictions.drop(['Result'], axis = 1)
    predictions = predictions.transpose()
    predict_proba = pd.DataFrame(predict_proba.reshape(len(df),2), columns = ['Credit Granted', 'Credit Refused']).transpose()
    del tmp, inputs, df, model
    gc.collect()
    return predictions, predict_proba

'''Fonction permettant de supprime les features ayant un taux de valeurs manquantes supérieur à un certain threshold dans un dataframe.
Elle est utilisée à la création des expériences, à la page DataProcessing de notre application.'''
def nan_dropper(df, thresh):
    tmp = df.isna().mean()
    dropped_cols = tmp.loc[tmp.values > thresh].index
    df = df.drop(dropped_cols, axis = 1)
    del tmp, dropped_cols, thresh
    gc.collect()
    return df

'''Fonction selectionnant les features trop corrélés au sein d'un jeux de données, suivant un seuil fourni en input.
Elle est utilisée à la création des expériences, à la page DataProcessing de notre application.'''
def Correlation_filter(df, thresh):
    #Création de la matrice des corrélations
    corr = correlation_lister_nomap(df)
    dropped = []
    #Sélection aléatoire des feature au delà d'un certain seuil.
    tmp = corr.loc[corr['Pearson']>thresh]
    tmp2 = corr.loc[corr['Pearson']<-thresh]
    final = pd.concat([tmp, tmp2])
    final = final.reset_index()
    #Sélection des features deux à deux
    for i in final.index:
        tmp3 = final.loc[final.index == i, 'Feature_B'].values[0]
        dropped.append(tmp3)
    dropped_cols = pd.DataFrame(dropped, columns = ['A'])['A'].unique()
    #Suppression des colonnes retenue par notre critère de sélection
    df = df.drop(dropped_cols, axis = 1)
    del corr, tmp, tmp2, tmp3, dropped_cols, final, dropped
    gc.collect()
    return df       

'''Fonction remplissant automatiquement les valeurs manquantes suivant une stratégie fournie en input, pour toutes les colonnes d'un dataframe
Elle est utilisée à la création des expériences, à la page DataProcessing de notre application.'''
def nan_filler(df, strategy):
    for i in df.columns:
        #Récuperation des valeurs max, min, mean et median des features présentant des valeurs manquantes
        if df.isna().mean()[i] > 0:
            mean = df.loc[~df[i].isna(), i].dropna(axis = 0).mean()
            median = df.loc[~df[i].isna(), i].dropna(axis = 0).median()
            mini = df.loc[~df[i].isna(), i].dropna(axis = 0).min()
            maxi = df.loc[~df[i].isna(), i].dropna(axis = 0).max()
            #Complétion des valeurs manquantes en fonction de la statégie
            if strategy == 'mean':
                df[i] = df[i].fillna(mean)
            elif strategy == 'median':
                df[i] = df[i].fillna(median)
            elif strategy == 'min':
                df[i] = df[i].fillna(mini)
            elif strategy == 'max':
                df[i] = df[i].fillna(maxi)
            else:
                pass
            del mean, median, mini, maxi
            gc.collect()
    del strategy
    gc.collect()
    return df

'''fonction permettant de filtrer les colonnes catégoriques trop corrélées à la target suivant le critère
du chi2, via un seuil.'''
def chi2_dropper(df, target, thresh):
    cat = df.dtypes.loc[df.dtypes == 'object'].index
    categ_df = df[cat]
    p_values = []
    for i in categ_df.columns:
        table = pd.crosstab(categ_df[i], target['TARGET'])
        results_test = chi2_contingency(table)
        p_values.append(results_test[1])
    chi2_df = pd.DataFrame({'cols' : categ_df.columns, 'p_values' : p_values}).sort_values(by = 'p_values', ascending = True)
    dropped = chi2_df.loc[chi2_df['p_values']>thresh, 'cols'].values
    final_df = df.drop(dropped, axis = 1)
    del cat, df, target, thresh, categ_df, p_values, chi2_df, dropped
    gc.collect()
    return final_df

'''Fonction permettant de calculer le coefficient eta_squared entre une variable qualitative, 
et une variable quantitative.'''
def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    del x, y, moyenne_y, classes
    gc.collect()
    return SCE/SCT

'''Fonction permettant de filtrer les variables trop corrélées à la target pour les colonnes
numériques.'''
def eta_squared_dropper(df, target, thresh):
    num = df.dtypes.loc[df.dtypes != 'object'].index
    num_df = df[num]
    eta_squared_coefs = []
    for i in num_df.columns:
        eta_squared_coefs.append(eta_squared(target['TARGET'], num_df[i].values))
    eta_sq_df = pd.DataFrame({'cols' : num, 'eta_sq' : eta_squared_coefs}).sort_values(by= 'eta_sq', ascending = True)
    dropped = eta_sq_df.loc[eta_sq_df['eta_sq']>thresh, 'cols'].values
    final_df = df.drop(dropped, axis = 1)
    del num, df, target, thresh, num_df, eta_squared_coefs, eta_sq_df, dropped
    gc.collect()
    return final_df

'''Fonction permettant de supprimer les colonnes avec trop peu de variance.'''
def std_dropper(df, thresh):
    std_list = []
    for i in df.columns:
        std_list.append(df[i].std())
    std_df = pd.DataFrame({'cols' : df.columns, 'std' : std_list}).sort_values(by= 'std', ascending = True)
    kept_columns = std_df.loc[std_df['std']>thresh, 'cols']
    final_df = df[kept_columns]
    return final_df

'''Fonction permettant de sélectionner les featurs categoriques entre eux de manière à 
limiter les corrélations entre features qualitatives de notre jeu de données.'''
def chi2_selector(df, thresh):
    cat = df.dtypes.loc[df.dtypes == 'object'].index
    categ_df = df[cat]
    p_values = []
    feature_A = []
    feature_B = []
    n = 0
    for i in categ_df.columns:
        n+=1
        for j in categ_df.columns[n:]:
            if (j!=i):
                table = pd.crosstab(categ_df[i], categ_df[j])
                results_test = chi2_contingency(table)
                p_values.append(results_test[1])
                feature_A.append(i)
                feature_B.append(j)
    chi2_df = pd.DataFrame({'A' : feature_A, 'B' : feature_B, 'p_values' : p_values})
    tmp = chi2_df.loc[chi2_df['p_values']>thresh]
    dropped_features = tmp['A'].unique()
    df = df.drop(dropped_features, axis = 1)
    return df


'''Fonction sélectionnant les features sur la base du coefficient eta_squared entre featurs categoriques et numériques.'''
def eta_squared_selector(df, thresh):
    cat = df.dtypes.loc[df.dtypes == 'object'].index
    categ_df = df[cat]
    num = df.dtypes.loc[df.dtypes != 'object'].index
    num_df = df[num]
    num_cols = []
    cat_cols = []
    eta_squared_list = []
    for i in cat:
        for j in num:
            eta_squared_list.append(eta_squared(df[i], df[j]))
            num_cols.append(j)
            cat_cols.append(i)
    eta_squared_list = np.abs(eta_squared_list)
    concat_df = pd.DataFrame({'Num' : num_cols, 'Cat' : cat_cols, 'eta_sq' : eta_squared_list})
    dropped = concat_df.loc[concat_df['eta_sq']>thresh, 'Cat'].unique()
    df = df.drop(dropped, axis = 1)
    return df


'''Fonction permettant de récupérer l'artéfact cv_results pour une run d'une expérience données.
Elle est utilisée à la page Experiences de notre application pour afficher les résultats de recherche sur grille par validation croisée.'''
def cv_results_returner(exp_id, run_id):
    path = '../mlruns/' + exp_id + '/' + run_id + '/artifacts/cv_results/'
    file = 'results.csv'
    df = pd.read_csv(path + file)
    del path, file, run_id, exp_id
    gc.collect()
    return df
