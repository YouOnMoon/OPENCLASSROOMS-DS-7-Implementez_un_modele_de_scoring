'''Dans ce fichier, nous effectuons les tests unitaires des fonctions du fichier .. /src/utils.py.
Ces fonctions servent principalement à pré-traiter, visualiser et organiser les données via mlflow.
Nous allons donc créer un premier jeu de données de test contenant des features de types object et float,
ainsi que des targets pour tester les différentes fonctions.'''

#Importation de sys et os
import sys
import os

#Ajout des chemins d'accès au fichier utils.py
sys.path.append('..')
sys.path.append('../src')
os.chdir('../src')

#Importation des fonctions à tester dans le fichier ../src/utils.py
from utils import correlation_lister_nomap
from utils import Quantitative_Distribution_Plotter
from utils import Qualitative_Distribution_Plotter
from utils import experiments_lister
from utils import run_lister
from utils import full_data_predict
from utils import nan_dropper
from utils import Correlation_filter
from utils import nan_filler
from utils import chi2_dropper
from utils import eta_squared_dropper
from utils import std_dropper
from utils import chi2_selector
from utils import eta_squared_selector
from utils import cv_results_returner
from utils import run_metrics

#Importation unittest
import unittest

#Importation numpy, pandas et matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importation d'un modèle sklearn 
from sklearn.linear_model import LogisticRegression


#Création d'un jeu de données de test
num_col_1 = [0.2, 0.4, 0.6, np.nan, 0.6]
num_col_2 = [0.1, 0.3, 0.4, 0.5, 0.3]
cat_col_1 = [0, 1, 1, 0, np.nan]
cat_col_2 = [0, 1, 0, 0, 1]

targets_list = [0, 0, 0, 0, 1]

test_df = pd.DataFrame({'num_1' : num_col_1, 'num_2' : num_col_2, 'cat_1' : cat_col_1, 'cat_2' : cat_col_2})
test_df['num_1'] = test_df['num_1'].astype(float)
test_df['num_2'] = test_df['num_2'].astype(float)
test_df['cat_1'] = test_df['cat_1'].astype(object)
test_df['cat_2'] = test_df['cat_2'].astype(object)

target = pd.DataFrame(targets_list, columns = ['TARGET'])

#Test des fonctions de preprocessing du fichier utils.py
class preprocessing_functions_tester(unittest.TestCase):

    #Tests de la fonction correlation_lister_nomap
    def test_correlation_lister_nomap(self):
        corr_df = correlation_lister_nomap(test_df)

        #Vérification liste des colonnes
        for i in corr_df.columns:
            self.assertIn(i, ['Feature_A', 'Feature_B', 'Pearson'])
        self.assertEqual(len(corr_df.columns), 3)

        #Test opérations éffectuée sur les seuls colonnes numériques
        self.assertNotIn('cat_1', corr_df['Feature_A'].values)
        self.assertNotIn('cat_2', corr_df['Feature_A'].values)
        self.assertNotIn('cat_1', corr_df['Feature_B'].values)
        self.assertNotIn('cat_2', corr_df['Feature_B'].values)

        #Test présence des deux colonnes numériques
        self.assertIn('num_1', corr_df['Feature_A'].values)
        self.assertIn('num_2', corr_df['Feature_A'].values)
        self.assertIn('num_1', corr_df['Feature_B'].values)
        self.assertIn('num_2', corr_df['Feature_B'].values)

    #Tests de la fonction full_data_predict
    def test_full_data_predict(self):
        #Hypothèses sur les inputs
        assert len(target) == len(test_df)
        assert len(target.columns) == 1

        #Création du modèle
        model = LogisticRegression()
        test_df_clean = test_df.fillna(0)
        model.fit(test_df_clean.values, target.values)
        predictions, predict_proba = full_data_predict(test_df_clean, model)

        #Test de présence des variables des dataframes
        self.assertNotIn('Result', predictions.columns)
        self.assertNotIn('Target', predictions.columns)
        self.assertEqual(len(predictions.columns), len(target))
        self.assertIn('Credit Granted', predict_proba.index)
        self.assertIn('Credit Refused', predict_proba.index)
    
    #Test de la fonction nan_dropper
    def test_nan_dropper(self):
        #Utilisation de la fonction et récupération du nombre de colonens attendus
        full_cols = test_df.isna().mean().loc[test_df.isna().mean()>0].index
        n_full_cols = len(full_cols)
        cleaned_df = nan_dropper(test_df, 0.00001)

        #Test du nombre de colonnes et de la présences de valeurs manquantes
        self.assertEqual(len(cleaned_df.columns), n_full_cols)
        self.assertTrue(cleaned_df.isna().mean().sum() == 0)

    #Test de la fonction Correlation_filter
    def test_Correlation_filter(self):
        test_df_clean = test_df.fillna(0)
        #Création d'une colonne fortement corrélée
        test_df_clean['num_3'] = test_df_clean['num_1']
        #Création d'une colonne qualitative corrélée à la colonne numérique 2
        test_df_clean['cat_3'] = test_df_clean['num_2'].astype(object)
        #remise en place des types
        for i in test_df_clean.columns:
            if 'cat' in i:
                test_df_clean[i] = test_df_clean[i].astype(object)
            else:
                test_df_clean[i] = test_df_clean[i].astype(float)
        precessed_df = Correlation_filter(test_df_clean, 0.95)
    
        #Test du nombre de colonnes (une seule colonne sensée être supprimée)
        self.assertEqual(len(precessed_df.columns) - 1, len(test_df.columns))
        #Test présence des colonnes qualitatives (non sujettes au filtre, même si corrélées entre elles)
        self.assertIn('cat_3', test_df_clean.columns)

    #Test de la fonction nan_filler
    def test_nan_filler(self):
        #Itération sur les stratégies différentes
        for i in ['mean', 'median', 'min', 'max']:
            preprocessed_df = nan_filler(test_df, i)

            #test présence des valeurs manquantes
            self.assertTrue(preprocessed_df.isna().mean().sum() == 0)

            #Test des valeurs de complétion
            if i == 'mean':
                self.assertEqual(preprocessed_df.mean().values.all(), test_df.mean().values.all())
            elif i == 'median':
                self.assertEqual(preprocessed_df.median().values.all(), test_df.median().values.all())
            elif i == 'min':
                self.assertEqual(preprocessed_df.min().values.all(), test_df.min().values.all())
            else:
                self.assertEqual(preprocessed_df.max().values.all(), test_df.max().values.all())
        
        #Cas où l'on ne compléte pas les valeurs manquantes
        preprocessed_none = nan_filler(test_df, 'None')
        #test similarité des valeurs manquantes
        self.assertTrue(preprocessed_df.isna().mean().sum() == test_df.isna().mean().sum())

    #Test de la fonction chi2_dropper
    def test_chi2_dropper(self):
        test_df_clean = test_df.fillna(0)
        #Création d'une colonne fortement corrélée, et une autre de type float (la fonction est sensée ignorer les types float)
        test_df_clean['cat_3'] = target['TARGET'].astype('object')
        test_df_clean['num_3'] = target['TARGET'].astype(float)
        #remise en place des types
        for i in test_df_clean.columns:
            if 'cat' in i:
                test_df_clean[i] = test_df_clean[i].astype(object)
            else:
                test_df_clean[i] = test_df_clean[i].astype(float)
        processed_df = chi2_dropper(test_df_clean, target, 0.95)

        #Test du nombre de colonnes et des colonnes en présence (une suppression prévue) - conservation de la colonne num_3
        self.assertTrue(len(processed_df.columns) == 5)
        self.assertIn('num_3', processed_df.columns)

    def test_eta_squared_dropper(self):
        test_df_clean = test_df.fillna(0)
         #Création d'une colonne fortement corrélée, et une autre de type object (la fonction est sensée ignorer les types object)
        test_df_clean['num_3'] = target['TARGET'].astype(float)
        test_df_clean['cat_3'] = target['TARGET']
        #remise en place des types
        for i in test_df_clean.columns:
            if 'cat' in i:
                test_df_clean[i] = test_df_clean[i].astype(object)
            else:
                test_df_clean[i] = test_df_clean[i].astype(float)
        processed_df = eta_squared_dropper(test_df_clean, target, 0.95)

        #Test du nombre de colonnes et des colonnes en présence (une suppression prévue) - conservation de la colonne cat_3
        self.assertTrue(len(processed_df.columns) == 5)
        self.assertIn('cat_3', processed_df.columns)

    def test_std_dropper(self):
        #Création d'une colonne trés sparse, fixation du seuil trés bas
        test_df_clean = test_df.fillna(0)
        test_df_clean['expected_dropped'] = [0, 0, 0, 0, 0]
        processed_df = std_dropper(test_df_clean, 0.0000001)

        #Test nombre de colonnes - expected_dropped doit être supprimée
        self.assertTrue(len(processed_df.columns) == 4)
        self.assertNotIn('expected_dropped', processed_df.columns)

    def test_chi2_selector(self):
        #Nettoyage du jeu de données te création de colonnes trés corrélées en elles, dont une de type float
        test_df_clean = test_df.fillna(0)
        test_df_clean['cat_3'] = test_df_clean['cat_1']
        test_df_clean['num_3'] = test_df_clean['cat_1']
        #remise en place des types
        for i in test_df_clean.columns:
            if 'cat' in i:
                test_df_clean[i] = test_df_clean[i].astype(object)
            else:
                test_df_clean[i] = test_df_clean[i].astype(float)
        processed_df = chi2_selector(test_df_clean, 0.95)

        #Test du nombre de colonnes et des colonnes en présence (au moins une suppression prévue) - conservation de la colonne num_3
        self.assertTrue(len(processed_df.columns) < 6)
        self.assertIn('num_3', processed_df.columns)

    def test_eta_squared_selector(self):
        #Nettoyage du jeu de données te création de colonnes trés corrélées en elles, dont une de type float
        test_df_clean = test_df.fillna(0)
        test_df_clean['cat_3'] = test_df_clean['cat_1']
        test_df_clean['num_3'] = test_df_clean['cat_1']
        #remise en place des types
        for i in test_df_clean.columns:
            if 'cat' in i:
                test_df_clean[i] = test_df_clean[i].astype(object)
            else:
                test_df_clean[i] = test_df_clean[i].astype(float)
        processed_df = eta_squared_selector(test_df_clean, 0.95)

        #Test du nombre de colonnes et des colonnes en présence (au moins une suppression prévue) - conservation de la colonne num_3
        self.assertTrue(len(processed_df.columns) < 6)
        self.assertIn('num_3', processed_df.columns)

#Test des fonction de visualisation des données
class plotting_functions_tester(unittest.TestCase):

    #Test de la fonction d'affichage des features quantitatives
    def test_Quantitative_Distribution_Plotter(self):
        test_df_clean = test_df.fillna(0)
        for i in test_df_clean.columns:
            if 'cat' in i:
                test_df_clean[i] = test_df_clean[i].astype(object)
            else:
                test_df_clean[i] = test_df_clean[i].astype(float)
        #Essai d'appel de la fonction, variable tmp à True en cas de succès
        try:
            fig = Quantitative_Distribution_Plotter(test_df_clean, 'num_1')
            tmp = True
        except:
            pass
            tmp = False
        #Test de succès 
        self.assertTrue(tmp)

    #Test de la fonction d'affichage des features qualitatives
    def test_Qualitative_Distribution_Plotter(self):
        test_df_clean = test_df.fillna(0)
        for i in test_df_clean.columns:
            if 'cat' in i:
                test_df_clean[i] = test_df_clean[i].astype(object)
            else:
                test_df_clean[i] = test_df_clean[i].astype(float)
        #Essai d'appel de la fonction, variable tmp à True en cas de succès
        try:
            fig = Qualitative_Distribution_Plotter(test_df_clean, 'cat_1')
            tmp = True
        except:
            pass
            tmp = False
        #Test de succès 
        self.assertTrue(tmp)

#Test des fonctions de récupération des éléments centralisés par mlflow tracking
class mlflow_tracking_tester(unittest.TestCase):

    #Test d'affichage des expériences
    def test_experiments_runs_lister(self):
        #Essai de la fonction experiment_lister - attente d'un nombre de clés strictement positif
        os.chdir('..')
        exp_dict = experiments_lister()
        self.assertTrue(len(exp_dict.keys())>0)
        os.chdir('..')

        #Essai itératif de la fonction experiment_lister - attente d'un nombre de clés strictement positif
        for i in exp_dict.keys():
            run_dict = run_lister(exp_dict[i])
            self.assertTrue(len(run_dict.keys())>0)
            os.chdir('..')

            #Essai itératif de la fonction run_metrics - vérification du nombre de métriques disponibles - et cv results
            for j in run_dict.keys():
                if j != 'DummyClassifier':
                    metrics_df = run_metrics(exp_dict[i], run_dict[j], j)
                    self.assertTrue(len(metrics_df.columns) == 15)

                    #Test cv_results_returner
                    os.chdir('src')
                    cv_results = cv_results_returner(exp_dict[i], run_dict[j])
                    os.chdir('..')
                    self.assertIn('mean_fit_time', cv_results.columns)
                else:
                    pass

if __name__ == '__main__':
    unittest.main()