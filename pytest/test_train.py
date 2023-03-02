'''Dans ce fichier, nous testons les fonction du fichier ../src/train.py.
Pour cela, les temps de traitement étants trés longs, nous utilisons une version de code avec un échantillonnage sur les données initiales
(ici à 1000 individus).'''

#Importation de sys et os
import sys
import os

#Ajout des chemins d'accès au fichier train.py
sys.path.append('..')
sys.path.append('../src')
os.chdir('../src')

#Importation des fonctions à tester dans le fichier ../src/train.py
from train import import_data
from train import hyperparam_tuning
from train import smote_hyperparam_tuning
from train import pkl_import
from utils import experiments_lister
from utils import run_lister
os.chdir('..')
#Importation unittest
import unittest

'''ATTENTION : Pour réaliser ces tests en un temps raisonnable, il est nécessaire de placer le paramètre debug de la fonction main
appelée dans la fonction import_data à True.'''

#Création d'un dataframe couvrant tous les cas d'utilisation de la fonction import_data

pickle_export = True
nan_thresh = 0.08
corr_thresh = 0.5
nan_strategy = ['median', 'mean', 'min', 'max', 'none']
exp_desc = 'unittest test all cases.'

#Classe de test de la fonction import_data
class all_cases_test_import_data(unittest.TestCase):

    #Fonction de test itérative sur toutes les manières de créer un nouveau jeu de données
    def test_import_data(self):
        exp_id_list = []
        for i in nan_strategy:
            exp_name = 'exp_' + str(i)
            train, test, target, EXPERIMENT_ID = import_data(pickle_export, nan_thresh, corr_thresh, i, exp_name,  exp_desc)
            exp_id_list.append(EXPERIMENT_ID)
            if i != 'none':
                self.assertTrue(train.isna().mean().sum() == 0 )
                self.assertTrue(test.isna().mean().sum() == 0 )
        exp_dict = experiments_lister()
        os.chdir('..')
        real_ids = []
        for i in exp_dict.keys():
            real_ids.append(exp_dict[i])
        for i in exp_id_list:
            self.assertIn(i, real_ids)

    #Fonction de test iterative de la fonction hyperparam_tuning et de la fonction smote_hyperparam_tuning
    def test_hyperparam_tuning(self):
        exp_name = 'unittest_hyperparam_tuning'
        train, test, target, EXPERIMENT_ID = import_data(True, 0.08, 0.5, 'mean', exp_name,  exp_desc)
        train, test, target, exp_id, exp_name = pkl_import()
        json_export = True
        scaler = [True, False]
        pca_state = [True, False]
        #Le gradientbooster n'ayant pas de méthode classweight, il sera automatiquement remplacé par histgradientbosster
        model_types = ['GradientBooster', 'HistGradientBooster', 'LightGBM', 'LogReg']
        n = 0
        new_runs_list = []
        for i in scaler:
            for j in pca_state:
                for mod in model_types:

                    #Hyperparam_tuning
                    if mod == 'LogReg':
                        grid_params = {'C' : [0.1,0.2]}
                        params = {}
                    else:
                        grid_params = {'learning_rate' : [0.1,0.2]}
                        params = {}
                    n+=1
                    new_run_name = 'run_' + str(n)
                    os.chdir('src')
                    params_float, new_run_name = hyperparam_tuning(train, target, json_export, i, j, params, grid_params, mod, new_run_name, exp_name, exp_id)
                    new_runs_list.append(new_run_name)

                    #Smote hyperparam tuning
                    if mod == 'LogReg':
                        grid_params = {'C' : [0.1,0.2]}
                        params = {}
                    else:
                        grid_params = {'learning_rate' : [0.1,0.2]}
                        params = {}
                    n+=1
                    new_run_name = 'run_' + str(n)
                    os.chdir('src')
                    params_float, new_run_name = smote_hyperparam_tuning(train, target, json_export, i, j, params, grid_params, mod, new_run_name, exp_name, exp_id)
                    new_runs_list.append(new_run_name)
        run_dict = run_lister(exp_id)
        os.chdir('..')    

        #Test de présence des repértoires loggés dans le repértoire central mlruns            
        for i in new_runs_list:
            self.assertIn(i, run_dict.keys())
            self.assertTrue(os.path.exists('mlruns/' + exp_id + '/' + str(run_dict[i]) + '/artifacts'))
            self.assertTrue(os.path.exists('mlruns/' + exp_id + '/' + str(run_dict[i]) + '/metrics'))
            self.assertTrue(os.path.exists('mlruns/' + exp_id + '/' + str(run_dict[i]) + '/params'))


