'''Dans ce fichier, nous effectuons les tests unitaires relatifs au kernel kaggle ../src/FeatureEngineering.py.
Pour cela, nous testons indépendamment chaqune des fonctions, et nous testons la capacité des fonctions à 
supprimer les valeurs aberrantes.'''

#Importation de sys et os
import sys
import os

#Importation des fonctions à tester
sys.path.append('..')
sys.path.append('../src')
os.chdir('../src')
from FeatureEngineering import application_train_test
from FeatureEngineering import bureau_and_balance
from FeatureEngineering import previous_applications
from FeatureEngineering import pos_cash
from FeatureEngineering import installments_payments
from FeatureEngineering import credit_card_balance
from FeatureEngineering import main

#Importation de unittest
import unittest

'''Attention, ces tests sont valables uniquement pour la version actuelle des données, notamment par rapport à la taille des datarames générés.
En cas de modification des données initiales, il faut modifier les valeurs de tailles de dataframes générés.'''

class Features_Engineering_Creation(unittest.TestCase):

    #Tests de la fonction application_train_test
    def test_df_creation(self):
        df = application_train_test()
        num_cols = len(df.columns)
        num_obs = len(df)
        unique_days_employed_values = df['DAYS_EMPLOYED'].unique()
        #Test taille du dataframe final
        self.assertEqual(num_cols, 248)
        self.assertEqual(num_obs,  356251)
        #Test valeurs aberrantes dans la colonne 'DAYS_EMPLOYED'
        self.assertNotIn(365243, unique_days_employed_values)

    #Tests de la fonction bureau_and_balance
    def test_bureau_and_balance(self):
        bureau_agg = bureau_and_balance()
        num_cols = len(bureau_agg.columns)
        num_obs = len(bureau_agg)

        #Nom de l'indice - important pour futur jointure
        index_name = bureau_agg.index.name

        #Test taille du dataframe final
        self.assertEqual(num_cols, 116)
        self.assertEqual(num_obs,  305811)
        #Nom de l'indice
        self.assertEqual(index_name, 'SK_ID_CURR')

    def test_previous_applications(self):
        prev_agg = previous_applications()
        num_cols = len(prev_agg.columns)
        num_obs = len(prev_agg)

        #Nom de l'indice - important pour futur jointure
        index_name = prev_agg.index.name

        #Check valeurs aberrantes
        bool_list_test = []
        for i in prev_agg.columns:
            tmp = 365243 in prev_agg[i].values
            bool_list_test.append(tmp)
 
        #Test taille du dataframe final
        self.assertEqual(num_cols, 249)
        self.assertEqual(num_obs,  338857)
        #Test valeurs aberrantes
        for i in bool_list_test:
            self.assertFalse(i)
        #Nom de l'indice
        self.assertEqual(index_name, 'SK_ID_CURR')

    def test_pos_cash(self):
        pos_agg = pos_cash()
        num_cols = len(pos_agg.columns)
        num_obs = len(pos_agg)
        #Nom de l'indice
        self.assertEqual(pos_agg.index.name, 'SK_ID_CURR')
        #Test taille du dataframe final
        self.assertEqual(num_cols, 18)
        self.assertEqual(num_obs,  337252)

    def test_installments_payments(self):
        ins_agg = installments_payments()
        num_cols = len(ins_agg.columns)
        num_obs = len(ins_agg)
        #Nom de l'indice
        self.assertEqual(ins_agg.index.name, 'SK_ID_CURR')
        #Test taille du dataframe final
        self.assertEqual(num_cols, 26)
        self.assertEqual(num_obs,  339587)

    def test_credit_card(self):
        cc_agg = credit_card_balance()
        num_cols = len(cc_agg.columns)
        num_obs = len(cc_agg)
        #Nom de l'indice
        self.assertEqual(cc_agg.index.name, 'SK_ID_CURR')
        #Test taille du dataframe final
        self.assertEqual(num_cols, 141)
        self.assertEqual(num_obs,  103558)

    def test_main(self):
        df = main(debug = False)
        num_cols = len(df.columns)
        num_obs = len(df)
        #Test taille du dataframe final
        self.assertEqual(num_cols, 768)
        self.assertEqual(num_obs,  356251)

if __name__ == '__main__':
    unittest.main()