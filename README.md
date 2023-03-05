# OPENCLASSROOMS - Parcours Data Scientist
# Projet 7 - *Implémentez un modèle de scoring*

**[https://github.com/YouOnMoon/OC-DS-Projet-7-Implementez_un_modele_de_scoring.git](https://github.com/YouOnMoon/OC-DS-Projet-7-Implementez_un_modele_de_scoring.git)**

# IMPORTANT:

**De manière à pouvoir déployer notre application sur le cloud, nous proposons 3 versions différentes de ce projet, chacune représentée par une branche sur le dépôt github.**

- La première version correspond à la branche main de notre application, et pèse un peu plus de 12 GB. Cette version est la version initiale, et est implémentez de manière à servir nos modèles en localhost, car trop volumineuse pour être déployée sur le cloud avec des ressources raisonnables. La raison en est que les jeux de données initiaux, ainsi que les éléments centralisés par MLFlow sont complets dans cette version.

*NB: Pour installer cette version sur un dépôt local, il est nécessaire d'utiliser Git LFS*.

Cette version met notamment à disposition tous les éléments de notre étude, y compris les notebooks illustratifs, et les fichiers de tests unitaires. 

**Téléchargement de la version main en local : git lfs clone -b main https://github.com/YouOnMoon/OPENCLASSROOMS-DS-7-Implementez_un_modele_de_scoring.git**

Pour servir l'application en localhost, placez-vous dans le répertoire de l'application via un terminal, et utilisez la commande : *python -m uvicorn app:app --reload*

Pour lancer le dashboard en localhost, placez-vous dans le répertoire de l'application via un terminal, et utilisez la commande : *python -m streamlit run dashboard.py*

- La seconde version de l'application est la version light-version-deployment. De manière à pouvoir déployer l'API sur le cloud (en l'occurrence la plateforme Heroku), nous avons mis en place cette version allégée, de manière à s'affranchir de Git LFS, et de passer la taille de notre répertoire de travail sous la barre des 500MB (correspondant à la mémoire maximale autorisée pour les machines virtuelles sur Heroku). Ainsi, les fichiers pythons permettant d'afficher le dashboard, d'effectuer les tests unitaires, et les notebooks ne sont pas présents dans cette version (nous ne conservons que le code permettant de servir l'application). Les différents jeux de données initiaux, ainsi que les éléments centralisés via MLFlow sont échantillonnés, et certaines fonctionnalités lourdes sont modifiées (prédictions groupées passées de 50 à 10 observations aléatoires, impossibilité d'effectuer une prédiction sur l'ensemble du jeu de données de production et impossibilité de lancer l'analyse de DataDrift sur l'ensemble du jeu de données de production, seule l'option colonne par colonne est conservée).

**Téléchargement de la version light-version-deployment sur un dépôt local : git clone -b light-version-deployment https://github.com/YouOnMoon/OPENCLASSROOMS-DS-7-Implementez_un_modele_de_scoring.git**

**[Liens vers le Swagger UI de l'API sur le cloud - https://ocds7ey.herokuapp.com/docs](https://ocds7ey.herokuapp.com/docs)**

*NB: Cette version a pour seul but de déployer l'API sur le cloud de manière à limiter les ressources computationnelles nécessaires. Elle ne conserve que 2 modèles dans notre expérience de référence en production, un LGBMClassifier, et une régression logistique. L'ensemble des modèles sont disponibles sur la version main de l’application.*

- La dernière version de l'application est la version heavy-version-localhost. Initialement prévue pour les tests des fonctionnalités de manière rapide, cette version est un clone de la version light-version-deployment, à laquelle sont intégrées les fichiers pythons permettant d'afficher le dashboard, de réaliser les tests unitaires, et les notebooks illustratifs des fonctionnalités des fichiers du répertoire src. Cette version nous permet notamment de déployer le dashboard sur la plateforme Streamlit Cloud, et les requêtes du dahsboard sont envoyés aux différents end-points de l'API déployée sur Heroku.

*NB: De manière à pouvoir livrer le projet avec une taille de fichier raisonnable, c'est également cette version qui correspond à notre deuxième livrable.*

**Téléchargement de la version heavy-version-localhost sur un dépôt local : git clone -b heavy-version-localhost https://github.com/YouOnMoon/OPENCLASSROOMS-DS-7-Implementez_un_modele_de_scoring.git**

**[Liens vers le dashboard déployé sur le cloud - https://youonmoon-openclassrooms-ds-7-implementez-un--dashboard-bhyhm8.streamlit.app/](https://youonmoon-openclassrooms-ds-7-implementez-un--dashboard-bhyhm8.streamlit.app/)**


## 1- Rappel de la problématique

Nous travaillons pour la société financière "Prêt à dépenser", pour laquelle nous développons un outil permettant de calculer la probabilité de défaut de paiement d'un client demandant un crédit. 

Pour cela, nous nous appuyons sur les données du concours Kaggle disponible à l'adresse suivante :

**[Liens vers les données initiales !](https://www.kaggle.com/c/home-credit-default-risk/data)**

A partir de ces données initiales, nous appliquons des traitements nous permettant de générer un jeu de données d'entraînement, dans lequel chaque observation correspond à une demande de crédit. Un jeu de données cible contenant les résultats est également généré. Dans ce jeu de données, un individu ayant réussi à rembourser son crédit prendra la valeur 0, et un individu en défaut de paiement aura la valeurs 1. 

Dans notre cas, ces deux classes sont présentes de manière déséquilibrés dans le jeu de données, étant donné que 8% des individus se sont retrouvés en défauts de paiement, ce qui sera une contrainte pour la modélisation.

De plus, un troisième jeu de donnés sans étiquettes est généré, qui nous servira de jeu de données de production pour l'utilisation des modèles.

Par la suite, nous nettoyons les données de manière à ne conserver que des variables exploitables pour nos futurs modèles. 

**Dans notre cas, au sein de l'application, l'utilisateur peut générer de nouveaux jeux de données en adaptant le taux de valeurs manquantes toléré par variable, la stratégie de complétion des valeurs manquantes, et le coefficient de Pearson toléré entre les variables quantitatives.**

Le traitement d'un nouveau jeu de données déclenchera la création d'une nouvelle expérience centralisée par **MLFlow Tracking**, et l'entraînement d'un nouveau modèle permettra la création d'une nouvelle run au sein de cette expérience. 

## 2- Répertoire *'input*'

Ce répertoire contient les différents jeux de données initiales de notre étude:

    - application_train.csv
    - application_test.csv
    - bureau.csv
    - bureau_balance.csv
    - POS_CASH.csv
    - previous_applications.csv
    - instalments_payments.csv
    - credit_card_balance.csv

Les différents jeux de données sont unis en un seul via des opérations d'algèbre relationnel pendant le traitement des données, notamment des agrégations sur la variable SK_ID_CURR, et des jointures avec cardinalité un à plusieurs sur la variable SK_ID_CURR autour des fichiers application_trian.csv pour le jeu de données d'entraînement et application_test.csv pour le jeu de données de production.

## 3- Répertoire *'src'*

Ce répertoire contient plusieurs fichiers python indispensables à notre étude. 

Le fichier **utils.py** contient différentes fonctions de traitement et visualisation des données, ainsi que des fonctions de gestion système.

Le fichier **FeatureEngineering.py** contient le kernel Kaggle permettant le feature engineering, et la création d'un jeu de données unique à partir de nos données initiales.

Le fichier **train.py** contient les 4 fonctions qui sont le cœur de notre étude:
- **import_data** permettant de traiter les données initiales par l'appel de la fonction main du kernel Kaggle, et appliquant un nettoyage du jeu de données en supprimant les features inexploitables, trop corrélée à la target, ou encore trop corrélées entre elles. Son utilisation crée une nouvelle expérience MLFlow, dont la première run correspond à un Dummy Classifier avec la stratégie *most frequent* adaptée à un jeu de données avec des étiquettes déséquilibrées.
- **hyperparam_tuning** permettant d'entraîner un modèle à partir d’une expérience donnée. Le modèle est entièrement customisable (choix des préprocesseurs et du type de classifieur) et les hyperparamètres sont optimisées par recherche sur grille par validation croisée. Le seuil de prédiction du modèle est également optimisé de cette manière. Pour cela, nous séparons le jeu de données d'entraînement en un jeu de données d'entraînement, et un de test, avec stratification sur la target. Le meilleur modèle est ensuite entraîné à partir des meilleurs hyperparamètres, nous mesurons les métriques, ainsi que l'interprétation globale du modèle, et nous centralisons le tout dans une nouvelle run MLFlow. Dans le cas de cette fonction, la gestion du déséquilibre des classes est réalisé par la méthode de pénalisation accrue sur la classe minoritaire ('balanced' class weight).
- **smote_hyperparam_tuning** réalise les mêmes opérations que la fonction précédente, mais la gestion du déséquilibre de la target est réalisé via l'utilisation de SMOTE, un algorithme permettant de générer des observations synthétiques sur la classe minoritaire, sur la base des plus proches voisins de chaque observation (le meilleur nombre de plus proches voisins est également optimisé par validation croisée).
- **Fonction coût métier** permettant d'optimiser les hyperparamètres sur la base d'un score basé sur le coût des faux négatifs et des faux positifs.

Les différents sous-répertoires du répertoire src nous servent à enregistrer différents éléments lors de nos traitements et modélisations, dans l'idée de les centraliser grâce à MLFlow par la suite. Par exemple, le répertoire cv_results enregistre temporairement les résultats de la recherche sur grille par validation croisée lors de l'optimisation des hyperparamètres.

## 4- Répertoire mlruns

Ce répertoire nous permet de centraliser les expériences et runs MLFlow à l'utilisation des différentes fonctions du fichier **src/train.py**. Il contient notamment 2 expériences, dont la première, avec l'identifiant *166951002022495853* est l'expérience de référence de notre projet. Au sein de cette expérience, nous avons entraîné un peu moins d'une dizaine de modèles autour d'un jeu de données commun à 192 variables, et chaque run centralise les hyperparamètres des modèles, les modèles eux-mêmes, les jeux de données d'entraînement et de test, les métriques, et les visualisations de features importances, roc_curves et matrices de confusion.

La seconde expérience, avec l'identifiant *453417704798808810* est une expérience de démonstration permettant d'effectuer les traitements dans un temps raisonnable sur l'application.

## 5- Répertoire Notebooks
Dans ce dossier, nous avons 3 notebooks permettant d'offrir des illustrations des traitements et des modélisations effectués sur les jeux de données avec des exemples, ainsi que les résultats de notre étude:

- **Notebook_1_Input_Data.ipynb** : Illustration du fonctionnement du fichier FeatureEngineering.py
- **Notebook_2_Illustrations_traitement_modelisation.ipynb** : Illustration des fonctions du fichier train.py
- **Notebook_3_Resultats.ipynb** : Visualisation des résultats de modélisations réalisés depuis l'interface de l'application en localhost.

## 5- Répertoire pytest

Dans ce répertoire, nous avons différents fichier pythons nous permettant de mettre en œuvre les tests unitaires via la librairie unittest. Par exemple, le fichier test_train.py permet de tester tous les cas d'utilisation des fonctions du fichier train.py, et vérifier la centralisation des différents éléments dans le répertoire mlruns.

## 6- Répertoire data_drift

Ce répertoire contient le tableau html d'analyse du data drift du jeu de données lié à l'expérience de référence du projet (id : *166951002022495853*). Comme nous avons pu le constater, certaines colonnes importantes pour l'interprétation globale de nos modèles ont des distributions différentes entre le jeu de données d'entraînement et de production.

## 7- Répertoire prédictions_tmp

Dans ce répertoire, nous stockons temporairement les prédictions de nos modèles, en attente de centralisation via mlflow tracking dans le but d'effectuer des analyses de datadrift entre les données utilisées en inférence par les modèles, et les données d'entraînement.

## 8- Dashboard

Le dashboard est généré grâce à la librairie streamlit, et contient plusieurs pages. 
Pour cela, nous envoyons des requêtes à un serveur via l'API (cf.partie 9), et nous récupérons les réponses de nos différentes requêtes pour les afficher dans l'interface du dashboard.

- Le fichier **dashboard.py** est la page de démarrage de notre application. Il contient une présentation générale du projet, ainsi que les liens vers les autres pages, toutes contenus dans le répertoire **pages**.

**Dans le répertoire pages, 4 fichiers pythons sont présents**:

- **Data_Analysis.py** est une page dédiée à la visualisation des données initiales. L'utilisateur peut sélectionner un jeu de données, un échantillon est affiché, et l'utilisateur peut afficher la distribution d'une variable en particulier.

- **DataProcessing.py** est une page dédiée au traitement des données, et à la modélisation. L'utilisateur peut créer de nouvelles expériences et de nouveaux modèles directement depuis cette page.

- **Experiences.py** permet de comparer les résultats des différentes expériences et des différentes runs. Nous pouvons sélectionner les modèles, comparer les métriques, afficher les courbes et matrices de confusion [...].

- **Production.py** est notre page d'utilisation des modèles en production. Depuis cette page, nous sélectionnons un jeu de données de production, un modèle, et nous effectuons des prédictions sur les données (individuelles ou de manière groupée). L'interprétation locale de chaque prédiction est également affichée, et nous mettons en place un système de visualisation des données, et des individus. De plus, le suivi des performances des modèles est inclus par la possibilité d'afficher des résultats d'analyse de data drift.

## 9- API

L'API REST est gérée au sein du fichier app.py, grâce à la librairie FastAPI. Chaque Endpoint correspond à une fonctionnalité du dashboard. A la réception des différentes requêtes, le serveur renvoi au client la réponse correpsondante. 

Ce fichier contient de nombreux Endpoint, pour chaque page de l'application dashboard. 
Par exemple, l’Endpoint correspondant à l'URI */production/inference* permet d'utiliser un modèle en inférence sur un individu du jeu de données de production. Nous recevons une requête POST avec le nom d'un jeu de données, l'index d'un individu, et le nom d'un modèle. 

Le serveur s'occupe ensuite de charger le jeu de données, l'individu, et le modèle, ainsi que le meilleur seuil de décision du modèle. Nous utilisons le modèle pour prédire la probabilité de défaut de paiement de notre individu, et nous renvoyons la prédiction, et la probabilité au client au format JSON. 

**Younes EL RHAZALI**
**Etudiant OpenClassrooms - Projet 7 - Implémentez un modèle de scoring**
