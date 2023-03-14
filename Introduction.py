'''Ce fichier nous permet d'afficher la page d'introduction de notre application.
Etant donnée les nombreuses fonctionnalités de notre applications, nous avons séparé les différentes fonctions sur plusieurs
pages, directement accessibles via streamlit.

Cette page est la page de gare, présentant le projet, et nous offrant les liens vers les autres pages de l'application.'''

#Importation de streamlit
import streamlit as st

#Configuration des pages de notre application - layout large et sidebar étirée
st.set_page_config(layout="wide", initial_sidebar_state = 'expanded')

#Titre et textes explicatifs 
st.title('**OpenClassrooms - Parcours Data Scientist - Projet 7**')
st.sidebar.markdown("# Getting Started !")
st.sidebar.markdown('Bienvenue sur cette application!')
st.sidebar.markdown('Pour accéder aux différentes fonctionnalitées, **veuillez sélectionner la page associée ci-dessus!**')
st.subheader("**Implémentez un modèle de scoring**")

st.info("**ATTENTION** : De manière à pouvoir déployer notre application tout en limitant les ressources computationnelles nécessaires, il s'agit ici de la version 'light' de l'application, bridée en utilisant des jeux de données échantillonnés, et avec certaines fonctionnalités non disponibles.")

st.markdown('**Bienvenue sur la page principale du projet 7 du Parcours Data Scientist OpenClassrooms!**')
st.write(' ')
st.write('Pour ce projet, nous allons mettre en place un modèle permettant de définir si une demande de crédit doit être accordéee ou non au regard des différents features associées à une demande en particulier.')
st.write('Pour ce faire, nous partons de différents jeux de données importés sous la forme de fichiers au format .csv.')
st.write("Sur cette base, nous pouvons efféctuer différentes opérations de feature engineering de manière à obtenir un jeu de données dans lequel chaque observation correpsond à une demande de crédit, et les targets correspondent au fait qu'un crédit ai été remboursé avec succès ou non.")
st.write("Dans notre cas, l'intérface nous permet de configurer les preprocessings appliqués à nos jeux de données, nous permettant de mettre en place différents jeux de données, dans le cadre de chaque expérience.")
st.write("Enfin ,pour chaque expérience, nous utilisons des algorithmes de machine learning pour lesquels nous récupérons des métriques de manière à pouvoir **comparer leurs performances**, et décider d'envoyer, ou non un modèle en particulier en production.")
st.write("Les différentes fonctionnalitées de notre application sont définies ci-dessous:")
st.write(' ')

#Accès au tableau de bord de production
st.write("**Pour accéder au tableau de bord de prédiction, veuillez cliquer sur le liens suivant, ou sélectionner l'option Dashboard dans la barre latérale de naviguation.**")
st.subheader('[Tableau de bord!](Dashboard)')
st.write(' ')
st.write("**Pour accéder aux options plus avancées de notre application, les liens ci-dessous sont disponibles.**")

#Colonnes contenant les liens vers les autres pages
col1, col2, col3, col4 = st.columns(4, gap = 'large')
with col1:
    st.subheader('[Visualisation des données initiales](Data_Analysis)')
    st.write("Page de visualisation des données initiales pour notre étude.")
with col2:
    st.subheader('[Pre-Processing des données](DataProcessing)')
    st.write("Applicaton des pré-processings à nos jeux de données et entraînement des modèles.")
with col3:
    st.subheader('[Gestion des modèles](Experiences)')
    st.write("Comparaison des perfrormances des modèles en utilisant différentes métriques.")
with col4:
    st.subheader('[Interprétation des résultats](Poduction)')
    st.write("Utilisation des modèles et explication des résultats.")
st.write(' ')
st.write(' ')
st.write(' ')
st.caption("_**Younes EL RHAZALI**_")
st.caption("_Etudiant OpenClassrooms - Parcours Data Scientist_")
