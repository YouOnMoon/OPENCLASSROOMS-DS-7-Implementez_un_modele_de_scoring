'''Cette page nous a plusieurs utilités dans notre cas.

Dans un premmier temps, elle nous permet de mettre en place des jeux de données en utilisant le kernel Kaggle pour le feature Engineering, et 
de pré-processer nos jeux de données via la fonction import_data du fichier ../src/train.py, de manière à obtenir 3 jeux de données différents 
train, production et target, dans lesquelles un individu correspond à un crédit.

Puis, nous visualisons ces jeux de données, et nous pouvons lancer l'entraînement de nos modèles et optimisant les hyperparamètres.

ATTENTION : Cette page est notament utile lorsque l'on a à disposition une machine disposant des ressources computationnelles pour effectuer des
entraînements. Si ce n'est pas le cas, les autres pages permettent de visualiser les résultats des précédents entraînements, et d'utiliser les 
modèles en inférence.'''

#Importation des librairies utiles

#Streamlit
import streamlit as st

#Manipulation des données
import pandas as pd
import numpy as np
import json

#Gestion des variables en mémoire
import gc

#Requests
import requests

#Test introduction de la page (explication des fonctionnalités)'''
st.title('**Application des Preprocessings à nos jeux de données**')

st.write("Cette page a deux fonctions au sein de notre application.")
st.write("La première est de créer de nouvelles expériences via MLFlow, en générant des pré-processings et Feature Engineering customisables par l'utilisateur.")

st.write(' ')
st.write("Cette option est disponible dans le premier onglet.")
st.write("Sur la base du kernel Kaggle, nous utilisons les jeux de données initiaux visibles à la page précédente pour générer un jeu de données d'entraînement, un jeu de données de production pour lesquels nous n'avons pas d'étiquette, et un jeu de données contenant les targets correspondantes au jeu de données d'entraînement.")
st.write("Cela est possible grâce à l'utilisation de techniques d'algébre relationnel en mergeant les différents jeux de données sur des colonnes communes, de manière à ce que chaque individu corresponde à une demande de crédit.")
st.write("Les features trop corrélés à la target sont automatiquement supprimés, et l'utilisateur peut définir plusieurs paramètres supplémentaires.")

#Le premier expander nous sert à envoyer des requêtes au serveur de manière à ce que celui-ci process les jeux de données, suivant des instructions
#fournis par l'utilisateur quant aux valeurs manquantes, les corrélations des colonnes ...
#Cela entraînera la création d'une expérience mlflow qui regroupera les modèles entraînés sur ces mêmes jeux de données.'''

st.warning("**ATTENTION**: La création d'une nouvelle expérience va appliquer différents traitements à nos données initiales. En fonction des ressources computationnelles à disposition, cette opération peut prendre plusieurs minutes à plusieurs heures.")
with st.expander("**Création d'une nouvelle expérience!**"):
    with st.container():
        st.subheader("**Informations sur les expériences précédentes:**")

        #Requête pour recevoir les informations sur les précédentes expérience - deux expériences ne peuvent pas avoir le même nom
        nb_exp_request = requests.get(url='http://ocds7ey.herokuapp.com/experiment_start')
        nb_exp = nb_exp_request.json()['Count']
        names = nb_exp_request.json()['Names']
        st.write("Nombre d'expériences précédement crées : **{}**.".format(nb_exp))
        st.write("Liste des noms d'expériences enregistrées : **{}**".format(names))
    st.subheader("Veuillez remplir les diférents champs ci-dessous:")

    #Choix du nom de l'expérience par l'utilisateur
    new_name = st.text_input('**Nom de la nouvelle expérience:**')

    #Getsion d'erreur en cas d'expériences ayant déjà le nom entré par l'utilisateur
    if new_name in names:
        st.error('**ATTENTION**: Ce nom est déjà attribué à une autre expérience, veuillez entrer un nouveau nom.')
    
    #Choix des filtres appliqués au jeu de données (nan, correlation ...)    
    nan_ratio = st.number_input('**Suppression des features ayant un taux de valeurs manquantes supérieur à:**', min_value = 0.0, max_value=1.0, step = 0.01)
    strategy = st.selectbox('**Stratégie de remplissage des valeurs manquantes:**', ['mean', 'median', 'min', 'max', 'None'])
    correlation_thresh = st.number_input('**Sélection des features ayant un coefficient de Pearson (valeur absolue) supérieur à**:', min_value = 0.0, max_value=1.0, step = 0.01)
    
    #Description de l'expérience fournie par l'utilisateur -- logguée en tag au sein de l'expérience
    description = st.text_area("**Description (facultative) de l'éxperience:**")

    #Lancement de la requête de création de l'expérience au serveur grâce au boutton, avec les choix de l'utilisateur comme inputs pour la fonction
    #import_data de ../src/train.py
    if st.button('Nouvelle Expérience!'):
        with st.spinner("Expérience en cours de création. En attente du serveur ..."):      
            exp_creation_inputs = {'name' : new_name,'na_values' : nan_ratio, 'na_strategy' : strategy, 'pearson_select' : correlation_thresh,'description' : description}                       
            exp_creation_request = requests.post(url = 'http://ocds7ey.herokuapp.com/experiment_start/create', data = json.dumps(exp_creation_inputs))
            exp_creation_response = exp_creation_request.json()
            exp_id = exp_creation_response['id']
            exp_name = exp_creation_response['name']

            #Affichage de l'id et le nom de l'expérience en cas de succès
            st.success('Nouvelle expérience crée - ID : {} - Name : {}'.format(exp_id, exp_name))
            st.snow()                               

#Mise en cache du résultat de la requête d'affichage d'un jeu de données sélectionné par l'utilisateur.
#L'utilisateur choisis une expérience, un des 3 jeux de données entraînement, production ou targets, et un échantillon du dataframe est affiché.'''
@st.cache(suppress_st_warning=True)
def df_request(exp):
    #input de la requête d'affichage des jeux de données dans le sélecteur - nom de l'expérience nécessaire
    experiment_selected_data = {'name' : exp,'na_values' : 0.0, 'na_strategy' : 'None', 'pearson_select' : 0.0,'description' : 'description'}
    exp_selection_request = requests.post(url = 'http://ocds7ey.herokuapp.com/experiment_start/select', data = json.dumps(experiment_selected_data))
    exp_selection_response = exp_selection_request.json()

    #recéption des jeux de données enchantillonnés et mise sous forme de dataframe
    train = pd.DataFrame(exp_selection_response['train'])
    test = pd.DataFrame(exp_selection_response['test'])
    target = pd.DataFrame(exp_selection_response['target'])
    del experiment_selected_data, exp_selection_request
    gc.collect()
    return train, test, target

#Requête permettant d'afficher la distribution d'une variable d'un jeu de données en particulier. 
#Le serveur crée un plot, mis sous forme d'image, et nous recevons les 4 channels de cette image (RGBA), que nous reconstituons ici'''
@st.cache(suppress_st_warning=True)
def plot_request(df, col, exp):
    #requête avec nom de l'expérience, le jeu de données demandé, et la colonne demandée par le client
    column_select = {'name' : str(df), 'column' : str(col), 'exp_name' : str(exp)}
    h1 = requests.post('http://ocds7ey.herokuapp.com/experiment_start/select/to_figure', data = json.dumps(column_select))

    #reception des signaux et reconstitution de l'image
    full_channels = h1.json()
    red = pd.DataFrame(full_channels['R']).values
    green = pd.DataFrame(full_channels['G']).values
    blue = pd.DataFrame(full_channels['B']).values
    last = pd.DataFrame(full_channels['A']).values
    rgb = np.dstack((red,green,blue, last))
    del column_select, h1, red, green, blue, last 
    gc.collect()
    return rgb


st.write("L'onglet suivant nous permet de visualiser les données générées pour nos différentes expériences.")
st.write("Un premier sélecteur nous permet de choisir une expérience, et un second nous permet de sélectionner un jeu de données.")
st.write("Un échantillon du jeu de données est alors affiché, et il est ensuite possible de sélectionner une colonne pour laquelle nous visualisons la distribution.")
with st.expander("**Affichage des jeux de données crées!**"):
    #Séléction de l'éxperience, requpete retournant la liste des epxériences disponilbes
    nb_exp_request2 = requests.get(url='http://ocds7ey.herokuapp.com/experiment_start')
    names2 = nb_exp_request.json()['Names'] 
    exp = st.selectbox('**Veuillez Sélectionner une éxperience :**', names2)
    train, test, target = df_request(exp)

    #Selection du jeu de données à afficher et affichage d'un sample
    expander_options = ['train', 'test', 'target']
    option =  st.selectbox('Veuillez sélectionner un fichier', expander_options)
    st.markdown('**Affichage du jeu de données sélectionné!**')
    st.markdown("Nom de l'expérience : **{}**".format(exp))
    st.dataframe(locals()[option])

    #Sélection de la colonne à visualiser et affichage de l'image
    column_display = st.selectbox("**Affichage de la colonne pour laquelle afficher la distribution**", locals()[option].columns)
    rgb = plot_request(str(option), str(column_display), str(exp))
    st.image(rgb, use_column_width=True)
    del rgb
    gc.collect()


#Définition des fonctions mises en caches pour la partie entraînement des modèles depuis l'application.'''

#Deamnde au serveu d'envoyer la liste des hyperparamètres d'un type de classifieur après sélection du modèle
#par l'utilisateur.'''
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def model_request(model_type):
    model_type_input = {'type' : str(model_type)}
    model_type_request = requests.post(url='http://ocds7ey.herokuapp.com/experiment_start/select/runs/model_selection', data = json.dumps(model_type_input))
    model_type_answer = model_type_request.json()
    return model_type_answer

#Demande au serveur d'envoyer la grille d'hyperparamètres de type integer à optimiser par l'utilisateur.
#Il est nécessaire de les passer en type float pour la communication avec le serveur, et sont repassés en type integer
#côté serveur.'''
@st.cache(suppress_st_warning=True)
def int_grid(mini, maxi, step, is_exp):
    int_grid_input = {'mini' : float(mini), 'maxi': float(maxi), 'step' : float(step), 'is_exp' : is_exp}
    input_grid_request = requests.post(url = 'http://ocds7ey.herokuapp.com/experiment_start/select/runs/model_selection/int_params', data = json.dumps(int_grid_input))
    input_grid_response = list(np.array(input_grid_request.json()['grid']).astype(int).astype(float))
    return input_grid_response

#Demande au serveur d'envoyer la grille d'hyperparamètres de type float à optimiser par l'utilisateur. Cela se fait
#à partir des valeurs maximales et minimales, ainsi que le step de la grille.'''
@st.cache(suppress_st_warning=True)
def float_grid(mini, maxi, step, is_exp):
    float_grid_input = {'mini' : mini, 'maxi': maxi, 'step' : step, 'is_exp' : is_exp}
    float_grid_request = requests.post(url = 'http://ocds7ey.herokuapp.com/experiment_start/select/runs/model_selection/float_params', data = json.dumps(float_grid_input))
    float_grid_response = float_grid_request.json()['grid']
    return float_grid_response

#Fonction permettant de concatener plusieurs dictionnaires de manière à crée un seul dictionnaire à partir 
#des différentes grilles d'hyperparamètres demandés par l'utilisateur.'''
@st.cache(suppress_st_warning=True)
def dict_merge(dict_1, dict_2):
    double_dict_input = {'dict_1' : dict_1, 'dict_2' : dict_2}
    double_dicts_merge_request = requests.post(url='http://ocds7ey.herokuapp.com/experiment_start/select/runs/model_selection/merge_params', data = json.dumps(double_dict_input))
    double_dicts_merge_response = double_dicts_merge_request.json()
    return double_dicts_merge_response

#Foncton permettant de demander au serveur de lancer un entraînement paramétré par l'utilisateur avec optimisation
#des hyperparamètres par recherche sur grille avec validation croisée. Les différents pré-processeurs sont également 
#paramètrables par l'utilisateur.'''
@st.cache(suppress_st_warning=True)
def hyper_param_tuning(train, target, halving_state, json_export, standard_state, pca_state, params, grid_params, model_type, run_name, exp_name):
    tuning_inputs = {'train' : train, 'target' : target, 'halving_state' : halving_state,'json_export' : True, 'standard_state' : standard_state, 'pca_state' : pca_state, 'params' : params, 'grid_params' : grid_params, 'model_type' : model_type, 'run_name' : run_name, 'exp_name' : exp_name}
    tuning_request = requests.post(url='http://ocds7ey.herokuapp.com/experiment_start/select/runs/model_selection/hyper_param_tuning', data = json.dumps(tuning_inputs))
    tuning_response = tuning_request.json()
    return tuning_response

#Dans cette  partie, au sein de l'expérience sélectionnée par l'utilisateur, nous pouvons entraîner nos modèles en enoyant des requêtes au serveur,
#ce qui entraînera la création d'une nouvelle run mlflow.
#L'utilisateur choisis le type de modèle, les hyperparamètres, et les preprocesseur (avec ou sans standardisation) ainsi que la manière de gérer 
#le déséquilibre des classes de la target (SMOTE ou class_weight balanced)'''
st.write(" ")
st.write("Dans l'onglet suivant, nous pouvons créer de nouvelles runs au sein de notre expérience, ce qui entraînera l'entraînement d'un modèle à partir des données précédemment générées.")
st.write("Le modèle est entièrement customisable dans cet onglet. Nous pouvons choisir le type de classifieur, entrer des hyperparamètres [...].")
st.write("En particulier, nous pouvons entrer les grilles d'hyperparamètres à optimiser par validation croisée, et les préprocesseurs disponibles.")
with st.expander("**Création d'un nouveau modèle!**"):
    st.info("**ATTENTION**: Pour des raisons de ressources computationnelles sur cette version côté serveur, il n'est possible d'entraîner que des régressions logistiques avec la méthode Class_Weight, sans PCA. Pour accéder à toutes les fonctionnnalités, merci d'utiliser la version complète de l'application.")
    #Nom de la run à considérer
    runs_select_input = {'name' : 'any', 'column' : 'any', 'exp_name' : str(exp)}
    runs_list_request = requests.post(url='http://ocds7ey.herokuapp.com/experiment_start/select/runs', data = json.dumps(runs_select_input))
    runs_list_response = runs_list_request.json()

    #Vérification des anciennes run de l'expérience avec erreur si la nouvelle run port le nom d'une ancienne run
    previous_runs_names = runs_list_response.keys()
    st.write("L'expérience **{}** contient {} runs, dont voici la liste:".format(exp, len(previous_runs_names)))
    st.write(list(previous_runs_names))
    st.subheader("Nom et type de modèle:")
    new_run_name = st.text_input('**Veuillez entrer le nom de la nouvelle run:**')
    if new_run_name in list(previous_runs_names):
        st.error("Ce nom est déjà attribué à une run précédente, veuillez en entrer un nouveau!")
    
    #Choix du type de classifieur - Logistic regression, xgboost, histgradientboosting classifier ou LGBMClassifier    
    model_type = st.selectbox("**Veuillez sélectionner le type de modèle:**", ['HistGradientBooster', 'LightGBM', 'LogisticRegression'])

    #Veriosn light de l'application, model_type == LogReg quoi qu'il en soit 
    model_type = 'LogisticRegression'
    
    #Liste des hyperparamètres du modèle sélectionné
    model_init_params = model_request(model_type)
    st.write("**Liste des paramètres pré-définis du modèle:**")
    st.write(model_init_params)
    st.subheader("Hyperparamètres entrés manuellement:")

    '''Par la suite, l'utilisateur peut choisir d'entrer des hyperparamètres qu'il aura défini lui même, sans optimisation.'''
    st.write("**Veuillez sélectionner les paramètres que vous souhaitez entrer manuellement:**")
    if 'objective' in model_init_params.keys():
        del model_init_params['objective']
    
    #multiséléction des hyperparamètres manuels et séparation par type de variable associé
    manual_params = st.multiselect("**Veuillez sélectionner les paramètres que vous souhaitez entrer manuellement:**", model_init_params.keys())
    st.write(manual_params)
    floats_params = []
    int_params = []
    str_params = []
    for i in manual_params:
        if type(model_init_params[i]) == float:
            floats_params.append(i)
        elif type(model_init_params[i]) == str:
            str_params.append(i)
        elif type(model_init_params[i]) == int:
            int_params.append(i)
        else:
            pass

    #Affichage des hyperparamètres par type
    st.write("**Paramètres de type float:**")
    for i in floats_params:
        tmp = i
        locals()[tmp] = st.number_input('Paramètre {}'.format(i), value=model_init_params[i], step=0.01)
    st.write("**Paramètres de type integer:**")
    for i in int_params:
        tmp = i
        locals()[tmp] = st.number_input('Paramètre {}'.format(i), value=model_init_params[i], step=1)
    st.write("**Paramètres de type string:**")
    for i in str_params:
        tmp = i
        locals()[tmp] = st.text_input('Paramètre {}'.format(i), value=model_init_params[i])
    dict_manual_params = {}
    for i in floats_params:
        dict_manual_params[i]=[locals()[i]]
    for i in int_params:
        dict_manual_params[i]=[locals()[i]]
    for i in str_params:
        dict_manual_params[i]=[locals()[i]]
    st.write("Nouveaux paramètres entrés manuellement:")
    st.write(dict_manual_params)
    
    #Merge des dictionnaires d'hyperparamètres manuels et de base 
    double_dict_input = {'dict_1' : model_init_params, 'dict_2' : dict_manual_params}
    double_dict_request = requests.post(url = 'http://ocds7ey.herokuapp.com/experiment_start/select/runs/model_selection/manual_params', data = json.dumps(double_dict_input))
    tuning_inputs = double_dict_request.json()
    
    #Ici, l'utilisateur peut choisir les hyperparamètres à optiiser par validation croisée. Un multisélécteur lui permet de choisir les hyperparamètres, 
    #et un pour chaque hyperparamètre, il pourra séléctionner un nombre de valeurs, la minimale et la maximale, pour chaque type de variable.'''
    st.subheader("Optimisation des hyperparamètres:")
    tuning_params = st.multiselect("**Veuillez sélectionner les paramètres pour une recherche sur grille par validation croisée:**", tuning_inputs.keys())
    grid_floats_params = []
    grid_int_params = []
    grid_str_params = []
    dict_grid_params = {}

    #Séparation du traitement et des requêtes en fonction des types de variables (cf. fonction int_grid_param, float_grid_params)
    for i in tuning_params:
        if type(model_init_params[i]) == float:
            st.markdown("**Valeurs pour l'hyperparamètre {}**".format(i))
            grid_floats_params.append(i)
            tmp1 = i + '_min'
            tmp2 = i + '_max'
            tmp3 = i + '_n_values'
            tmp4 = i + '_is_exp'
            locals()[tmp1] = st.number_input('Valeur minimale de {}'.format(i), value=model_init_params[i], step=0.1)
            locals()[tmp2] = st.number_input('Valeur maximale de {}'.format(i), value=model_init_params[i], step=0.1)
            locals()[tmp3] = st.number_input('Nombre de valeurs de {}'.format(i), value=5.0, step=1.0, min_value=2.0)
            locals()[tmp4] = st.checkbox("Exponential Scale pour {}".format(i))
            dict_grid_params[i] = float_grid(locals()[tmp1], locals()[tmp2], locals()[tmp3], locals()[tmp4])
        elif type(model_init_params[i]) == str:
            st.markdown("**Valeurs pour l'hyper-paramètre {}**".format(i))
            grid_str_params.append(i)
            num_values = st.slider("Veuillez sélectionner le nombre de valeurs possibles de la variable",2,10)
            for j in range(num_values):
                tmp = 'value_' + str(j)
                locals()[tmp] = st.text_input('Veuillez entrer la valeur {} de la variable.'.format(j))
        elif type(model_init_params[i]) == int:
            st.markdown("**Valeurs pour l'hyper-paramètre {}**".format(i))
            grid_int_params.append(i)
            tmp1 = i + '_min'
            tmp2 = i + '_max'
            tmp3 = i + '_n_values'
            tmp4 = i + '_is_exp'
            locals()[tmp1] = st.number_input('Valeur minimale de {}'.format(i), value=model_init_params[i], step=1)
            locals()[tmp2] = st.number_input('Valeur maximale de {}'.format(i), value=model_init_params[i], step=1)
            locals()[tmp3] = st.number_input('Nombre de valeurs de {}'.format(i), value=5, step=1, min_value=2)
            locals()[tmp4] = st.checkbox("Exponential Scale pour {}".format(i))
            dict_grid_params[i] = int_grid(int(locals()[tmp1]), int(locals()[tmp2]), int(locals()[tmp3]), bool(locals()[tmp4]))
        else:
            pass
    st.write("**Récapituliatif des hyper-paramètres à optimiser par validation croisée:**")
    st.write(dict_grid_params)

    #merge des dictionnaires de grilles d'hyperparamètres
    model_init_params = dict_merge(model_init_params, dict_manual_params)

    #Choix de la stratégie de gestion du déséquilibre et des préprocesseurs
    imba_strat = st.radio("**Gestion du déséquilibre des classes cibles:**", ('SMOTE', 'Class_Weight'), horizontal=True)
    if imba_strat == 'SMOTE':
        imba_state = False
    else:
        imba_state = True
    standard_scale = st.radio("**Utilisation du Standard Scaler pour avoir des features centrées réduites dans notre jeu de données:**", ('Oui', 'Non'), horizontal=True)
    if standard_scale == 'Oui':
        standard_state = True
    else:
        standard_state = False
        st.warning("**ATTENTION:** Il est fortement recommandé d'utiliser un Standard Scaler, en particulier pour le cas où l'on souhaite effectuer une réduction dimensionnelle par PCA!")
    use_pca = st.radio("**Réduction dimensionnelle par PCA avec conservation de 95% de la variance expliquée du jeu de données:**", ('Oui', 'Non'), horizontal=True)
    if use_pca == 'Oui':
        pca_state = True
    else:
        pca_state = False
    
    #Utilisation d'un boutton pour envoyer la demande d'entraînement au serveur. Les jeux de données associés à l'expérience seront splité en train, test,
    #les préprocesseurs appliqués, la recherche sur grille effectuée, et le modèle entaîné.
    #MLFlow se charge ensuite de logguer le modèle, les jeux de données, et les métriques pour comparer les résultats à la page Experiences'''
    if st.button("Entraînement nouveau modèle!"):
        if len(dict_grid_params.keys()) == 0:
            st.error('**Veuillez choisir des hyper-paramètres à optimiser!**')
        elif exp == 'PROJECT_7_OC_DS_EXPERIENCE_REFERENCE':
            st.error("**Vous avez sélectionné l'expérience de référence du projet. Auncun modèle ne peut être entraîné sur cette expérience, car elle contient tous nos résultats de modélisation. Pour entraîner un nouveau modèle, merci de sélectionner une nouvelle expérience dans le deuxième onglet.**")
        else:
            with st.spinner("Run en cours! En attente du serveur ..."):  
                train_dict = train.to_dict()
                target_dict = target.to_dict()

                #Modification des types de préproecsseurs pour des raisons de ressources computationnelles limitées sur cette version
                imba_state = True
                pca_state = True
                pca_state = False
                model_type = 'logreg'
                response = hyper_param_tuning(train_dict, target_dict, imba_state, True, standard_state, pca_state, model_init_params, dict_grid_params, model_type, new_run_name, exp)
                st.write(response)
                st.success('Modèle entraîné avec succès. Nom de la run : {}'.format(new_run_name))
                st.snow()
            