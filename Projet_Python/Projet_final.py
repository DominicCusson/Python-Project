#!/usr/bin/env python
# coding: utf-8

# In[214]:


#Importer os
import os

#Importer la librairie Pandas pour la visualisation
import pandas as pd

#Importer Numpy pour la manipulation de données
import numpy as np

#Importer SkLearn
import sklearn

#Importer Pyplot pour la visualisation de données
import matplotlib
import matplotlib.pyplot as plt


# In[170]:


#Créer la classe "ReadFile" pour ouvrir le fichier de données
class ReadFile():
    
    def __init__(self, file):
        self.file = file
        
        #Si le fichier est un .csv, aller à la fonction "read_csv()"
        if file.endswith("csv"):
            print("csv file")
            self.read_csv()
            
        #Si le fichier est un .xlsx, aller à la fonction "read_excel()"
        elif file.endswith("xlsx"):
            print("xlsx file")
            self.read_excel()
           
        #Si le fichier est un .txt, aller à la fonction "read_csv()"
        elif file.endswith("txt"):
            print("txt file")
            self.read_csv()

    def read_csv(self):
        global dataframe
        dataframe = pd.read_csv(self.file)
        return print("Dataframe ready: use \"dataframe\" to view")

    def read_excel(self):
        global dataframe
        dataframe = pd.read_excel(self.file, engine = "openpyxl")
        return print("Dataframe ready: use \"dataframe\" to view")

    #Vérifier si le fichier existe
    def verif_presence(self):
        return os.path.exists(self.file)


# In[171]:


#Définir le fichier de données à ouvrir
#Nom de fichier original : "FBI-Hate-Crime-Statistics", sur GitHub
#https://github.com/emorisse/FBI-Hate-Crime-Statistics/blob/master/2013/table14.csv

#Trouver le dossier et le fichier 
from os import sys
fichier = os.path.join(sys.path[0], "Project Data.txt")

#Afficher le chemin du fichier
print(fichier)

#Appeler la fonction "ReadFile" pour lire le fichier
df_import = ReadFile(fichier)

#Vérifier si le fichier a été trouvé
df_import.verif_presence()


# In[172]:


#Visualiser la base de données
dataframe


# In[173]:


#Créer une copie indépendante "df" de la base de données
df = dataframe.copy(deep = True)


# In[174]:


#Vérifier la copie "df"
df


# In[175]:


#Éliminer les valeurs manquantes "NaN" dans la base de donées et créer une nouvelle base de données "df1"
df1 = df.dropna()
df1


# In[176]:


#Réattribuer les index à la base de données dans une nouvelle base de données "df2"
df2 = df1.reset_index()
df2


# In[177]:


#Créer une liste de population avec des nombres séparés par un point

#Créer une liste qui contient toute l'information de la colonne "Population" de la base de données "df1"
list_population = list(df1["Population"])

#Créer une nouvelle liste vide pour les nombres changés avec un séparateur par point
list_population_changed = []

#Boucle qui prend tous les nombres un à un de la liste "list_population" et les transforme avec aucun séparateur avant
#de les ajouter dans la liste "list_population_changed"
for i in range(0,len(list_population)):
    a = list_population[i].replace(",","")
    b = int(a)
    list_population_changed.append(b)

#Vérification par affichage de la liste "list_population_changed"
len(list_population_changed)


# In[178]:


#Créer une mini-base de données "df_new_population" avec les données de la liste "list_population_changed"
df_new_population = pd.DataFrame(list_population_changed, columns = ["Population_new"])

#Vérification de la mini-base de données
df_new_population


# In[179]:


#Créer une nouvelle base de données "df_Concat" mise à jour avec l'ajout de la mini-base de données

#Créer la nouvelle base de données mise à jour "df_Concat"
df_Concat = pd.concat([df2, df_new_population], axis = 1)

#Vérification de la base de données mise à jour "df_Concat"
df_Concat


# In[180]:


#Modifier la colonne "Gender" pour que les valeurs soient des entiers
Gender_new = df_Concat.Gender.astype(int)

#Créer une mini-base de données "df_new_gender" avec les données de "Gender_new"
df_new_gender = pd.DataFrame(Gender_new)

#Créer une nouvelle mini-base de données "df_new_gender_modified" où la colonne est renommée
df_new_gender_modified = df_new_gender.rename(columns={'Gender': 'Gender_new'}) 

#Vérification de la mini-base de données
df_new_gender_modified


# In[181]:


#Créer une nouvelle base de données "df_Concat_2" mise à jour avec l'ajout de la colonne "Gender" modifiée
df_Concat_2 = pd.concat([df_Concat, df_new_gender_modified], axis = 1)
df_Concat_2


# In[182]:


#Appeler une fonction externe

#Importer tout de la fonction "Verification"
from Verification import *

#Appel de la fonction "Verification"
Verification(df_Concat)


# In[183]:


#Créer une liste avec tous les éléments de la colonne "Population_new"
list_population_new = df_Concat_2.Population_new.tolist()

#Organiser la nouvelle liste en ordre croissant
list_population_new_sorted = sorted(list_population_new)

#Afficher la liste organisée en ordre croissant
list_population_new_sorted


# In[184]:


#Créer un algorithme automatisé de recherche qui détermine si une valeur de population se trouve dans la base de données, ou
#sinon, qui détermine les valeurs de population les plus proches de la valeur entrée
def PopulationSearch (table, value):
    
    #Définir le résultat initial de la recherche
    resultat = 0
    
    #Trouver dans tout le tableau la valeur exacte qui correspond à la valeur entrée
    for i in range (0, len(table)):
        if table[i] == value:
            
            #Définir le résultat comme étant final
            resultat = 1
            
            #Retourner la réponse finale
            return ("Cette valeur est dans la base de données : il y a un type d'agence avec cette population exacte")
    
    #Si le résultat à l'étape précédente n'est pas obtenu, continuer avec cette section
    if resultat != 1:
        
        #Retourner la première réponse
        print("Cette valeur n'est pas dans la base de données : il n'y a aucun type d'agence avec cette population exacte")
        
        #Trouver dans tout le tableau la première valeur qui dépasse la valeur entrée
        for i in range (0, len(table)):
            if value < table[i]:
                
                #Définir la valeur immédiatement supérieure à la valeur entrée
                valeur_superieure = table[i]
                
                #Terminer cette boucle dès la première itération
                break
        
        #Définir le nouvel index de recherche pour le tableau, soit celui précédant l'index de la valeur supérieure à la
        #valeur entrée
        new_index = (table.index(valeur_superieure) - 1)
        
        #Définir la valeur immédiatement inférieure à la valeur entrée
        valeur_inferieure = table[new_index]
        
    #Retourner la réponse complémentaire finale
    return "Les valeurs de population les plus proches sont {0} et {1}".format(valeur_inferieure, valeur_superieure)
      


#Données de recherche
#Tableau des valeurs de population de la base de données
tab = list_population_new_sorted

#Valeur de population dont la présence est à vérifier dans la base de données
pop_value = 15000
 
#Appel de la fonction et affichage du résultat
result = PopulationSearch(tab, pop_value)
print(result)


# In[185]:


#Préparer les données à être analysées par apprentissage automatique (AA)

#Importer le module "StandardScaler" de "sklearn"
from sklearn.preprocessing import StandardScaler


#Définir les données qui seront utilisée pour l'AA

#Données de prédiction, soit les données des quatre quartiles de l'année 2013
features = np.asarray(df_Concat_2[["1st quarter", "2nd quarter", "3rd quarter", "4th quarter"]])

#Données à prédire (cible), soit les données de genre
target = np.asarray(df_Concat_2["Agency type"])

#Observer les données de prédiction originales
print(features)


#Transformer les données de prédiction

#Instancier la variable de mise à l'échelle
scaler = StandardScaler()

#Fit les données de prédiction à l'opérateur de mise à l'échelle
scaler.fit(features)

#Instancier la variable des données de prédiction transformées
scaled_features = scaler.transform(features)

#Observer les données de prédiction transformées
print(scaled_features)


# In[186]:


#Apprentissage automatique (AA) non-supervisé : l'analyse en composantes principales (ACP)

#Importer le module "PCA" de "sklearn"
from sklearn.decomposition import PCA

#Définir les paramètres de l'ACP : ici, on réduit les 4 composantes des données de prédiction transformées
#initiales en 2 composantes
pca = PCA(n_components=2)

#"Fit" les données de prédiction transformées à l'ACP
pca.fit(scaled_features)

#Instancier la variable des données de prédiction obtenues suite à la réduction de la dimensionnalité par ACP
features_pca = pca.transform(scaled_features)

#Vérification de la réduction de dimensionnalité
print("original shape:   ", scaled_features.shape)
print("transformed shape:", features_pca.shape)


#Affichage des modification effectuées par l'ACP

#Importer "seaborn"
import seaborn as sns
sns.set()

#Instancier la variable de la transformée inverse des données de prédiction réduites en dimensionnalité
features_new = pca.inverse_transform(features_pca)

#Afficher le graphique des données de prédiction transformées initiales et des de la transformée inverse des données de
#prédiction réduites en dimensionnalité, pour observer la réduction de dimensinnalité
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], alpha=0.2)
plt.scatter(features_new[:, 0], features_new[:, 1], alpha=0.8)
plt.axis('equal');

#Ajouter les colonnes "PCA1" et "PCA2" à la bas de données "df_Concat_2", soit celles des données de prédiction réduites en 
#dimensionnalité, afin de les utiliser pour tracer le graphique de l'utilité de la réduction de dimensionnalité
df_Concat_2['PCA1'] = features_pca[:, 0]
df_Concat_2['PCA2'] = features_pca[:, 1]

#Tracer le graphique de l'utilité de la réduction de dimensionnalité
sns.lmplot(x = "PCA1", y = "PCA2", hue="Agency type", data = df_Concat_2, fit_reg=False)


# In[187]:


#On voit ici que les données des quatre quartiles de l'année 2013, réduites à 2 dimensions, sont moyennement utiles pour
#prédire le type d'agence


# In[188]:


#Validation croisée des données par la méthode "split sample" avec "ShuffleSplit" de "sklearn" sur SVM

#Importation des modules
from sklearn.model_selection import ShuffleSplit
from sklearn import svm

#Données de prédiction transformées  et réduites en dimensionnalité, soit les données des quatre quartiles de l'année 2013
x = features_pca

#Données à prédire (cible), soit les données de genre
y = target


#Validation croisée de l'apprentissage machine en utilisant 5 essais d'entraînement consécutifs différents

#Méthode "Shuffle split" utilisant 60% des données pour l'entraînement et 40% des données pour le test : les données
#d'entraînement se divisent en "X_train", soit une partie des données de prédiction et "y_train", soit une partie des données
#à prédire, alors que les données pour le test sont "X_test" et "y_test", soit le reste des données qui est utilisé afin
#de valider l'apprentissage qui s'est fait durant la période d'entraînement
rs = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)

#Apprentissage machine par machine de support vectoriel (support vector machine, SVM) avec noyeau linéaire (classificateur)
classifier = svm.SVC(kernel='linear', C=1)
model = classifier.fit(x, y)

#Importer le module "cross_val_score" pour calculer le score de précision du classificateur
from sklearn.model_selection import cross_val_score

#Instancier la variable donnant les scores de prédiction du classificateur, soit le score à chaque essai d'entraînement
scores = cross_val_score(model, x, y)
scores

#Afficher le score de prédiction moyen du classificateur, avec écart-type
print("{0} accuracy with a standard deviation of {1}.".format(scores.mean(), scores.std()))


# In[189]:


#Afficher le graphique du fonctionnement du classificateur

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(rs.split(x, y)):
    classifier.fit(x[train], y[train])
    viz = sklearn.metrics.RocCurveDisplay.from_estimator(classifier, x[test], y[test],
                         name='ROC essai {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'ROC moyen (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Caractéristique de fonctionnement du classificateur")
ax.legend(loc="lower right")
plt.show()


# In[190]:


#Utilisation de la base de données SQLite3

#Importer SQLite3 et le module de connection
import sqlite3
from sqlite3 import connect

#Créer un espace de données "data.db" où se connecter
connection = sqlite3.connect("data.db")

#Vérifier que la base de données "data_table_1" n'existe pas déjà
connection.execute("DROP TABLE data_table_1")

#Instancier la variable de la base de données à créer
tableau = 'data_table_1'

#Créer la base de données "data_table_1"
connection.execute('''CREATE TABLE IF NOT EXISTS {0} ('state', 'agency_name', 'population')'''.format(tableau))

#Enregistrer les modifications
connection.commit()

#Fermer l'espace de données
connection.close()


# In[191]:


#Se connecter à l'espace de données "data.db"
connection = sqlite3.connect("data.db")

#Insérer des colonnes et des valeurs dans la base de données 
connection.execute('''INSERT INTO data_table_1 ('state', 'agency_name', 'population') VALUES ('Alabama', 'Florence', 39481),
                                                                                       ('Alaska', 'Anchorage', '299455')''')
#Enregistrer les modifications
connection.commit()

#Affifcher la base de données 
df_sql = pd.read_sql_query("SELECT * from data_table_1", connection)
df_sql


# In[192]:


#Récupérer les éléments de la colonne "state" de la base de données "data_table_1"
selection = connection.execute('''SELECT state FROM data_table_1''').fetchall()
selection


# In[193]:


#Utiliser les valeurs de la table SQLite3

#Transformer les valeurs de "selection" en chaînes
var1 = str(selection[0][0])
var2 = str(selection[1][0])

#Écrire une phrase avec ces éléments
print(("{0} and {1} are states of the United States of America.").format(var1, var2))


# In[194]:


#Grouper les données de la base de données "df_Concat" par la colonne "State" et trier par ordre croissant de population

#Instancier la variable de groupement retournant une base de données regroupée selon la rangée des états et pour 
#laquelle les valeurs sont la somme calculée un fois le regroupement effectué
groupby_state = df_Concat_2.groupby('State').sum()

#Créer une nouvelle base de données à partir du résultat précédent pour pouvoir effectuer des manipulations
groupby_state_new = groupby_state.reset_index()

#Trier en ordre croissant de "Population_new" la nouvelle base de données
groupby_state_new_sorted = groupby_state_new.sort_values(by=["Population_new"])

#Renommer les colonnes contenant des espaces
groupby_state_new_sorted.rename(columns = {'1st quarter':'first_quarter', '2nd quarter':'second_quarter',
                                           '3rd quarter':'third_quarter', '4th quarter':'fourth_quarter'}, inplace = True)

#Afficher la nouvelle base de données
groupby_state_new_sorted


# In[195]:


#Définir les données à analyser

#Créer la variable de la population groupée par état en ordre croissant
sorted_pop = groupby_state_new_sorted["Population_new"]

#Créer la variable du nombre de crimes haineux du premier quartile de l'année 2013 groupé par état
q1_by_state = groupby_state_new_sorted["first_quarter"]

#Créer la variable du nombre de crimes haineux du deuxième quartile de l'année 2013 groupé par état
q2_by_state = groupby_state_new_sorted["second_quarter"]

#Créer la variable du nombre de crimes haineux du troisième quartile de l'année 2013 groupé par état
q3_by_state = groupby_state_new_sorted["third_quarter"]

#Créer la variable du nombre de crimes haineux du quatrième quartile de l'année 2013 groupé par état
q4_by_state = groupby_state_new_sorted["fourth_quarter"]


# In[196]:


#Définir les éléments à afficher dans les graphiques

#Paramètre X : population par état en ordre croissant
X = np.array(sorted_pop)

#Paramètres Y : nombre de crimes haineux par état par quartile de l'année 2013
Y1 = np.array(q1_by_state)
Y2 = np.array(q2_by_state)
Y3 = np.array(q3_by_state)
Y4 = np.array(q4_by_state)


# In[197]:


#Définir les paramètres des figures à tracer pour la visualisation des données

#Afficher en tant que figures
fig = plt.figure()

#Définir la hauteur et la largeur des figures
fig.set_figheight(8)
fig.set_figwidth(18)


#Créer le premier graphique
plt.subplot(2, 2, 1)

#Définir les données des axes "x" et "y"
x = X
y = Y1

#Définir les limites du graphique
xlim_min = X.min()
xlim_max = X.max()
ylim = Y1.max()

plt.axis([xlim_min, xlim_max, 0, ylim])
fig.subplots_adjust(hspace=0.5)

#Définir le titre du graphique et le titre des axes
plt.title('Nombre de crimes haineux pour le premier quartile de l\'année 2013 \n en fonction de la population des états')
plt.xlabel('Population')
plt.ylabel('Nombre de crimes haineux')

#Tracer le graphique
plt.plot(x, y, '-', color='k')


#Créer le deuxième graphique
plt.subplot(2, 2, 2)

#Définir les données des axes "x" et "y"
x = X
y = Y2

#Définir les limites du graphique
xlim_min = X.min()
xlim_max = X.max()
ylim = Y2.max()

plt.axis([xlim_min, xlim_max, 0, ylim])

#Définir le titre du graphique et le titre des axes
plt.title('Nombre de crimes haineux pour le deuxième quartile de l\'année 2013 \n en fonction de la population des états')
plt.xlabel('Population')
plt.ylabel('Nombre de crimes haineux')

#Tracer le graphique
plt.plot(x, y, '-', color='b')


#Créer le troisième graphique
plt.subplot(2, 2, 3)

#Définir les données des axes "x" et "y"
x = X
y = Y3

#Définir les limites du graphique
xlim_min = X.min()
xlim_max = X.max()
ylim = Y3.max()

plt.axis([xlim_min, xlim_max, 0, ylim])

#Définir le titre du graphique et le titre des axes
plt.title('Nombre de crimes haineux pour le troisième quartile de l\'année 2013 \n en fonction de la population des états')
plt.xlabel('Population')
plt.ylabel('Nombre de crimes haineux')

#Tracer le graphique
plt.plot(x, y, '-', color='g')


#Créer le quatrième graphique
plt.subplot(2, 2, 4)

#Définir les données des axes "x" et "y"
x = X
y = Y4

#Définir les limites du graphique
xlim_min = X.min()
xlim_max = X.max()
ylim = Y4.max()

plt.axis([xlim_min, xlim_max, 0, ylim])

#Définir le titre du graphique et le titre des axes
plt.title('Nombre de crimes haineux pour le quatrième quartile de l\'année 2013 \n en fonction de la population des états')
plt.xlabel('Population')
plt.ylabel('Nombre de crimes haineux')

#Tracer le graphique
plt.plot(x, y, '-', color='r')


# In[198]:


#Statistiques

#Importer le module "statsmodel"
import statsmodels

#Importer le module "ordinary least squares (OLS)"
from statsmodels.formula.api import ols


# In[199]:


#Régressions linéaires

#Effectuer la régression linéaire entre la population des états et le nombre de crimes haineux dans le premier quartile de
#l'année 2013
model1 = ols("first_quarter  ~ Population_new", groupby_state_new_sorted).fit()
print(model1.summary())

#Effectuer la régression linéaire entre la population des états et le nombre de crimes haineux dans le deuxième quartile de
#l'année 2013
model2 = ols("second_quarter ~ Population_new", groupby_state_new_sorted).fit()
print(model2.summary())

#Effectuer la régression linéaire entre la population des états et le nombre de crimes haineux dans le troisième quartile de
#l'année 2013
model3 = ols("third_quarter ~ Population_new", groupby_state_new_sorted).fit()
print(model3.summary())

#Effectuer la régression linéaire entre la population des états et le nombre de crimes haineux dans le quatrième quartile de
#l'année 2013
model4 = ols("fourth_quarter ~ Population_new", groupby_state_new_sorted).fit()
print(model4.summary())


# In[200]:


#Visualisation des régressions linéaires

#Importer le module "seaborn"
import seaborn as sns

#Tracer la régression linéaire entre la population des états et le nombre de crimes haineux dans le premier quartile de
#l'année 2013 
sns.lmplot(y='first_quarter', x='Population_new', data = groupby_state_new_sorted)

#Tracer la régression linéaire entre la population des états et le nombre de crimes haineux dans le deuxième quartile de
#l'année 2013 
sns.lmplot(y='second_quarter', x='Population_new', data = groupby_state_new_sorted)

#Tracer la régression linéaire entre la population des états et le nombre de crimes haineux dans le troisième quartile de
#l'année 2013 
sns.lmplot(y='third_quarter', x='Population_new', data = groupby_state_new_sorted)

#Tracer la régression linéaire entre la population des états et le nombre de crimes haineux dans le quatrième quartile de
#l'année 2013 
sns.lmplot(y='fourth_quarter', x='Population_new', data = groupby_state_new_sorted)


# In[201]:


#Effectuer l'ANOVA pour les quatre quartiles de l'année 2013 afin de voir si des quartiles sont significativement différents

#Importer "SciPy"
import scipy.stats as stats


# In[202]:


#Créer la base de données pour l'ANOVA
quarters = pd.DataFrame(groupby_state_new_sorted[["first_quarter", "second_quarter", "third_quarter", "fourth_quarter"]])

#Afficher la base de données
quarters.head()


# In[203]:


#Observer les données de la base de données "quarters"
pd.plotting.scatter_matrix(quarters[['first_quarter','second_quarter', 'third_quarter', 'fourth_quarter']])


# In[204]:


#Vérifier les prémisses pour l'ANOVA

#Vérification de la normalité avec le test de Shapiro-Wilk
normality = stats.shapiro(quarters)

#Vérification de l'homogénéité des variances avec le test de Levene
variance = stats.levene(groupby_state_new_sorted['first_quarter'], groupby_state_new_sorted['second_quarter'], 
             groupby_state_new_sorted['third_quarter'], groupby_state_new_sorted['fourth_quarter'])

print(normality)
print(variance)


# In[205]:


#Le résultat du test de Shapiro-Wilk est significatif, donc la normalité des données n'est pas respectés. Il faut utiliser
#une transformation des données afin d'effectuer l'ANOVA, les données présentant une forte asymétrie positive (vers la droite).


# In[206]:


#Transfomration des données en utilisant le logarithme

#Créer une base de données vide "df_log"
df_log = pd.DataFrame()

#Instancier la fonction anonyme "to_log"
to_log = lambda x: list(np.log(x+1))

#Instancier les variables pour la transformation de données par la fonction anonyme "to_log"
column1 = to_log(q1_by_state)
column2 = to_log(q2_by_state)
column3 = to_log(q3_by_state)
column4 = to_log(q4_by_state)

#Ajouter les colonnes à la base de données vide "df_log"
df_log["q1_log"] = column1
df_log["q2_log"] = column2
df_log["q3_log"] = column3
df_log["q4_log"] = column4

#Afficher la nouvelle base de données "df_log"
df_log.head()


# In[207]:


#Vérification de la normalité avec le test de Shapiro-Wilk
normality = stats.shapiro(df_log)
print(normality)


# In[208]:


#Puisque malgré la transformation des données, celles-ci demeurent trop asymétriques, on utilise le Mann-Whitney-U au lieu
#de l'ANOVA


# In[209]:


#Importer le module du test Mann-Whitney-U
from scipy.stats import mannwhitneyu

#Effectuer les tests sur chaque paire de quartiles
resultat_1_2 = mannwhitneyu(df_log['q1_log'], df_log['q2_log'])
resultat_1_3 = mannwhitneyu(df_log['q1_log'], df_log['q3_log'])
resultat_1_4 = mannwhitneyu(df_log['q1_log'], df_log['q4_log'])
resultat_2_3 = mannwhitneyu(df_log['q2_log'], df_log['q3_log'])
resultat_2_4 = mannwhitneyu(df_log['q2_log'], df_log['q4_log'])
resultat_3_4 = mannwhitneyu(df_log['q3_log'], df_log['q4_log'])

#Afficher le résultat de tous les tests
print("résultat 1-2", resultat_1_2)
print("résultat 1-3", resultat_1_3)
print("résultat 1-4", resultat_1_4)
print("résultat 2-3", resultat_2_3)
print("résultat 2-4", resultat_2_4)
print("résultat 3-4", resultat_3_4)


# In[210]:


#On voit ici qu'aucun quartile est significativement différent d'un autre.


# In[213]:


#Instancier la variable du fichier "ReadMe" à lire
file = os.path.join(sys.path[0], "ReadMe.txt")

#Lire le fichier
f = open(file, "r", encoding="utf-8")
print(f.read())

#Fermer le fichier
f.close()

