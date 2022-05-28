Description du script

Voici ce que ce script fait :

1. Importer les librairies et les modules nécessaires.
2. Créer une classe pour l'ouverture du fichier de données selon son type.
3. Localiser le fichier de données à ouvrir et lire le fichier avec la classe
   précédente, avec la librairie Pandas.
4. Visualiser la base de données.
5. Créer une copie indépendante de la base de données pour la manipuler.
6. Éliminer les valeurs manquantes de la base de donées.
7. Changer les valeurs de la colonne "Population" de type str à int pour les
   manipulations.
8. Changer les valeurs de la colone "Gender" de type str à int pour les 
   manipulations.
9. Appeler la fonction externe "Verification" qui vérifie les différents attributs
   des bases de données créées jusqu'à présent.
10. Créer une liste de la population triée en ordre croissant.
11. Créer un algorithme de recherche automatisé pour trouver une valeur de
    population, ou sinon suggérer les deux valeurs de population les plus près.
12.Préparer les données des "quartiles" de la base de données à être analysées
   par l'apprentissage automatique en les standardisant (transformation).
13.Effectuer une analyse par apprentissge automatique non-supervisé (ACP) 
   en composante principale, en réduisant les dimensions des données
   standardisées de 4 à 2 dans le modèle, puis visualiser les résultats.
14. Effectuer une validation croisée des données réduites avec "ShuffleSplit"
    avant de lancer une machine de support vectoriel (MSV) comme apprentissage
	automatique supervisé, basé sur une validation croisée à 5 itérations.
15. Évaluer l'efficacité de la machine de support vectoriel en tant que
    classificateur, avec des scores et de façon graphique.
16. Créer un espace de données dans la base de données SQLite3, y insérer une base
    de données avec des valeurs et afficher la base de données avec Pandas.
17. Récupérer des éléments de la base de données sur SQLite3 afin de les utiliser
    dans une phrase.
18. Grouper la base de donnée initiale par les valeurs de la colonne "State" et
    définir les données à analyser, soit la "population" et les "quartiles".
19. Tracer des figures pour la visualisation des données (corrélation).
20. Effectuer une régression linéaire sur chaque "quartile" en relation avec la
    "population", puis afficher les résultats numériques et graphiques.
21. Effectuer les tests de prémisses pour l'ANOVA entre "quartiles".
22. Transformer les données pour tenter de les normaliser aux fins de l'ANOVA.
23. Effectuer des tests Mann-Whitney-U entre chaque paire de "quartiles" et 
    afficher les résultats.
24. Lire ce fichier de texte présentant le code et ses résultats.

Description des résultats

Le but de ce projet était de vérifier si certains états étaient plus susceptibles
de retrouver des crimes haineux sur leur territoire selon leur population. L’année
2013 était divisée en quartiles, analysés de façon indépendante au niveau du 
nombre de crimes haineux.

Deux hypothèses on été posées :

Hypothèse 1. Un état plus populeux recensera un plus grand nombre de crimes en 
général.

Hypothèse 2. Les deux derniers quartiles de l'année présenteront plus de crimes 
haineux.

Voici les résultats :

Les populations des états, classées en ordre croissant, auraient une relation 
linéaire directement proportionnelle avec le nombre de crimes haineux dans une
analyse de régression linéaire. Cette analyse se ferait indépendamment sur chaque
quartile.

Visuellement, cette supposition semble vraie en regardant les graphiques 
descriptifs des données. De plus, les graphiques de régression linéaire montrent 
clairement cette tendance. Pour les régressions linéaires utilisant la méthode des 
"ordinary least squares (OLS)", le pourcentage de variance de la variable 
dépendante expliquée par la variable indépendante est respectivement de 0.812, 
0.668, 0.685 et 0.639 pour les quartiles 1 à 4, ce qui est assez élevé. Les 
valeurs de p des modèles sont respectivement toutes <.001, donc les modèles de 
régression linéaire sont statistiquement significatifs.

Ainsi, l'hypotèse 1 est confirmée.

Puis, on s’attendait à ce que les deux derniers quartiles de l'année aient un 
nombre total de crimes haineux, tout états confondus, plus élevé que pour les 
deux premiers quartiles. Toutefois, cette hypothèse n'a pas pu être véfiriée
par une ANOVA, car la prémisee de normalité des données testée par le test de 
Shapiro-Wilk n'était pas respectée, même suite à une transformation des données
par le logarithme. À cause de la trop grande asymétrie dans les données, c'est
donc le test de Mann-Whitney-U qui a été utilisé entre tous les quratiles afin
de les comparer. Les résultats ont démontré qu'aucune quartile ne se différenciait
significativement d'un autre.

Ainsi, l'hypothèse 2 est infirmée.

Conclusion

Donc, on peut conclure que les états plus populeux ont plus de risque de conaître
des crimes haineux sur leur territoire, toutefois de façon non différenciée tout
au long de l'année.

Le plan original de ce script a été suivi dans son entiêreté.

Dominic Cusson