# C'était mieux avant  

La société "Réfractaire", fleuron de la recherche médicale, utilise des algorithmes de Deep Learning pour ses projets d'analyses d'images (principalement autour des cancers).  
Aujourd'hui, ses projets fonctionnent mais la société fait parfois face à des soucis de régressions dans leurs applications.
Après quelques recherche il s'avère, que une fois sur deux, l'algorithme de Deep Learning fait des erreurs.  
Elle vous sollicite pour l'aider.

## OBJECTIF :  

La société "Réfractaire" est une spécialiste de l'intelligence artificielle autour de la recherche médicale.  
Elle possède des Data Scientists et a déjà réalisée de nombreux algorithmes.  

Elle n'est cependant pas très mature sur les outils dits "MLOps", qui permettent entre autre, de monitorer les résultats des algorithmes qu'elle conçoit.  
Algorithmes qui aujourd'hui, peuvent lui poser des problèmes lorsque l'entrainement dégrade les performances de ces derniers (pour diverses raisons).  

Elle vous demande de lui présenter l'outil MLFlow à l'aide d'un projet (brief) sur lequel vous avez déjà travaillé.  
De lui montrer les principales fonctionnalités :  

*Création d'un fichier MLProject*  
*Versionning avec Git*  
*Gestion des paramètres entre les runs*  
*Exécuter sur un fichier python et/ou un notebook*  

## EXECUTION :  

Télécharger ou cloner le repo.  
Lancer la commande suivante dans un terminal, dans le dossier REFRACTAIRE en remplaçant les paramètres {} par les valeurs souhaitées (à renouveler autant de fois que nécessaire avec différents paramètres) :
```
python BetterBefore.py -r {Conv1_layers} {Conv1_kernelsize} {Conv1_activation} {Conv2_layers} {Conv2_kernelsize} {Conv2_activation} {Dropout1} {Dense_units} {Dense_activation} {Dropout2}
``` 

Puis lancer la commande suivante :
```
mlflow ui
```

Enfin, accédez à l'adresse [http://localhost:5000/](http://localhost:5000/) pour visualiser les résultats du monitoring.




