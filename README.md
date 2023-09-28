# Projet 7 - Implémentez un modèle de scoring

## Contexte

L'entreprise "Prêt à dépenser" propose des crédits à la consommation pour des personnes ayant peu ou pas d'historique de prêt. Elle souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).
De plus, dans un souci de transparence, l'entreprise veut développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

## Mission

- Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.  
- Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.  
- Mettre en production le modèle de scoring de prédiction à l’aide d’une API, ainsi que le dashboard interactif qui appelle l’API pour les prédictions.

## Les données

Voici les [données](https://www.kaggle.com/c/home-credit-default-risk/data) dont vous aurez besoin pour réaliser le dashboard.

## Livrables

- L’application de dashboard interactif répondant aux spécifications ci-dessus et l’API de prédiction du score, déployées chacunes sur le cloud.  
- Un dossier, géré via un outil de versioning de code contenant :  
    - Le notebook ou code de la modélisation (du prétraitement à la prédiction), intégrant via MLFlow le tracking d’expérimentations et le stockage centralisé des modèles  
    - Le code générant le dashboard  
    - Le code permettant de déployer le modèle sous forme d'API  
    - Pour les applications dashboard et API, un fichier introductif permettant de comprendre l'objectif du projet et le découpage des dossiers, et un fichier listant les packages utilisés seront présents dans les dossiers  
    - Le tableau HTML d’analyse de data drift réalisé à partir d’evidently  
- Une note méthodologique décrivant :  
    - La méthodologie d'entraînement du modèle (2 pages maximum)  
    - Le traitement du déséquilibre des classes (1 page maximum)  
    - La fonction coût métier, l'algorithme d'optimisation et la métrique d'évaluation (1 page maximum)  
    - Un tableau de synthèse des résultats (1 page maximum)  
    - L’interprétabilité globale et locale du modèle (1 page maximum)  
    - Les limites et les améliorations possibles (1 page maximum)  
    - L’analyse du Data Drift (1 page maximum)  
- Un support de présentation pour la soutenance, détaillant le travail réalisé (Powerpoint ou équivalent, 30slides maximum). 
