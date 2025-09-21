## Pipeline de rehaussement vasculaire

Ce projet est une implémentation modulaire d'un pipeline de rehaussement vasculaire en vue d'étudier l'influence des opérateur de dérivation discrète sur le résultat de la segmentation. Il permet d'effectuer les différentes étapes de pré-traitement, de traitement et d'analyse.

### Table des matières

[1. Structure](#1-structure)

[2. Installation](#2-installation)

[3. Utilisation](#3-utilisation)

[4. Remarques](#4-remarques)

### 1. Structure

Le code est organisé selon la structure suivante.

#### 1.1 Configs

Le dossier `configs` réuni toutes les configurations permettant de lancer les différentes expériences. Il permet de modifier facilement les paramètres d'une expérience et de gérer les commandes terminal pour une utilisation plus ergonomique.

#### 1.2 Core

Le dossier `core` contient toute la logique interne au traitement des données, allant du chargement des données à la sauvegarde des résultats en passant par le traitement de l'image.

Le sous-dossier `benchmark` contient la logique de traitement d'un ensemble de jeu de données en vue d'étudier l'influence des paramètres de rehaussement (`enhancement.py`) ou d'étudier l'influence de l'opérateur de dérivation (`hessian.py`).

Le sous-dossier `config` contient la définition des différents types et classe de l'ensemble du projet. Il permet de s'assurer de la cohérence d'ensemble et de bénéficier de l'auto-complétion. C'est de cette logique que découle les différentes configurions présentées dans le dossier `configs`.

Le sous-dossier `io` (input-output) contient la logique relative au entrée et sortie du pipeline de traitement. Il permet entre autre de gérer le chargement des données, les logs intermédiaire et la sauvegarde des résultats.

Le sous-dossier `processing` gère le coeur du pipeline et les différentes étapes du traitement. Il gère notamment les étapes de calcul de la matrice Hessienne (`derivator.py`), rehaussement (`enhacer.py`), d'application du traitement (`processor.py`) et de segmentation (`segmenter.py`). Le fichier `pipeline.py` permet de lancer tout le pipelie de traitement sur une image spécifique, selon les paramètres défini dans le fichier yaml correspondant.

Le sous-dossier `utils` contient toutes les fonctions utilitaire.

### 1.5 Data

Le dossier `docs` contient les différentes données à traiter. ATTENTION, ce fichier ne doit pas être renommé ou inclu dans le nom du fichier de chargement des données dans les paramètre de l'expérience car il est déjà inclu à partir du fichier `configs/args.py`.

### 1.4 Docs

Le dossier `docs` contient la documentation liées au différents éléments de code présent dans le projet. Il fournit une explication des fonction principale ainsi qu'un exemple d'utilisation.

### 1.5 Logs

Le dossier `logs` contient tout les logs, triés par expérience et nommer avec la date exacte d'éxécution. ATTENTION, ce fichier ne doit pas être renommé ou inclu dans le nom du fichier des logs dans les paramètre de l'expérience car il est déjà inclu à partir du fichier `configs/args.py`.

### 1.6 Outputs

Le dossier `logs` contient tout les résultats sauvegardé selon le nom de l'expérience et la date exacte d'éxécution. ATTENTION, ce fichier ne doit pas être renommé ou inclu dans le nom du fichier de sauvegarde des données dans les paramètre de l'expérience car il est déjà inclu à partir du fichier `configs/args.py`.

### 2. Installation

Commandes d'installation :
`git clone [repo_url]`
`cd [project_folder]`
`pip install -r requirements.txt`

### 3. Utilisation

#### 3.1 Pipeline

Le module `pipeline.py` dans le dossier `core/processing` sert à orchestrer les différentes étapes du traitement de données ou d'images dans le projet. Il permet de définir, organiser et exécuter une séquence d'opérations (prétraitement, segmentation, post-traitement, etc.) de manière structurée et automatisée. Ce module facilite la reproductibilité des expériences et la modularité du code en regroupant les différentes fonctions de traitement dans une pipeline configurable.

##### Fonctionnalités principales

- Définition d'une pipeline de traitement avec plusieurs étapes.
- Exécution séquentielle ou conditionnelle des étapes.
- Gestion des entrées/sorties et des paramètres de chaque étape.
- Intégration avec les autres modules du projet (prétraitement, segmentation, etc.).

##### Exemple d'utilisation

1. Depuis un notebook

Un exemple complet est donné dans le fichier `exemple.ipynb` section Pipeline.

2. Depuis le terminal

La commande `python main.py --run_pipeline [--test]` lance le pipeline sur une image définie dans le fichier de configuration `pipeline.yaml`. Tous les paramètres du traitement sont définis dans ce fichier. La sous-commande `--test` permet d'aller chercher les fichiers de configuration dans `tests/configs` plutôt que `configs` afin de faciliter la gestion des tests.

#### 3.2 Benchmark

##### Fonctionnalités principales

Il est possible de lancer 2 types de benchmark différents : une étude sur les paramètres du rehaussement (`enhancement.py`) ou une évaluation de l'impact des méthode de dérivation sur le rehaussement (`hessian`). Le fichier `runner.py` permet d'orcherstrer le choix de la méthode et l'itération sur les différentes image du jeu de données sélectionné.

##### Exemple d'utilisation

Le benchmark peut se lancer depuis le terminal avec la commande `python main.py --run_benhchmark --benchmark_type [hessian|enhancement] [--test]`. La sous-commande `--test` permet d'aller chercher les fichiers de configuration dans `tests/configs` plutôt que `configs` afin de faciliter la gestion des tests. Les paramètres relatif au setup et au jeu de données sont dans `runner.yaml`, les paramètres de l'expérience dans `experiment.yaml` et les paramètres relatifs à chaque benchmark dans `hessian.yaml` ou `enhancement.yaml`.

### 4. Remarques

- ATTENTION, le paramètre black_ridges est crucial pour obtenir des résultats cohérents, c'est la première source d'erreur à laquelle prendre garde.

- Lorsqu'on utilise le pipeline la parallelisation pour les images 3D on ne peut pas utiliser gamma = Null car le cas échéant le gamma sélectionné pour chaque chunk sera différent, ce qui résulte en une correction inhomongène de l'intensité au sein de l'image. Dans ce cas, il est important de fixer le gamma au préalable.

- La ligne droite visible sur la courbe ROC à partir d'un certain seuil signifie qu'il n'y a plus qu'une seule classe à partir de ce seuil.

- Dans le benchmark sur les paramètres du rehaussement, le paramètre alpha ne varie pas dans le cas 2D car il n'est pas utilisé dans le cas 2D (blobness)...
