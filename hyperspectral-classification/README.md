# Hyperspectral Classification

Ce dépôt contient du code pour l'entraînement et l'évaluation de modèles de segmentation de zones urbaines sur des données hyperspectrales (jeu de données LCZ / PRISMA / Sentinel). Le projet inclut des modèles basés sur SegFormer, FCN-ResNet50, un CNN propriétaire et un classifieur Random Forest pour segmentation.

Ce README donne les instructions d'installation, d'utilisation, la structure du projet et des conseils pour reproduire les expériences présentes dans `save/`.

## Fonctionnalités

- Entraînement et validation de modèles PyTorch (SegFormer, ResNet50-FCN, UNet, CNN personnalisé).
- Option de segmentation par Random Forest (non GPU) pour comparaison.
- Logging et suivi d'expériences avec Weights & Biases (W&B).
- Utilitaires pour préparation de dataset et visualisation.

## Prérequis

- Python 3.8+ (développé et testé sur 3.10/3.11)
- CUDA et pilotes GPU (optionnel mais recommandé pour l'entraînement)

Les dépendances principales (non exhaustif) :

- torch
- torchvision
- segmentation-models-pytorch
- torchmetrics
- scikit-learn
- numpy
- matplotlib
- tqdm
- wandb
- configargparse
- iterstrat

Créez un environnement virtuel puis installez les dépendances. Exemple avec `pip` :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Si vous n'avez pas de `requirements.txt` dans le dépôt, installez au moins :

```powershell
pip install torch torchvision segmentation-models-pytorch torchmetrics scikit-learn numpy matplotlib tqdm wandb configargparse iterstrat
```

Remarque : pour `segmentation-models-pytorch` et certaines versions de `torch`, il peut être nécessaire d'installer des versions compatibles. Consultez la documentation des paquets.

## Structure du dépôt

Principaux fichiers et dossiers :

- `main.py` : point d'entrée pour lancer l'entraînement et l'évaluation.
- `arguments/` : définitions des arguments CLI (ex. `train.py`).
- `models/` : implémentations de modèles (CNN, Random Forest, ...).
- `utils/` : fonctions utilitaires (`dataset.py`, `helper_functions.py`, visualisation).
- `save/` : dossiers d'expérimentations et checkpoints.
- `tests/` : tests rapides (ex. `test_cuda.py`).
- `notebooks/` : notebooks d'exploration et visualisation.

## Utilisation

Le point d'entrée principal est `main.py`. Les options CLI sont définies dans `arguments/train.py`. Exemple d'utilisation minimale :

```powershell
# entraînement par défaut (segformer, dataset 'berlin')
python main.py

# avec options (exemple) :
python main.py --model resnet50 --dataset berlin --num_epochs 50 --batch-size 8 --lr 0.0005 --wandb --wandb-project "MonProjet"

# utiliser un fichier de configuration (configargparse) :
python main.py -c config.ini
```

Options importantes (extraites de `arguments/train.py`) :

- `--save-dir` : dossier pour sauvegarder logs et checkpoints (par défaut `./save/`).
- `--num_epochs` : nombre d'époques.
- `--batch-size` : taille de batch.
- `--dataset` : `berlin`, `athens`, `milan` ou `full`.
- `--sampler` : stratégie d'échantillonnage (`random`, `stratified`, `skfold`).
- `--model` : `segformer`, `resnet50`, `ownCNN`, `randomforest`.
- Arguments spécifiques Random Forest : `--rf-n-estimators`, `--rf-max-depth`, etc.

Exemples rapides :

```powershell
# Entrainer un ResNet-FCN sur Berlin pendant 100 époques et log sur W&B
python main.py --model resnet50 --dataset berlin --num_epochs 100 --batch-size 8 --lr 0.001 --wandb --wandb-project Hyperspectral

# Lancer l'entraînement d'une Random Forest (pas de GPU requis)
python main.py --model randomforest --dataset berlin --rf-n-estimators 200
```

## Tests rapides

Vérifiez que PyTorch et CUDA sont disponibles avec le script de test :

```powershell
python tests/test_cuda.py
```

## Reproduire les expériences sauvegardées

Les runs W&B et artefacts se trouvent sous `save/` (ex. `save/berlin/experiment_...`). Chaque dossier contient :

- `args.csv` : arguments utilisés pour l'expérience.
- `wandb/` : export offline des logs W&B pour reproduction.

Pour reproduire une expérience, récupérez les arguments dans `args.csv` et lancez `python main.py -c args.csv` ou reconstruisez la ligne de commande correspondante.

## Contribution

Contributions bienvenues : ouvrez une issue ou un pull request. Avant d'ajouter des dépendances lourdes, vérifiez la compatibilité GPU/CUDA et documentez les versions.

Checklist minimale pour PR :

1. Tests unitaires / smoke tests locaux.
2. Mise à jour du `README.md` si nécessaire.
3. Pas d'informations sensibles dans les commits.

## Licence

Indiquez ici la licence du projet (ex. MIT). Si aucune licence n'est fournie, le code est considéré comme propriétaire.

## Remarques / Limitations connues

- Certaines parties utilisent W&B — sans clé W&B, les logs locaux sont encore écrits dans `save/` mais l'envoi peut échouer.
- Les chemins des datasets dans `main.py` sont relatifs et attendent des fichiers `PRISMA_30.tif`, `S2.tif` et `LCZ_MAP.tif` sous `./dataset/<city>/`.
- Les noms de classes dans les matricies de confusion sont actuellement `Class 0..17` — remplacez-les par des étiquettes métier si disponibles.

---

Si vous voulez, je peux :

- Générer automatiquement un `requirements.txt` basé sur les imports du projet.
- Ajouter des exemples de config (`config.ini`) prêts à l'emploi.
- Traduire/ajouter des sections en anglais.

Dites-moi ce que vous préférez.
