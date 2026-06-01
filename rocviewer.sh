# utiliser le fichier par défaut
python3 scripts/plot_roc.py

# spécifier un fichier CSV
python3 scripts/plot_roc.py ./logs/predictions_3d.csv

# spécifier fichier et sortie
python3 scripts/plot_roc.py ./logs/predictions_3d.csv -o docs/results/roc_3d.png