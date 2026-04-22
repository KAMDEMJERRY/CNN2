# mode live (par défaut) avec fichier par défaut
python3 scripts/logviewer.py

# fichier de log personnalisé
python3 scripts/logviewer.py ./logs/training_log.txt

# mode statique (génère et affiche le plot final)
python3 scripts/logviewer.py ./logs/training_log.txt --mode static

# résumé uniquement
python3 scripts/logviewer.py ./logs/training_log.txt --mode summary

# live avec rafraîchissement toutes les 3s
python3 scripts/logviewer.py ./logs/training_log.txt --mode live --refresh 3