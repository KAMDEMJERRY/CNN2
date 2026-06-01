#!/bin/bash
#  3d, sparse, attn, sparse_attn
arch=("2d" "3d" "sparse" "attn" "sparse_attn" "convnext_dense" "convnext_sparse")
skipTrain="--skip-train"
idx=5 # 0: 2d, 1: 3d, 2: sparse, 3: attn, 4: sparse_attn, 5: convnext_dense, 6: convnext_sparse

# Définir la variable
pipeline=${arch[$idx]}

# for i in {1..5}
# do
#     echo "Exécution $i"
#     ./build/src/CNN "$pipeline" 
#     # > "output_run_$i.txt" 2>&1
#     echo "Résultats sauvegardés dans output_run_$i.txt"
# done

# Exécuter le programme en passant la variable comme argument
# ./build/src/CNN "$pipeline"
# ./build/src/CNN "$pipeline" > "output_run_$i.txt" 2>&1

# Mesurer les performances avec perf
# sudo perf stat -e cpu-cycles,instructions,cache-misses,cache-references,page-faults,context-switches ./build/src/CNN2D 2>> perf_results.txt





# # Nettoyer le build précédent (obligatoire après changement de flags)
# rm -rf build/ && mkdir build && cd build

# # Debug (comportement actuel, + ASan)
# cmake .. -DCMAKE_BUILD_TYPE=Debug
# make -j$(nproc)

# # RelWithDebInfo (pour profiler)
# cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
# make -j$(nproc)

# # Release (mesure du speedup)
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j$(nproc)


cd build && cmake .. -DCMAKE_BUILD_TYPE=Release &&

make -j$(nproc) && cd .. && 
./build/src/CNN "$pipeline"  &&

# Après exécution, sélectionner les fichiers appropriés selon le pipeline
case "$pipeline" in
	2d)
		PRED="./logs/predictions_2d.csv"
		TRAIN_LOG="./logs/train_2d.txt"
		;;
	3d)
		PRED="./logs/predictions_3d.csv"
		TRAIN_LOG="./logs/train_3d.txt"
		;;
	sparse)
		PRED="./logs/predictions_sparse.csv"
		TRAIN_LOG="./logs/train_sparse.txt"
		;;
	attn)
		PRED="./logs/predictions_3d_attn.csv"
		TRAIN_LOG="./logs/train_3d_attn.txt"
		;;
	sparse_attn)
		PRED="./logs/predictions_sparse_attn.csv"
		TRAIN_LOG="./logs/train_sparse_attn.txt"
		;;
	convnext_dense)
		PRED="./logs/predictions_convnext_dense.csv"
		TRAIN_LOG="./logs/train_convnext_dense.txt"
		;;
	convnext_sparse)
		PRED="./logs/predictions_convnext_sparse.csv"
		TRAIN_LOG="./logs/train_convnext_sparse.txt"
		;;
	*)
		PRED="./logs/predictions.csv"
		TRAIN_LOG="./logs/training_log.txt"
		;;	
esac

# Tracer ROC pour le pipeline courant et afficher le log d'entraînement
python3 scripts/plot_roc.py "$PRED" -o "docs/results/roc_${pipeline}.png"
python3 scripts/logviewer.py "$TRAIN_LOG" --mode static -o "docs/results/train_${pipeline}.png"



# Usage:

# Static: python3 logviewer.py train_3d.txt --mode static -o docs/results/train_3d.png
# Live: python3 logviewer.py train_3d.txt --mode live -o docs/results/live_3d.png (Ctrl+C pour sauvegarder)
# plot_roc.py: déjà acceptait -o/--out. Usage:

# python3 plot_roc.py predictions_3d.csv -o docs/results/roc_3d.png