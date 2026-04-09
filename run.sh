#!/bin/bash
#  3d, sparse, attn, sparse_attn
arch=("2d" "3d" "sparse" "attn" "sparse_attn")

idx=2 # 0: 2d, 1: 3d, 2: sparse, 3: attn, 4: sparse_attn

# Définir la variable
pipeline=${arch[$idx]}

for i in {1..5}
do
    echo "Exécution $i"
    ./build/src/CNN "$pipeline" 
    # > "output_run_$i.txt" 2>&1
    echo "Résultats sauvegardés dans output_run_$i.txt"
done

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

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)