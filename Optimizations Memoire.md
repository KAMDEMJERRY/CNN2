Guide des Optimisations Mémoire — Projet CNN C++
Ce document détaille les opportunités d'optimisation mémoire identifiées dans le projet, classées par impact et difficulté d'implémentation.

1. Élimination des Copies Profondes dans les Caches (input_cache)
   Le Problème (Pourquoi ?)
   Presque toutes les couches (
   ConvLayer
   ,
   DenseLayer
   ,
   BatchNorm3D
   ,
   ReLULayer
   ,
   MaxPoolLayer
   ) utilisent ce motif dans le
   forward
   :

cpp
input_cache = input; // Déclenche la copie de tout le buffer de données (Eigen::Tensor)
C'est coûteux en temps CPU (passage en mémoire) et double la consommation RAM pendant l'entraînement car chaque couche garde une copie complète de ses entrées.

La Solution (Comment ?)
Utiliser un stockage par pointeur ou référence :

Comment : Remplacer Tensor input_cache par const Tensor\* input_cache_ptr = nullptr;.
Condition : L'objet input source doit rester vivant jusqu'à la fin de la passe
backward
. Dans une boucle d'entraînement standard, c'est garanti.
TIP

Gain estimé : Division par 2 de la mémoire utilisée par les activations du réseau.

2. Opérations "In-Place" pour les Activations
   Le Problème (Pourquoi ?)
   Les couches d'activation (
   ReLU
   ,
   LeakyReLU
   ) font souvent deux copies :

input_cache = input; (Première copie pour le backward)
Tensor output = input; (Deuxième copie pour la sortie) Puis elles modifient
output
.
La Solution (Comment ?)
In-Place : Si la couche précédente ne réutilise pas son propre output, on peut modifier le tenseur directement d'un point de vue mathématique.
Zéro-Copy Forward : Au lieu de créer
output
, on peut modifier input_cache (si on accepte de stocker la version activée) ou simplement stocker un pointeur vers l'input et n'allouer qu'UN SEUL nouveau tenseur pour le résultat. 3. Optimisation du Pipeline de Données (DataLoader & Augment)
Le Problème (Pourquoi ?)
Dans
MedMNIST3DDataset
et
DataLoader3D
, les données subissent une cascade de memcpy :

Extraction du volume du dataset (std::memcpy)
Copie vers le buffer d'augmentation (std::memcpy)
Copie du résultat de l'augmentation (std::memcpy)
Assemblage dans le batch final (std::memcpy)
La Solution (Comment ?)
Eigen::Map : Utiliser des "Vues" (Mapping mémoire). Au lieu de copier un volume d'un dataset, on devrait pouvoir créer un
Tensor
qui pointe directement sur l'adresse mémoire du dataset : images.getData() + offset.
Augmentation in-place : Modifier le MedMNIST3DDataset::getBatch pour qu'il écrive directement dans le tenseur de batch final, et que l'augmentateur travaille aussi directement sur ce buffer de destination. 4. Gestion des Buffers Temporaires (
BatchNorm3D
)
Le Problème (Pourquoi ?)
Dans
BatchNorm3DLayer.hpp
, des vecteurs sont redimensionnés à chaque passage :

cpp
x*hat*.resize(B _ C _ D _ H _ W); // Allocation dynamique au milieu du forward
mean\_.resize(C);
La Solution (Comment ?)
Pré-allocation : Faire de ces buffers des membres de la classe. Ils ne seront ré-alloués que si la taille du batch change (rare durant l'entraînement).
Recréer le buffer une seule fois : Utiliser reserve() ou simplement laisser la taille s'adapter au premier batch et ne plus y toucher. 5. Exploitation de GEMM (Eigen) et noalias()
Le Problème (Pourquoi ?)
Plusieurs couches effectuent des produits matriciels vers un buffer temporaire puis recopient les données :

cpp
Eigen::MatrixXf out*local = W * col;
// ... copie de out*local vers output ...
La Solution (Comment ?)
noalias() : Eigen permet d'écrire le résultat d'un produit matriciel directement dans la mémoire finale d'un
Tensor
via un Eigen::Map.
Comment :
cpp
Eigen::Map<Eigen::MatrixXf> out_map(output.getData(), rows, cols);
out_map.noalias() = W * col; // Zéro allocation temporaire, GEMM direct 6. Design "Tensor Views" (Architecture Avancée)
Pourquoi ?
Le design actuel de la classe
Tensor
impose la propriété des données (Eigen::Tensor data).

Comment ?
Une refonte majeure consisterait à séparer le Buffer de données (via std::shared_ptr<float[]>) et la Shape/Strides.

Cela permettrait au
reshape()
,
as4D()
,
as5D()
et au slicing (extraire un batch) de s'exécuter en temps constant (O(1)) sans aucune copie de données, en partageant simplement le pointeur vers le buffer de base
