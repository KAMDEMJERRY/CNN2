Générer les tests unitaires pour toutes les classes
Classes à couvrir
 Tensor — déjà couvert (
test/Tensor.cpp
)
 ConvLayer — déjà couvert (
test/ConvLayer.cpp
)
 DenseLayer → test/test_DenseLayer.cpp
 ActivationLayer (ReLU, LeakyReLU, Sigmoid, Softmax) → test/test_ActivationLayer.cpp
 PoolLayer (MaxPool2D, GlobalAvgPool2D, MaxPool3D, GlobalAvgPool3D) → test/test_PoolLayer.cpp
 LossLayer (CrossEntropyLoss, MSELoss, SoftmaxCrossEntropyLayer) → test/test_LossLayer.cpp
 Optimizer (SGD, Adam, StepDecay, GradientUtils) → test/test_Optimizer.cpp
 DropoutLayer → test/test_DropoutLayer.cpp
 DimensionCalculator → test/test_Dimensions.cpp
 CNN (forward/backward/addLayer/predict) → test/test_CNN.cpp
Mise à jour CMakeLists.txt
 Ajouter toutes les nouvelles cibles dans 
test/CMakeLists.txt
Vérification
 Compiler avec cmake --build build --target <test>
 Exécuter tous les tests

Comment
