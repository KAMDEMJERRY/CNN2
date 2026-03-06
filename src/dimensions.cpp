#include "CNNBuilder.hpp"

void demoMNIST() {
    std::cout << "\n=== MNIST Architecture (2D) ===" << std::endl;
    CNNBuilder mnist(28, 28, 3);  // 28×28×3 (C×H×W)

    mnist.addConvSame(32, 3)      // 32×28×28
         .addConvSame(32, 3)      // 32×28×28
         .addMaxPool(2)           // 32×14×14
         .addConvSame(64, 3)      // 64×14×14
         .addConvSame(64, 3)      // 64×14×14
         .addMaxPool(2)           // 64×7×7
         .addFlatten()            // 3136×1×1
         .addDense(128)           // 128×1×1
         .addDense(10);           // 10×1×1

    mnist.printArchitecture();
    std::cout << "Première taille dense : " << mnist.firstDenseInputSize() << std::endl;
}

void demoImageNet() {
    std::cout << "\n=== ImageNet Architecture (2D) ===" << std::endl;
    CNNBuilder imagenet(224, 224, 3);  // 3×224×224

    imagenet.addConv(64, 11, 4, 2)     // 64×55×55 (224→55 avec k=11,s=4,p=2)
            .addMaxPool(3, 2)          // 64×27×27 (55→27 avec pool=3,s=2)
            .addConv(192, 5, 1, 2)     // 192×27×27
            .addMaxPool(3, 2)          // 192×13×13
            .addConv(384, 3, 1, 1)     // 384×13×13
            .addConv(256, 3, 1, 1)     // 256×13×13
            .addConv(256, 3, 1, 1)     // 256×13×13
            .addMaxPool(3, 2)          // 256×6×6
            .addFlatten()              // 9216×1×1
            .addDense(4096)            // 4096×1×1
            .addDense(4096)            // 4096×1×1
            .addDense(1000);           // 1000×1×1

    imagenet.printArchitecture();
    std::cout << "Première taille dense : " << imagenet.firstDenseInputSize() << std::endl;
}

void demoVideo3D() {
    std::cout << "\n=== Video Classification Architecture (3D) ===" << std::endl;
    // Pour des clips vidéo 16×112×112 avec 3 canaux (C×D×H×W)
    CNNBuilder video3d(16, 112, 112, 3);

    video3d.addConv3D(64, 3, 1, 1)                 // 64×16×112×112
            .addMaxPool3D(1, 2, 2)                 // 64×16×56×56
            .addConv3D(128, 3, 1, 1)               // 128×16×56×56
            .addMaxPool3D(2, 2, 2)                 // 128×8×28×28
            .addConv3D(256, 3, 1, 1)               // 256×8×28×28
            .addConv3D(256, 3, 1, 1)               // 256×8×28×28
            .addMaxPool3D(2, 2, 2)                 // 256×4×14×14
            .addConv3D(512, 3, 1, 1)               // 512×4×14×14
            .addMaxPool3D(2, 2, 2)                 // 512×2×7×7
            .addGlobalAvgPool()                    // 512×1×1×1
            .addFlatten()                           // 512×1×1
            .addDense(512)                          // 512×1×1
            .addDense(400);                          // 400×1×1 (400 classes UCF-101)

    video3d.printArchitecture();
    std::cout << "Première taille dense : " << video3d.firstDenseInputSize() << std::endl;
}

void demoMedical3D() {
    std::cout << "\n=== Medical 3D Segmentation Architecture ===" << std::endl;
    // Scanner médical 64×128×128, 1 canal
    CNNBuilder medical(64, 128, 128, 1);

    medical.addConv3D(32, 3, 1, 1)                  // 32×64×128×128
           .addConv3D(32, 3, 1, 1)                  // 32×64×128×128
           .addMaxPool3D(2)                          // 32×32×64×64
           .addConv3D(64, 3, 1, 1)                  // 64×32×64×64
           .addConv3D(64, 3, 1, 1)                  // 64×32×64×64
           .addMaxPool3D(2)                          // 64×16×32×32
           .addConv3D(128, 3, 1, 1)                 // 128×16×32×32
           .addConv3D(128, 3, 1, 1)                 // 128×16×32×32
           .addGlobalAvgPool()                       // 128×1×1×1
           .addFlatten()                              // 128×1×1
           .addDense(256)                             // 256×1×1
           .addDense(3);                              // 3 classes

    medical.printArchitecture();
    std::cout << "Première taille dense : " << medical.firstDenseInputSize() << std::endl;
}

void demoComparaison() {
    std::cout << "\n=== Comparaison 2D vs 3D ===" << std::endl;
    
    // Même architecture en 2D et 3D
    std::cout << "\n-- Version 2D --" << std::endl;
    CNNBuilder net2d(32, 32, 3);
    net2d.addConv(64, 3, 1, 1)
         .addMaxPool(2)
         .addConv(128, 3, 1, 1)
         .addGlobalAvgPool()
         .addFlatten()
         .addDense(10);
    net2d.printArchitecture();

    std::cout << "\n-- Version 3D --" << std::endl;
    CNNBuilder net3d(16, 32, 32, 3);
    net3d.addConv3D(64, 3, 1, 1)
         .addMaxPool3D(2)
         .addConv3D(128, 3, 1, 1)
         .addGlobalAvgPool()
         .addFlatten()
         .addDense(10);
    net3d.printArchitecture();
}

int main() {
    try {
        // Démonstrations des différentes utilisations
        demoMNIST();
        demoImageNet();
        demoVideo3D();
        demoMedical3D();
        demoComparaison();

        // Exemple d'utilisation avec un Tensor existant
        std::cout << "\n=== Construction depuis un Tensor ===" << std::endl;
        Tensor input4d(1, 3, 224, 224);  // Batch×C×H×W
        CNNBuilder fromTensor(input4d);
        fromTensor.addConvSame(64, 7, 2)
                  .addMaxPool(3, 2)
                  .addFlatten()
                  .addDense(1000);
        fromTensor.printArchitecture();

    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }

    return 0;
}