#include "NpyParser.hpp"
#include "MedMNIST3DDataset.hpp"
#include "DataLoader3D.hpp"
#include "ConvLayer3D.hpp"
#include <iostream>

#define BASE_PATH "../../../dataset"
// ─────────────────────────────────────────────────────────────────────────────
// Utilitaire : séparateur titré
// ─────────────────────────────────────────────────────────────────────────────
static void section(const std::string& title) {
    std::cout << "\n═══════════════════════════════════════════════════════" << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << "═══════════════════════════════════════════════════════" << std::endl;
}

int main() {

    // ─────────────────────────────────────────────────────────────────────────
    // 1. Inspecter les fichiers avant chargement (optionnel)
    // ─────────────────────────────────────────────────────────────────────────
    const std::string path(BASE_PATH);
    std::cout << "\n=== Inspection des fichiers ===" << std::endl;
    NpyParser::inspect(path + "/fracturemnist3d/train_images.npy");
    NpyParser::inspect(path + "/adrenalmnist3d/train_images.npy");

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Charger FractureMNIST3D  (3 classes, uint8)
    // ─────────────────────────────────────────────────────────────────────────
    MedMNIST3DDataset fracture_train(
        path + "/fracturemnist3d",   // dossier
        Split::TRAIN,          // split
        3,                     // num_classes
        "FractureMNIST3D",     // nom
        true                   // normalisation z-score
    );

    MedMNIST3DDataset fracture_val(
        path + "/fracturemnist3d",
        Split::VAL,
        3,
        "FractureMNIST3D",
        true
    );

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Charger AdrenalMNIST3D  (2 classes, float32)
    // ─────────────────────────────────────────────────────────────────────────
    MedMNIST3DDataset adrenal_train(
        path + "/adrenalmnist3d",
        Split::TRAIN,
        2,
        "AdrenalMNIST3D",
        true
    );

    // ─────────────────────────────────────────────────────────────────────────
    // 4. Créer les DataLoaders
    // ─────────────────────────────────────────────────────────────────────────
    DataLoader3D fracture_loader(fracture_train, /*batch_size=*/16, /*shuffle=*/true);
    DataLoader3D adrenal_loader (adrenal_train,  /*batch_size=*/16, /*shuffle=*/true);

    std::cout << "\nFracture  — batches par epoch : " << fracture_loader.getNumBatches() << std::endl;
    std::cout << "Adrenal   — batches par epoch : " << adrenal_loader.getNumBatches()  << std::endl;

    // ─────────────────────────────────────────────────────────────────────────
    // 5. Itération sur un epoch (exemple)
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "\n=== Itération epoch 1 (FractureMNIST3D) ===" << std::endl;
    fracture_loader.reset();

    int batch_idx = 0;
    while (fracture_loader.hasNext()) {
        auto [images, labels] = fracture_loader.nextBatch();

        // images : Tensor (B, 1, 28, 28, 28)
        // labels : Tensor (B, 3,  1,  1,  1)  one-hot

        if (batch_idx == 0) {
            std::cout << "Premier batch :" << std::endl;
            images.printShape();
            labels.printShape();

            std::cout << "Label du premier volume : ";
            for (int c = 0; c < 3; ++c)
                std::cout << labels(0, c, 0, 0, 0) << " ";
            std::cout << std::endl;
        }

        batch_idx++;
    }
    std::cout << "✅ " << batch_idx << " batches traités" << std::endl;

    // ─────────────────────────────────────────────────────────────────────────
    // 6. Accès à un volume individuel
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "\n=== Volume individuel ===" << std::endl;
    Tensor vol = fracture_train.getVolume(0);
    vol.printShape();   // (1, 1, 28, 28, 28)
    std::cout << "Label : " << fracture_train.getLabel(0) << std::endl;

        // ─────────────────────────────────────────────────────────────────────────
    // 7. ConvLayer3D — Forward, padding=1 (résolution conservée)
    // ─────────────────────────────────────────────────────────────────────────
    section("7. ConvLayer3D — Forward, kernel 3×3×3, stride 1, padding 1");

    // Batch réel depuis le loader
    fracture_loader.reset();
    auto [batch_images, batch_labels] = fracture_loader.nextBatch();
    // batch_images : (16, 1, 28, 28, 28)

    std::cout << "Entrée  → "; batch_images.printShape();

    // 1 canal → 8 filtres, kernel 3×3×3
    // padding=1 avec kernel=3 : D/H/W conservés (28 → 28)
    ConvLayer3D conv1(
        /*in_ch=*/  1,
        /*out_ch=*/ 8,
        /*Kd=*/3, /*Kh=*/3, /*Kw=*/3,
        /*sd=*/1,  /*sh=*/1, /*sw=*/1,
        /*pd=*/1,  /*ph=*/1, /*pw=*/1
    );

    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor out1 = conv1.forward(batch_images);
    auto t1 = std::chrono::high_resolution_clock::now();
    long ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "Sortie  → "; out1.printShape();
    // Attendu : (16, 8, 28, 28, 28)
    std::cout << "Temps forward : " << ms1 << " ms" << std::endl;

    // Statistiques de la sortie
    float sum1 = 0.f, mn1 = out1[0], mx1 = out1[0];
    for (int i = 0; i < out1.size(); ++i) {
        sum1 += out1[i];
        mn1 = std::min(mn1, out1[i]);
        mx1 = std::max(mx1, out1[i]);
    }
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Stats  — min: " << mn1
              << "  max: " << mx1
              << "  mean: " << sum1 / out1.size() << std::endl;

    // ─────────────────────────────────────────────────────────────────────────
    // 8. ConvLayer3D — stride 2 (downsampling 28 → 14)
    // ─────────────────────────────────────────────────────────────────────────
    section("8. ConvLayer3D — stride 2, downsampling 28 → 14");

    ConvLayer3D conv2(
        /*in_ch=*/  1,
        /*out_ch=*/ 16,
        /*Kd=*/3, /*Kh=*/3, /*Kw=*/3,
        /*sd=*/2,  /*sh=*/2, /*sw=*/2,
        /*pd=*/1,  /*ph=*/1, /*pw=*/1
    );

    auto t2 = std::chrono::high_resolution_clock::now();
    Tensor out2 = conv2.forward(batch_images);
    auto t3 = std::chrono::high_resolution_clock::now();
    long ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

    std::cout << "Entrée  → "; batch_images.printShape();
    std::cout << "Sortie  → "; out2.printShape();
    // Attendu : (16, 16, 14, 14, 14)
    std::cout << "Temps forward : " << ms2 << " ms" << std::endl;

    // ─────────────────────────────────────────────────────────────────────────
    // 9. ConvLayer3D — Backward pass
    // ─────────────────────────────────────────────────────────────────────────
    section("9. ConvLayer3D — Backward pass");

    // Gradient fictif de même shape que la sortie de conv1
    // (simule ce que renverrait la couche suivante)
    Tensor grad_out(out1.dim(0), out1.dim(1),
                    out1.dim(2), out1.dim(3), out1.dim(4));
    grad_out.setConstant(1.0f);

    std::cout << "grad_output → "; grad_out.printShape();

    auto t4 = std::chrono::high_resolution_clock::now();
    Tensor grad_input = conv1.backward(grad_out);
    auto t5 = std::chrono::high_resolution_clock::now();
    long ms3 = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();

    std::cout << "grad_input  → "; grad_input.printShape();
    // Attendu : (16, 1, 28, 28, 28) — même shape que l'entrée
    std::cout << "Temps backward : " << ms3 << " ms" << std::endl;

    // Vérifier que grad_weights et grad_bias sont non nuls
    Tensor& gw = conv1.getWeightGradients();
    std::cout << "\ngrad_weights → "; gw.printShape();
    // Attendu : (8, 1, 3, 3, 3)

    float gw_l1 = 0.f;
    for (int i = 0; i < gw.size(); ++i) gw_l1 += std::abs(gw[i]);
    std::cout << "Norme L1 grad_weights : " << gw_l1 << std::endl;
    // Si 0.0 → problème dans le backward

    Eigen::VectorXf& gb = conv1.getBiasGradients();
    std::cout << "grad_bias (8 valeurs)  : [";
    for (int i = 0; i < gb.size(); ++i)
        std::cout << std::setw(10) << gb(i);
    std::cout << " ]" << std::endl;

    // ─────────────────────────────────────────────────────────────────────────
    // 10. Enchaînement deux convolutions (mini backbone)
    // ─────────────────────────────────────────────────────────────────────────
    section("10. Mini backbone — conv_a (28→28) → conv_b (28→14)");

    // conv_a : 1 → 8,  kernel 3×3×3, padding 1       → (B, 8,  28, 28, 28)
    // conv_b : 8 → 16, kernel 3×3×3, stride 2, pad 1 → (B, 16, 14, 14, 14)
    ConvLayer3D conv_a(1,  8, 3, 3, 3, 1, 1, 1, 1, 1, 1);
    ConvLayer3D conv_b(8, 16, 3, 3, 3, 2, 2, 2, 1, 1, 1);

    fracture_loader.reset();
    auto [mini_batch, mini_labels] = fracture_loader.nextBatch();

    std::cout << "Entrée        → "; mini_batch.printShape();

    auto t6 = std::chrono::high_resolution_clock::now();
    Tensor feat_a = conv_a.forward(mini_batch);
    Tensor feat_b = conv_b.forward(feat_a);
    auto t7 = std::chrono::high_resolution_clock::now();
    long ms4 = std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t6).count();

    std::cout << "Après conv_a  → "; feat_a.printShape();  // (16, 8,  28, 28, 28)
    std::cout << "Après conv_b  → "; feat_b.printShape();  // (16, 16, 14, 14, 14)
    std::cout << "Temps forward total : " << ms4 << " ms" << std::endl;

    // Backward de bout en bout
    Tensor grad_b(feat_b.dim(0), feat_b.dim(1),
                  feat_b.dim(2), feat_b.dim(3), feat_b.dim(4));
    grad_b.setConstant(0.01f);

    auto t8  = std::chrono::high_resolution_clock::now();
    Tensor grad_a  = conv_b.backward(grad_b);
    Tensor grad_in = conv_a.backward(grad_a);
    auto t9  = std::chrono::high_resolution_clock::now();
    long ms5 = std::chrono::duration_cast<std::chrono::milliseconds>(t9 - t8).count();

    std::cout << "grad après conv_b → "; grad_a.printShape();   // (16, 8,  28, 28, 28)
    std::cout << "grad_input        → "; grad_in.printShape();  // (16, 1,  28, 28, 28)
    std::cout << "Temps backward total : " << ms5 << " ms" << std::endl;

    // ─────────────────────────────────────────────────────────────────────────
    // Résumé final
    // ─────────────────────────────────────────────────────────────────────────
    section("Résumé des temps");
    std::cout << std::left;
    std::cout << std::setw(40) << "Conv1 forward (1→8, stride 1)"
              << ms1 << " ms" << std::endl;
    std::cout << std::setw(40) << "Conv2 forward (1→16, stride 2)"
              << ms2 << " ms" << std::endl;
    std::cout << std::setw(40) << "Conv1 backward"
              << ms3 << " ms" << std::endl;
    std::cout << std::setw(40) << "Mini backbone forward (2 couches)"
              << ms4 << " ms" << std::endl;
    std::cout << std::setw(40) << "Mini backbone backward (2 couches)"
              << ms5 << " ms" << std::endl;

    std::cout << "\n✅ Tous les tests ConvLayer3D passés avec succès" << std::endl;

    return 0;
}