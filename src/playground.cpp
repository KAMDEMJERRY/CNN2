// // #include "shared.hpp"
// #include "CNNLIB.hpp"
// #include <chrono>


// #include <iostream>
// #include <Eigen/Dense>

// using namespace std;
// using namespace Eigen;

// void test_performance() {
//     const int size = 10000;

//     // 1. Déclaration d'une matrice Col-Major (Défaut Eigen)
//     Matrix<float, Dynamic, Dynamic, ColMajor> col_mat(size, size);

//     // 2. Déclaration d'une matrice Row-Major (Style C / PyTorch)
//     Matrix<float, Dynamic, Dynamic, RowMajor> row_mat(size, size);

//     // Initialisation
//     col_mat.setRandom();
//     row_mat.setRandom();

//     float sum = 0;

//     // --- TEST 1 : Parcours Col-Major (Optimisé pour le défaut Eigen) ---
//     auto start = chrono::high_resolution_clock::now();
//     for (int j = 0; j < size; ++j) {
//         for (int i = 0; i < size; ++i) {
//             sum += col_mat(i, j); // Premier indice (i) change le plus vite
//         }
//     }
//     auto end = chrono::high_resolution_clock::now();
//     cout << "Col-Major (Parcours Col) : " 
//          << chrono::duration<double>(end - start).count() << "s" << endl;

//     // --- TEST 2 : Parcours Row-Major (Optimisé pour RowMajor) ---
//     start = chrono::high_resolution_clock::now();
//     for (int i = 0; i < size; ++i) {
//         for (int j = 0; j < size; ++j) {
//             sum += row_mat(i, j); // Dernier indice (j) change le plus vite
//         }
//     }
//     end = chrono::high_resolution_clock::now();
//     cout << "Row-Major (Parcours Row) : " 
//          << chrono::duration<double>(end - start).count() << "s" << endl;

//     // --- TEST 3 : LA CATASTROPHE (Parcours Row sur matrice Col-Major) ---
//     start = chrono::high_resolution_clock::now();
//     for (int i = 0; i < size; ++i) {
//         for (int j = 0; j < size; ++j) {
//             sum += col_mat(i, j); // i est fixe, on saute des colonnes entières !
//         }
//     }
//     end = chrono::high_resolution_clock::now();
//     cout << "Col-Major (MAUVAIS PARCOURS) : " 
//          << chrono::duration<double>(end - start).count() << "s" << endl;
// }

// int main() {
//     test_performance();
//     return 0;
// }


#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;
using namespace Eigen;

void visualiser_mapping() {
    // 1. Création d'un petit tenseur 3D (Profondeur=2, Lignes=2, Cols=2)
    // Par défaut, Eigen::Tensor est COL-MAJOR
    Tensor<float, 3> t_col(2, 2, 2);

    // Remplissage séquentiel : 0, 1, 2, 3, 4, 5, 6, 7
    for (int i = 0; i < 8; ++i) t_col.data()[i] = i;

    // 2. Vérification du Layout
    cout << "--- VERIFICATION LAYOUT ---" << endl;
    if (t_col.Layout == Eigen::ColMajor) cout << "Le tenseur est COL-MAJOR (Défaut)" << endl;
    else cout << "Le tenseur est ROW-MAJOR" << endl;

    // 3. Visualisation du Mapping vers une Matrice (2x4)
    // On prend le tenseur (2,2,2) et on le voit comme une Matrice (2,4)
    Map<MatrixXf> m_col(t_col.data(), 2, 4);

    cout << "\n--- VISUALISATION MATRICE (Map ColMajor) ---" << endl;
    cout << m_col << endl;
    /*
       Interprétation attendue (ColMajor) :
       Chaque colonne est remplie en premier.
       0 2 4 6
       1 3 5 7  <-- Le premier indice (0,1) change le plus vite
    */

    // 4. Test avec Row-Major (Simulation de données C-style/Torch)
    Tensor<float, 3, RowMajor> t_row(2, 2, 2);
    for (int i = 0; i < 8; ++i) t_row.data()[i] = i;

    // On doit utiliser un Map qui accepte le Row2Major
    using RowMatrixMap = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>;
    RowMatrixMap m_row(t_row.data(), 2, 4);

    cout << "\n--- VISUALISATION MATRICE (Map RowMajor) ---" << endl;
    cout << m_row << endl;
    /*
       Interprétation attendue (RowMajor) :
       Chaque ligne est remplie en premier.
       0 1 2 3
       4 5 6 7  <-- Le dernier indice change le plus vite
    */
}

int main() {

    int k = 0;
    std::cin >> k;
    while (k--) {
        int a, b, c, d;
        cout << "Entrez 4 entiers : ";
        std::cin >> a >> b >> c >> d;

        if (a == b && b == c && c == d) {
            cout << "YES" << endl;
        }
        else {
            cout << "NO" << endl;
        }
    }

    return 0;
}
