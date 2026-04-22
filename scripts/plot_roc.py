import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

# Paramètres
CSV_PATH = "logs/predictions.csv"
OUTPUT_PATH = "docs/results/roc_curve.png"
CLASSES = ["No fracture", "Fracture T1", "Fracture T2"]
N_CLASSES = len(CLASSES)

def plot_roc_auc():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Please run the C++ training/evaluation first.")
        return

    # 1. Charger les données
    df = pd.read_csv(CSV_PATH)
    y_true = df['true_label'].values
    
    # Extraire les probs
    y_score = np.zeros((len(df), N_CLASSES))
    for i in range(N_CLASSES):
        y_score[:, i] = df[f'prob_class_{i}'].values

    # Binariser les labels vrais
    y_test = label_binarize(y_true, classes=range(N_CLASSES))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(N_CLASSES)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(N_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= N_CLASSES

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 2. Tracer
    plt.figure(figsize=(10, 8))
    
    # Couleurs personnalisées (vibrantes, qualité article)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Tracer les classes individuelles
    for i, color in zip(range(N_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {CLASSES[i]} (area = {roc_auc[i]:.2f})')

    # Tracer micro-average
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    # Tracer macro-average
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'macro-average ROC curve (area = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)

    # Ligne diagonale (random)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) - Multi-Class', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Enregistrer
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"ROC Curve successfully saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    plot_roc_auc()
