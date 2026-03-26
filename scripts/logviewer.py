import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import os
import time

class TrainingLogger:
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.epochs = []
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
    def parse_epoch_line(self, line):
        """Parse une ligne d'epoch et extrait les métriques"""
        # Pattern pour les logs avec Time
        pattern_time = r'Epoch\s+(\d+)/(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Acc:\s+([\d.]+)%\s+\|\s+Time:\s+(\d+)ms\s+\|\s+Val Loss:\s+([\d.]+)\s+\|\s+Val Acc:\s+([\d.]+)%'
        # Pattern pour les logs sans Time
        pattern_no_time = r'Epoch\s+(\d+)/(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Acc:\s+([\d.]+)%\s+\|\s+Val Loss:\s+([\d.]+)\s+\|\s+Val Acc:\s+([\d.]+)%'
        
        match = re.search(pattern_time, line)
        if not match:
            match = re.search(pattern_no_time, line)
        
        if match:
            groups = match.groups()
            epoch = int(groups[0])
            train_loss = float(groups[2])
            train_acc = float(groups[3])
            
            if len(groups) == 7:  # Avec Time
                val_loss = float(groups[5])
                val_acc = float(groups[6])
            else:  # Sans Time
                val_loss = float(groups[4])
                val_acc = float(groups[5])
            
            return epoch, train_loss, train_acc, val_loss, val_acc
        return None
    
    def find_last_training(self):
        """Trouve le dernier entraînement dans le fichier de log"""
        if not os.path.exists(self.log_file):
            print(f"Fichier {self.log_file} non trouvé")
            return None
        
        with open(self.log_file, 'r') as f:
            content = f.read()
        
        # Trouver toutes les dates dans le fichier
        date_pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})'
        dates = re.findall(date_pattern, content)
        
        if not dates:
            print("Aucune date trouvée dans le fichier")
            return None
        
        # Prendre la dernière date
        last_date = dates[-1]
        print(f"Dernier entraînement trouvé: {last_date}")
        
        # Extraire la section après la dernière date
        last_date_pattern = re.escape(last_date) + r'\s*={3,}\s*\n(.*?)(?=\n\d{4}-\d{2}-\d{2}|\Z)'
        match = re.search(last_date_pattern, content, re.DOTALL)
        
        if match:
            return match.group(1)
        return None
    
    def parse_training_section(self, section):
        """Parse une section d'entraînement"""
        self.epochs = []
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
        lines = section.split('\n')
        for line in lines:
            parsed = self.parse_epoch_line(line)
            if parsed:
                epoch, train_loss, train_acc, val_loss, val_acc = parsed
                self.epochs.append(epoch)
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
        
        return len(self.epochs) > 0
    
    def load_last_training(self):
        """Charge le dernier entraînement"""
        section = self.find_last_training()
        if section:
            return self.parse_training_section(section)
        return False


class LivePlotter:
    def __init__(self, log_file, refresh_interval=5):
        self.log_file = log_file
        self.refresh_interval = refresh_interval
        self.logger = TrainingLogger(log_file)
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Training Progress - Live Update', fontsize=16)
        
        # Configurer les subplots
        self.ax_loss = self.axes[0, 0]
        self.ax_acc = self.axes[0, 1]
        self.ax_val_loss = self.axes[1, 0]
        self.ax_val_acc = self.axes[1, 1]
        
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        self.ax_acc.set_title('Training Accuracy')
        self.ax_acc.set_xlabel('Epoch')
        self.ax_acc.set_ylabel('Accuracy (%)')
        self.ax_acc.grid(True, alpha=0.3)
        
        self.ax_val_loss.set_title('Validation Loss')
        self.ax_val_loss.set_xlabel('Epoch')
        self.ax_val_loss.set_ylabel('Loss')
        self.ax_val_loss.grid(True, alpha=0.3)
        
        self.ax_val_acc.set_title('Validation Accuracy')
        self.ax_val_acc.set_xlabel('Epoch')
        self.ax_val_acc.set_ylabel('Accuracy (%)')
        self.ax_val_acc.grid(True, alpha=0.3)
        
        # Lignes pour les graphiques
        self.line_train_loss, = self.ax_loss.plot([], [], 'b-', label='Training Loss', linewidth=2)
        self.line_train_acc, = self.ax_acc.plot([], [], 'b-', label='Training Accuracy', linewidth=2)
        self.line_val_loss, = self.ax_val_loss.plot([], [], 'r-', label='Validation Loss', linewidth=2)
        self.line_val_acc, = self.ax_val_acc.plot([], [], 'r-', label='Validation Accuracy', linewidth=2)
        
        # Ajouter des légendes
        self.ax_loss.legend()
        self.ax_acc.legend()
        self.ax_val_loss.legend()
        self.ax_val_acc.legend()
        
        plt.tight_layout()
        
    def update_plots(self):
        """Met à jour les graphiques avec les dernières données"""
        if self.logger.load_last_training():
            epochs = self.logger.epochs
            train_losses = self.logger.train_losses
            train_accs = self.logger.train_accs
            val_losses = self.logger.val_losses
            val_accs = self.logger.val_accs
            
            if epochs:
                # Mettre à jour les données des graphiques
                self.line_train_loss.set_data(epochs, train_losses)
                self.line_train_acc.set_data(epochs, train_accs)
                self.line_val_loss.set_data(epochs, val_losses)
                self.line_val_acc.set_data(epochs, val_accs)
                
                # Ajuster les limites des axes
                for ax, data in [(self.ax_loss, train_losses + val_losses),
                                 (self.ax_acc, train_accs + val_accs),
                                 (self.ax_val_loss, val_losses),
                                 (self.ax_val_acc, val_accs)]:
                    if data:
                        ax.set_xlim(0, max(epochs) + 1)
                        if ax == self.ax_loss or ax == self.ax_val_loss:
                            ax.set_ylim(0, max(data) * 1.1)
                        else:
                            ax.set_ylim(0, 105)
                
                # Mettre à jour le titre avec les dernières métriques
                last_epoch = epochs[-1]
                last_train_loss = train_losses[-1]
                last_train_acc = train_accs[-1]
                last_val_loss = val_losses[-1] if val_losses else 0
                last_val_acc = val_accs[-1] if val_accs else 0
                
                self.fig.suptitle(f'Training Progress - Epoch {last_epoch} | '
                                 f'Train Loss: {last_train_loss:.4f} | '
                                 f'Train Acc: {last_train_acc:.2f}% | '
                                 f'Val Loss: {last_val_loss:.4f} | '
                                 f'Val Acc: {last_val_acc:.2f}%', 
                                 fontsize=12)
                
                return True
        return False
    
    def animate(self, i):
        """Fonction d'animation appelée périodiquement"""
        self.update_plots()
        return [self.line_train_loss, self.line_train_acc, 
                self.line_val_loss, self.line_val_acc]
    
    def start(self):
        """Démarre l'affichage en direct"""
        print(f"Surveillance du fichier: {self.log_file}")
        print("Mise à jour toutes les", self.refresh_interval, "secondes")
        print("Appuyez sur Ctrl+C pour arrêter")
        
        # Charger les données initiales
        self.update_plots()
        
        # Démarrer l'animation
        ani = animation.FuncAnimation(self.fig, self.animate, 
                                      interval=self.refresh_interval * 1000,
                                      cache_frame_data=False)
        plt.show()
    
    def save_final_plot(self, output_file='training_plot.png'):
        """Sauvegarde le graphique final"""
        self.update_plots()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé: {output_file}")


class TrainingMonitor:
    def __init__(self, log_file, output_dir='plots'):
        self.log_file = log_file
        self.output_dir = output_dir
        self.logger = TrainingLogger(log_file)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def generate_static_plots(self):
        """Génère des graphiques statiques à partir du dernier entraînement"""
        if not self.logger.load_last_training():
            print("Impossible de charger les données d'entraînement")
            return False
        
        epochs = self.logger.epochs
        train_losses = self.logger.train_losses
        train_accs = self.logger.train_accs
        val_losses = self.logger.val_losses
        val_accs = self.logger.val_accs
        
        if not epochs:
            print("Aucune donnée d'epoch trouvée")
            return False
        
        # Créer les graphiques
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Results - Last Session', fontsize=16)
        
        # Loss
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training Loss (détail)
        axes[1, 0].plot(epochs, train_losses, 'b-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss Details')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Validation Accuracy (détail)
        axes[1, 1].plot(epochs, val_accs, 'r-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Validation Accuracy Details')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Ajouter les meilleures valeurs
        best_val_acc = max(val_accs)
        best_epoch = val_accs.index(best_val_acc) + 1
        axes[1, 1].axhline(y=best_val_acc, color='g', linestyle='--', alpha=0.5)
        axes[1, 1].text(0.02, 0.98, f'Best: {best_val_acc:.2f}% at epoch {best_epoch}',
                        transform=axes[1, 1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Sauvegarder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.output_dir, f'training_plot_{timestamp}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé: {output_file}")
        
        plt.show()
        return True
    
    def print_summary(self):
        """Affiche un résumé du dernier entraînement"""
        if not self.logger.load_last_training():
            print("Impossible de charger les données d'entraînement")
            return
        
        epochs = self.logger.epochs
        train_accs = self.logger.train_accs
        val_accs = self.logger.val_accs
        train_losses = self.logger.train_losses
        val_losses = self.logger.val_losses
        
        if not epochs:
            return
        
        best_val_acc = max(val_accs)
        best_epoch = val_accs.index(best_val_acc) + 1
        best_train_acc = train_accs[best_epoch - 1]
        
        print("\n" + "="*60)
        print("RÉSUMÉ DU DERNIER ENTRAÎNEMENT")
        print("="*60)
        print(f"Nombre d'epochs: {len(epochs)}")
        print(f"Meilleure accuracy validation: {best_val_acc:.2f}% (epoch {best_epoch})")
        print(f"Accuracy entraînement correspondante: {best_train_acc:.2f}%")
        print(f"Loss finale - Entraînement: {train_losses[-1]:.4f}, Validation: {val_losses[-1]:.4f}")
        print(f"Accuracy finale - Entraînement: {train_accs[-1]:.2f}%, Validation: {val_accs[-1]:.2f}%")
        
        # Détecter le surapprentissage
        overfitting = val_losses[-1] > min(val_losses) * 1.2
        if overfitting:
            print("\n⚠️  ATTENTION: Possible surapprentissage détecté!")
            print(f"   La loss de validation augmente: {min(val_losses):.4f} → {val_losses[-1]:.4f}")
        print("="*60)


def main(logfilepath="training_log.txt"):
    # Configuration
    LOG_FILE = logfilepath  # Remplacez par le chemin de votre fichier de log
    MODE = "live"  # Options: "live", "static", "summary"
    
    if not os.path.exists(LOG_FILE):
        print(f"Erreur: Fichier {LOG_FILE} non trouvé")
        return
    
    if MODE == "live":
        # Mode en direct - met à jour les graphiques en temps réel
        plotter = LivePlotter(LOG_FILE, refresh_interval=5)
        try:
            plotter.start()
        except KeyboardInterrupt:
            print("\nArrêt demandé par l'utilisateur")
            plotter.save_final_plot('final_training_plot.png')
    
    elif MODE == "static":
        # Mode statique - génère un graphique final
        monitor = TrainingMonitor(LOG_FILE)
        monitor.generate_static_plots()
        monitor.print_summary()
    
    elif MODE == "summary":
        # Mode résumé - affiche seulement les statistiques
        monitor = TrainingMonitor(LOG_FILE)
        monitor.print_summary()


if __name__ == "__main__":
    main("logs/training_log.txt")