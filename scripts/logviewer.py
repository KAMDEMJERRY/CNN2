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
        
        # 2 graphiques au lieu de 4
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.suptitle('Training Progress - Live Update', fontsize=16)
        
        # Graphique des losses (train + val)
        self.ax_loss.set_title('Loss Curves')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        # Graphique des accuracies (train + val)
        self.ax_acc.set_title('Accuracy Curves')
        self.ax_acc.set_xlabel('Epoch')
        self.ax_acc.set_ylabel('Accuracy (%)')
        self.ax_acc.grid(True, alpha=0.3)
        
        # Lignes pour les graphiques
        self.line_train_loss, = self.ax_loss.plot([], [], 'b-', label='Training Loss', linewidth=2)
        self.line_val_loss, = self.ax_loss.plot([], [], 'r-', label='Validation Loss', linewidth=2)
        
        self.line_train_acc, = self.ax_acc.plot([], [], 'b-', label='Training Accuracy', linewidth=2)
        self.line_val_acc, = self.ax_acc.plot([], [], 'r-', label='Validation Accuracy', linewidth=2)
        
        # Ajouter des légendes
        self.ax_loss.legend(loc='upper right')
        self.ax_acc.legend(loc='lower right')
        
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
                self.line_val_loss.set_data(epochs, val_losses)
                self.line_train_acc.set_data(epochs, train_accs)
                self.line_val_acc.set_data(epochs, val_accs)
                
                # Ajuster les limites des axes
                # Pour le graphique des losses
                all_losses = train_losses + val_losses
                if all_losses:
                    self.ax_loss.set_xlim(0, max(epochs) + 1)
                    self.ax_loss.set_ylim(0, max(all_losses) * 1.1)
                
                # Pour le graphique des accuracies
                all_accs = train_accs + val_accs
                if all_accs:
                    self.ax_acc.set_xlim(0, max(epochs) + 1)
                    self.ax_acc.set_ylim(0, 105)
                
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
        return [self.line_train_loss, self.line_val_loss, 
                self.line_train_acc, self.line_val_acc]
    
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
    
    def save_final_plot(self, output_file='final_training_plot.png'):
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
        
        # Créer les graphiques (2 subplots)
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Training Results - Last Session', fontsize=16)
        
        # Graphique des losses
        ax_loss.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax_loss.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Loss Curves')
        ax_loss.legend(loc='upper right')
        ax_loss.grid(True, alpha=0.3)
        
        # Ajouter le point du meilleur loss validation
        best_val_loss_idx = val_losses.index(min(val_losses))
        best_val_loss = val_losses[best_val_loss_idx]
        ax_loss.plot(best_val_loss_idx + 1, best_val_loss, 'go', markersize=10, 
                    label=f'Best Val Loss: {best_val_loss:.4f}')
        ax_loss.legend(loc='upper right')
        
        # Graphique des accuracies
        ax_acc.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
        ax_acc.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.set_title('Accuracy Curves')
        ax_acc.legend(loc='lower right')
        ax_acc.grid(True, alpha=0.3)
        
        # Ajouter le point de la meilleure accuracy validation
        best_val_acc_idx = val_accs.index(max(val_accs))
        best_val_acc = val_accs[best_val_acc_idx]
        ax_acc.plot(best_val_acc_idx + 1, best_val_acc, 'go', markersize=10,
                   label=f'Best Val Acc: {best_val_acc:.2f}%')
        ax_acc.legend(loc='lower right')
        
        # Ajouter une grille et ajuster les limites
        ax_acc.set_ylim(0, 105)
        
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
        
        best_val_loss = min(val_losses)
        best_loss_epoch = val_losses.index(best_val_loss) + 1
        
        print("\n" + "="*60)
        print("RÉSUMÉ DU DERNIER ENTRAÎNEMENT")
        print("="*60)
        print(f"Nombre d'epochs: {len(epochs)}")
        print(f"\nMeilleure accuracy validation: {best_val_acc:.2f}% (epoch {best_epoch})")
        print(f"Accuracy entraînement correspondante: {best_train_acc:.2f}%")
        print(f"\nMeilleure loss validation: {best_val_loss:.4f} (epoch {best_loss_epoch})")
        print(f"\nMétriques finales:")
        print(f"  - Loss - Entraînement: {train_losses[-1]:.4f}, Validation: {val_losses[-1]:.4f}")
        print(f"  - Accuracy - Entraînement: {train_accs[-1]:.2f}%, Validation: {val_accs[-1]:.2f}%")
        
        # Détecter le surapprentissage
        if len(val_losses) > 10:
            recent_losses = val_losses[-5:]
            if recent_losses[-1] > min(val_losses) * 1.1:
                print("\n⚠️  ATTENTION: Possible surapprentissage détecté!")
                print(f"   La loss de validation augmente: {min(val_losses):.4f} → {val_losses[-1]:.4f}")
        print("="*60)


def main(logfilepath="./logs/training_log.txt"):
    # Configuration
    LOG_FILE = logfilepath
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
    filepath = "logs/training_log.txt"
    main()