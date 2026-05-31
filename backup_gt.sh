#!/bin/bash
# backup_gt.sh — sauvegarde des annotations ground truth + modèle entraîné

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="ground_truth_backup_${TIMESTAMP}.zip"

zip -r "$BACKUP" ground_truth/

# Ajouter best.pt et last.pt si disponibles
if [ -f train/checkpoints/best.pt ]; then
    zip "$BACKUP" train/checkpoints/best.pt
fi
if [ -f train/checkpoints/last.pt ]; then
    zip "$BACKUP" train/checkpoints/last.pt
fi

echo ""
echo "  Sauvegarde créée: $BACKUP"
echo "  Taille: $(du -h "$BACKUP" | cut -f1)"
echo ""
