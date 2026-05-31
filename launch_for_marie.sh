#!/bin/bash
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
APP_PASSWORD=$(cat ~/.axon_pwd)
NTFY_TOPIC=axon-marie-k7x9q2

# Tuer une éventuelle session existante
tmux kill-session -t axon 2>/dev/null

# Lancer en arrière-plan (détaché)
tmux new-session -d -s axon \
    -e "APP_PASSWORD=$APP_PASSWORD" \
    -e "NTFY_TOPIC=$NTFY_TOPIC" \
    "cd $WORKDIR && ./start_vacation.sh"

echo "Services lancés en arrière-plan."
echo "Pour voir les logs : tmux attach -t axon"
echo "Pour arrêter      : tmux kill-session -t axon"
