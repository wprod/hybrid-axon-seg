#!/bin/bash
# start_vacation.sh — mode vacances : redémarrage auto, empêche le sleep, notif ntfy
#
# Usage:
#   tmux new -s axon './start_vacation.sh'
#   # puis Ctrl-B D pour détacher
#
# Pour arrêter :
#   tmux kill-session -t axon

set -uo pipefail

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$WORKDIR/vacation_logs"
mkdir -p "$LOGDIR"

APP_LOG="$LOGDIR/app.log"
CF_LOG="$LOGDIR/cf.log"
URL_FILE="$LOGDIR/url.txt"

# --- Mot de passe fixe (à définir avant de partir) ---
export APP_PASSWORD="${APP_PASSWORD:?Définis APP_PASSWORD avant de lancer: APP_PASSWORD=xxx ./start_vacation.sh}"

# --- ntfy topic (installer l'app ntfy et s'abonner à ce topic) ---
NTFY_TOPIC="${NTFY_TOPIC:-axon-seg-labo}"

PYTHON="/Users/amand/.pyenv/versions/3.12.13/bin/python3"
CLOUDFLARED="/opt/homebrew/bin/cloudflared"

notify() {
    local title="$1"
    local msg="$2"
    curl -s \
        -H "Title: $title" \
        -H "Priority: high" \
        -H "Tags: computer" \
        -d "$msg" \
        "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1
}

cleanup() {
    echo "[$(date)] Arrêt..."
    kill "$APP_PID" "$CF_PID" "$CAFFEINE_PID" 2>/dev/null
    wait 2>/dev/null
    exit 0
}
trap cleanup INT TERM

# --- Empêcher le Mac de dormir ---
caffeinate -dims &
CAFFEINE_PID=$!
echo "[$(date)] caffeinate actif (PID $CAFFEINE_PID) — le Mac ne dormira pas"

# --- Boucle principale ---
while true; do
    echo ""
    echo "[$(date)] ====== Démarrage des services ======"

    # --- app.py ---
    cd "$WORKDIR"
    PYTHONUNBUFFERED=1 "$PYTHON" app.py >>"$APP_LOG" 2>&1 &
    APP_PID=$!
    echo "[$(date)] app.py lancé (PID $APP_PID)"

    # Attendre que app.py soit prêt
    READY=false
    for i in $(seq 1 30); do
        if curl -s -o /dev/null http://127.0.0.1:8000/health 2>/dev/null || \
           curl -s -o /dev/null http://127.0.0.1:8000/ 2>/dev/null; then
            READY=true
            break
        fi
        if ! kill -0 "$APP_PID" 2>/dev/null; then
            echo "[$(date)] ERREUR: app.py a crashé au démarrage"
            break
        fi
        sleep 1
    done

    if [ "$READY" = false ]; then
        echo "[$(date)] app.py pas prêt, retry dans 10s..."
        kill "$APP_PID" 2>/dev/null
        sleep 10
        continue
    fi

    echo "[$(date)] app.py prêt sur :8000"

    # --- cloudflared (log frais à chaque restart pour éviter de matcher l'ancienne URL) ---
    : > "$CF_LOG"
    "$CLOUDFLARED" tunnel --url http://127.0.0.1:8000 >>"$CF_LOG" 2>&1 &
    CF_PID=$!
    echo "[$(date)] cloudflared lancé (PID $CF_PID)"

    # Récupérer l'URL
    URL=""
    for i in $(seq 1 60); do
        URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" 2>/dev/null | tail -1)
        if [ -n "${URL:-}" ]; then
            break
        fi
        if ! kill -0 "$CF_PID" 2>/dev/null; then
            echo "[$(date)] ERREUR: cloudflared a crashé au démarrage"
            break
        fi
        sleep 1
    done

    if [ -n "${URL:-}" ]; then
        echo "$URL" > "$URL_FILE"
        echo ""
        echo "  ┌─────────────────────────────────────────────────────┐"
        echo "  │  SERVICES ACTIFS                                    │"
        echo "  ├─────────────────────────────────────────────────────┤"
        printf "  │  URL:       %-39s│\n" "$URL"
        echo "  │  User:      axon                                    │"
        printf "  │  Password:  %-39s│\n" "$APP_PASSWORD"
        echo "  └─────────────────────────────────────────────────────┘"
        echo ""

        # Copier dans le presse-papiers (utile au premier lancement)
        printf '%s\n%s\n%s' "$URL" "axon" "$APP_PASSWORD" | pbcopy 2>/dev/null

        # Notification push
        notify "axon-seg en ligne" "$URL"
    else
        echo "[$(date)] Pas d'URL cloudflared, retry dans 10s..."
        kill "$APP_PID" "$CF_PID" 2>/dev/null
        sleep 10
        continue
    fi

    # --- Surveiller les deux processus ---
    LAST_DAILY_RESTART=$(date +%Y-%m-%d)
    echo "[$(date)] Surveillance active. Redémarrage quotidien à 16h."
    while true; do
        if ! kill -0 "$APP_PID" 2>/dev/null; then
            echo "[$(date)] app.py est mort ! Redémarrage..."
            kill "$CF_PID" 2>/dev/null
            sleep 3
            break
        fi
        if ! kill -0 "$CF_PID" 2>/dev/null; then
            echo "[$(date)] cloudflared est mort ! Redémarrage..."
            kill "$APP_PID" 2>/dev/null
            sleep 3
            break
        fi
        # Redémarrage quotidien à 16h
        CURRENT_HOUR=$(date +%H)
        CURRENT_DATE=$(date +%Y-%m-%d)
        if [ "$CURRENT_HOUR" = "16" ] && [ "$CURRENT_DATE" != "$LAST_DAILY_RESTART" ]; then
            echo "[$(date)] Redémarrage quotidien programmé..."
            kill "$APP_PID" "$CF_PID" 2>/dev/null
            LAST_DAILY_RESTART="$CURRENT_DATE"
            sleep 3
            break
        fi
        sleep 30
    done
done
