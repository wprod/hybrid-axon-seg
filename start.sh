#!/bin/bash
# start.sh — lance app.py + cloudflared, copie les identifiants dans le presse-papiers

set -uo pipefail

LOGDIR=$(mktemp -d)
APP_LOG="$LOGDIR/app.log"
CF_LOG="$LOGDIR/cf.log"

echo ""
echo "  Démarrage de app.py..."
PYTHONUNBUFFERED=1 python app.py >"$APP_LOG" 2>&1 &
APP_PID=$!

# Attendre que le mot de passe apparaisse dans les logs
for i in $(seq 1 60); do
    if grep -q "password:" "$APP_LOG" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$APP_PID" 2>/dev/null; then
        echo "  ERREUR: app.py s'est arrêté. Logs: $APP_LOG"
        cat "$APP_LOG"
        exit 1
    fi
    sleep 0.5
done

PASSWORD=$(grep "password:" "$APP_LOG" | sed 's/.*password: //' | tr -d ' \r\n')
if [ -z "${PASSWORD:-}" ]; then
    echo "  ERREUR: mot de passe introuvable dans les logs. Vérifier: $APP_LOG"
    exit 1
fi

echo "  app.py démarré (PID $APP_PID)"
echo ""
echo "  Démarrage du tunnel cloudflared..."
cloudflared tunnel --url http://127.0.0.1:8000 >"$CF_LOG" 2>&1 &
CF_PID=$!

# Attendre que l'URL apparaisse dans les logs cloudflared
URL=""
for i in $(seq 1 60); do
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" 2>/dev/null | head -1)
    if [ -n "${URL:-}" ]; then
        break
    fi
    if ! kill -0 "$CF_PID" 2>/dev/null; then
        echo "  ERREUR: cloudflared s'est arrêté. Logs: $CF_LOG"
        cat "$CF_LOG"
        exit 1
    fi
    sleep 1
done

if [ -z "${URL:-}" ]; then
    echo "  ERREUR: URL cloudflared introuvable. Vérifier: $CF_LOG"
    exit 1
fi

echo "  Tunnel cloudflared actif (PID $CF_PID)"
echo ""

# Copier dans le presse-papiers
CLIPBOARD="$URL
axon
$PASSWORD"
printf '%s' "$CLIPBOARD" | pbcopy

echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │  Copié dans le presse-papiers !                     │"
echo "  ├─────────────────────────────────────────────────────┤"
printf "  │  URL:       %-39s│\n" "$URL"
echo "  │  User:      axon                                    │"
printf "  │  Password:  %-39s│\n" "$PASSWORD"
echo "  └─────────────────────────────────────────────────────┘"
echo ""
echo "  Logs: $LOGDIR/"
echo "  Pour arrêter: kill $APP_PID $CF_PID"
echo ""
