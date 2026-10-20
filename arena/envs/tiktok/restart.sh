#!/bin/bash
# Restart the Next.js dev server before scoring, so the scorer sees the current source even if the agent left the server down.
pkill -f "next dev"; pkill -f next-server; sleep 1
cd /app/src && PORT=80 nohup pnpm run dev > /tmp/next-dev.log 2>&1 &
for i in $(seq 1 150); do
    curl -sf http://localhost:80/api/random-video | grep -q '"id"' && exit 0
    sleep 1
done
echo "app not ready 150s after restart" >&2
tail -30 /tmp/next-dev.log >&2
exit 1
