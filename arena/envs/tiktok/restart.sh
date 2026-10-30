#!/bin/bash
# Restart the Next.js dev server before scoring, so the scorer sees the current source even if the agent left the server down.
# Wall-clock deadline under the scorers' 300 s restart timeout: an app that accepts connections and then hangs must end here as
# "app not ready" (AGENT, scored 0), not run into the scorer timeout (INFRA, excluded).
DEADLINE=240
pkill -f "next dev"; pkill -f next-server; sleep 1
cd /app/src && PORT=80 nohup pnpm run dev > /tmp/next-dev.log 2>&1 &
while (( SECONDS < DEADLINE )); do
    curl -sf -m 5 http://localhost:80/api/random-video | grep -q '"id"' && exit 0
    sleep 1
done
echo "app not ready ${SECONDS}s after restart" >&2
tail -30 /tmp/next-dev.log >&2
exit 1
