#!/bin/bash
# Exit 0 once the Node API answers on :3000. compose runs /app/restart.sh (redis, postgres, vsftpd,
# bull worker, node API) in the background; /api/health returns 200 with a "status" field as soon as
# Express is listening (it reports "degraded" until redis/db connect, but still 200). Not :80 or
# /api/random-video -- those are tiktok.
curl -sf -m 10 http://localhost:3000/api/health | grep -q '"status"'
