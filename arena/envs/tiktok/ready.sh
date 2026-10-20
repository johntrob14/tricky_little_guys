#!/bin/bash
# Exit 0 once the app answers on :80. The compose command runs scripts/setup.sh (migrate, seed, `next dev`) in the background.
curl -sf -m 10 http://localhost:80/api/random-video | grep -q '"id"'
