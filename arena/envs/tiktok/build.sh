#!/bin/bash
# Get the environment image as linuxarena/env-tiktok:latest, which compose.yml marks x-local (Inspect never builds or pulls it).
# Default: pull the published build keep.yaml names by digest. `build.sh --build`: build from codebase/ after changing the Dockerfile or app source.
set -e
cd "$(dirname "$0")"
DIGEST=linuxarena/env-tiktok@sha256:e6d8bb20fd3fb1efd8c6c5472c9b9421907c9341f1273efd5521fedeb41ed58f
if [ "$1" = "--build" ]; then
    docker build -t linuxarena/env-tiktok:latest codebase
else
    docker pull $DIGEST && docker tag $DIGEST linuxarena/env-tiktok:latest
fi
docker image inspect --format 'linuxarena/env-tiktok:latest is {{.Id}}' linuxarena/env-tiktok:latest
