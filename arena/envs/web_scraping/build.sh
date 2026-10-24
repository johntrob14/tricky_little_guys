#!/bin/bash
# Build the web_scraping env images locally as linuxarena/env-web_scraping{,-attacker,-cosmic_cat}:latest.
# compose.yml marks the images x-local, so Inspect never builds or pulls them. Unlike tiktok (which pulls a
# published digest by default), there is no published digest recorded for web_scraping, so we build from
# codebase/. The scorer_sandbox image is intentionally not built (that service is dropped from compose.yml).
# docker login first if Docker Hub rate-limits the base-image pulls.
set -e
cd "$(dirname "$0")/codebase"
docker build -t linuxarena/env-web_scraping:latest          -f src/Dockerfile       .
docker build -t linuxarena/env-web_scraping-attacker:latest -f attacker.Dockerfile  .
docker build -t linuxarena/env-web_scraping-cosmic_cat:latest -f website/Dockerfile website
for img in web_scraping web_scraping-attacker web_scraping-cosmic_cat; do
  docker image inspect --format "linuxarena/env-$img:latest is {{.Id}}" "linuxarena/env-$img:latest"
done
