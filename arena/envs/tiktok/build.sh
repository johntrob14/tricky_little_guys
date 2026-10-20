#!/bin/bash
# Build the environment image. compose.yml marks it x-local, so Inspect uses the local image and never rebuilds or pulls it.
docker build -t linuxarena/env-tiktok:latest "$(dirname "$0")/codebase"
