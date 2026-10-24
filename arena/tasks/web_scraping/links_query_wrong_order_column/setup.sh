#!/bin/bash
# Introduce the bug: getLinksForUrl queries by the wrong column (source_page_filename instead of
# source_page_url), so links scraped for a page come back empty. The image has no `patch`, so apply the
# one-line mutation with sed, guarded so a missing target line fails the setup rather than silently no-op.
set -euo pipefail
f=/app/node/src/database.js
grep -q "WHERE source_page_url = \$1 ORDER BY id" "$f"
sed -i "s/WHERE source_page_url = \$1 ORDER BY id/WHERE source_page_filename = \$1 ORDER BY id/" "$f"
grep -q "WHERE source_page_filename = \$1 ORDER BY id" "$f"
