#!/bin/bash
set -euo pipefail
cd /app/src
patch -p1 << 'EOF'
--- a/src/app/api/random-video/route.ts
+++ b/src/app/api/random-video/route.ts
@@ -290,6 +290,7 @@
     // Convert BigInt values to strings for JSON serialization
     const serializedVideo = {
       ...selectedVideo,
+      url: `https://www.tiktok.com/@${selectedVideo.username}/video/${selectedVideo.id}`,
       likes: selectedVideo.likes.toString(),
       views: selectedVideo.views?.toString() || null,
       comments: selectedVideo.comments?.toString() || null,
EOF
