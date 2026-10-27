#!/bin/bash
# index.html = _css.txt + _body.html + "<script>(() => {" + _helpers.txt + _main.js (the main file closes the IIFE and script).
cd "$(dirname "$0")"; { cat _css.txt _body.html; printf '\n<script>\n(() => {\n'; cat _helpers.txt _main.js; } > index.html
