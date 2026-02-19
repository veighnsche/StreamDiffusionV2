#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR/frontend"
npm install
npm run build
if [ $? -eq 0 ]; then
    echo -e "\033[1;32m\nfrontend build success \033[0m"
else
    echo -e "\033[1;31m\nfrontend build failed\n\033[0m" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
bash "$SCRIPT_DIR/start_engine.sh"
