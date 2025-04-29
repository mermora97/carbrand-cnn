#!/usr/bin/env bash
#
# download_data.sh ─ Fetches the Kaggle “50-Car-Brands” dataset and extracts it into the "data" folder
#
# Prerequisites:
#   • Install Kaggle CLI:  pip install kaggle
#   • Add your Kaggle API token to ~/.kaggle/kaggle.json  **OR**
#     export KAGGLE_USERNAME && KAGGLE_KEY in your shell.

set -e          # abort on first error
set -o pipefail # catch errors in pipelines

DATA_DIR="../data"
mkdir -p "$DATA_DIR"

echo "⬇️  Downloading dataset to $DATA_DIR ..."
kaggle datasets download -d yamaerenay/100-images-of-top-50-car-brands -p "$DATA_DIR" -q

echo "📦  Extracting..."
unzip -q "$DATA_DIR"/*.zip -d "$DATA_DIR"
rm "$DATA_DIR"/*.zip

echo "✅  Done. Raw images are in $DATA_DIR"
