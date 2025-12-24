#!/bin/bash
set -eoux pipefail

# This gets the directory where the script lives, then goes up one level to the root
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$(dirname "$SCRIPT_DIR")

# Define paths relative to Repo Root
CONFIG_PATH="$REPO_ROOT/config/config.yaml"
ENV_PATH="$REPO_ROOT/.env"

check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: Required dependency '$1' is not installed."
        case $1 in
            dvc) echo "   Install: pip install dvc" ;;
            yq)  echo "   Install: https://github.com/mikefarah/yq" ;;
            git) echo "   Install: https://git-scm.com/install/linux" ;;
        esac
        exit 1
    fi
}

# Fail for missing dependencies
check_dependency "git"
check_dependency "yq"
check_dependency "dvc"

# Load secrets and configs
if [ -f "$ENV_PATH" ]; then
    source "$ENV_PATH"
else
    echo "Warning: .env not found at $ENV_PATH"
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: config.yaml not found at $CONFIG_PATH"
    exit 1
fi

REPO_NAME=$(yq '.project.repo_name' "$CONFIG_PATH")
REMOTE_NAME=$(yq '.project.remote_name' "$CONFIG_PATH")
DATASET_URL=$(yq '.data.url' "$CONFIG_PATH")
DATASET_NAME=$(yq '.data.dataset_name' "$CONFIG_PATH")
TARGET_DIR_NAME=$(yq '.data.target_dir' "$CONFIG_PATH")
TARGET_PATH="$REPO_ROOT/$TARGET_DIR_NAME"

# Initialize DVC
# Change directory to root so DVC commands execute in the right context
cd "$REPO_ROOT"

if [ ! -d ".dvc" ]; then
    echo "Initializing DVC in $REPO_ROOT..."
    dvc init
    
    REMOTE_URL="https://dagshub.com/${DAGSHUB_USER_NAME}/${REPO_NAME}.dvc"
    dvc remote add -d "$REMOTE_NAME" "$REMOTE_URL"
    dvc remote modify "$REMOTE_NAME" --local auth basic 
    dvc remote modify "$REMOTE_NAME" --local user "$DAGSHUB_USER_NAME"
    dvc remote modify "$REMOTE_NAME" --local password "$DAGSHUB_TOKEN"
fi

echo "Fetching dataset to $TARGET_PATH..."
mkdir -p "$(dirname "$TARGET_PATH")"

# Download dataset
dvc get-url "$DATASET_URL" "$TARGET_PATH/$DATASET_NAME" --force
# Track dataset
dvc add "$TARGET_PATH/$DATASET_NAME"

echo "Committing DVC metadata..."
git add "$TARGET_PATH/$DATASET_NAME.dvc" "$TARGET_PATH/.gitignore"
git commit -m "chore: update dataset from $DATASET_URL" || echo "No changes to commit"

echo "Pushing data to DagsHub..."
dvc push -r "$REMOTE_NAME"

echo "✅ Bootstrap complete! ✅"


