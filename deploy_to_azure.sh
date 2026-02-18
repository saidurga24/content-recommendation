#!/bin/bash

# =============================================================================
# Azure Deployment Script for Project 9 - Article Recommendation System
# =============================================================================
# Deploys the recommendation API to Azure Functions (Consumption Plan)
#
# Prerequisites:
#   - Azure CLI installed (az)
#   - Logged in to Azure (az login)
#   - Python 3 with pip
#
# Usage:
#   ./deploy_to_azure.sh [command]
#
# Commands:
#   setup     - Create all Azure resources
#   deploy    - Build package with deps, upload to Blob, deploy
#   status    - Check deployment status
#   logs      - View function logs
#   delete    - Delete all resources (cleanup)
#   local     - Run Flask API locally
# =============================================================================

set -e

# Configuration
RESOURCE_GROUP="rg-mycontent-p9"
LOCATION="eastus"
FUNCTION_APP="func-mycontent-recommendation"
BLOB_CONTAINER="deployments"
PYTHON_VERSION="3.11"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_msg() {
    echo -e "${2}${1}${NC}"
}

# Load saved config (storage account name + connection string)
load_config() {
    if [ -f ".azure_config" ]; then
        source .azure_config
    else
        print_msg "Config not found. Run './deploy_to_azure.sh setup' first." "$RED"
        exit 1
    fi
}

check_azure_cli() {
    if ! command -v az &> /dev/null; then
        print_msg "Azure CLI not found. Install: brew install azure-cli" "$RED"
        exit 1
    fi
}

check_azure_login() {
    if ! az account show &> /dev/null; then
        print_msg "Not logged in to Azure. Running 'az login'..." "$YELLOW"
        az login
    fi
    SUBSCRIPTION=$(az account show --query name -o tsv)
    print_msg "Using Azure subscription: $SUBSCRIPTION" "$GREEN"
}

# =============================================================================
# SETUP
# =============================================================================
setup() {
    print_msg "Setting up Azure resources..." "$BLUE"

    # Use a fixed storage account name (must be globally unique, lowercase, 3-24 chars)
    STORAGE_ACCOUNT="stmycontentp9$(whoami | tr -dc 'a-z0-9' | head -c 4)"

    # Register required providers
    print_msg "Registering resource providers..." "$YELLOW"
    az provider register --namespace Microsoft.Storage --wait 2>/dev/null || true
    az provider register --namespace Microsoft.Web --wait 2>/dev/null || true

    # Create Resource Group
    print_msg "Creating Resource Group: $RESOURCE_GROUP" "$YELLOW"
    az group create \
        --name $RESOURCE_GROUP \
        --location $LOCATION \
        --output none

    # Create Storage Account
    print_msg "Creating Storage Account: $STORAGE_ACCOUNT" "$YELLOW"
    az storage account create \
        --name $STORAGE_ACCOUNT \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --sku Standard_LRS \
        --output none

    # Get Storage Connection String
    STORAGE_CONNECTION=$(az storage account show-connection-string \
        --name $STORAGE_ACCOUNT \
        --resource-group $RESOURCE_GROUP \
        --query connectionString -o tsv)

    # Create Blob Container for deployment packages
    print_msg "Creating Blob Container: $BLOB_CONTAINER" "$YELLOW"
    az storage container create \
        --name $BLOB_CONTAINER \
        --connection-string "$STORAGE_CONNECTION" \
        --output none

    # Create Function App
    print_msg "Creating Function App: $FUNCTION_APP (Python $PYTHON_VERSION, Consumption Plan)" "$YELLOW"
    az functionapp create \
        --name $FUNCTION_APP \
        --resource-group $RESOURCE_GROUP \
        --storage-account $STORAGE_ACCOUNT \
        --consumption-plan-location $LOCATION \
        --runtime python \
        --runtime-version $PYTHON_VERSION \
        --functions-version 4 \
        --os-type Linux \
        --output none

    # Save config for deploy/status/delete commands
    # Connection string contains semicolons - must be quoted to survive bash source
    cat > .azure_config <<CFGEOF
STORAGE_ACCOUNT='$STORAGE_ACCOUNT'
STORAGE_CONNECTION='$STORAGE_CONNECTION'
CFGEOF

    print_msg "Setup complete!" "$GREEN"
    print_msg "Storage Account: $STORAGE_ACCOUNT" "$BLUE"
    print_msg "Function App URL: https://$FUNCTION_APP.azurewebsites.net" "$BLUE"
    print_msg ""
    print_msg "Next: ./deploy_to_azure.sh deploy" "$YELLOW"
}

# =============================================================================
# DEPLOY
# =============================================================================
deploy() {
    load_config
    print_msg "Deploying Function App..." "$BLUE"

    DEPLOY_DIR=$(mktemp -d)
    print_msg "Building package in $DEPLOY_DIR" "$YELLOW"

    # Copy function code
    cp P9_02_azure_function/function_app.py "$DEPLOY_DIR/"
    cp P9_02_azure_function/host.json "$DEPLOY_DIR/"
    cp P9_02_azure_function/requirements.txt "$DEPLOY_DIR/"

    # Copy src/ module
    cp -r src "$DEPLOY_DIR/src"
    find "$DEPLOY_DIR/src" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

    # Copy models
    cp -r P9_02_azure_function/model "$DEPLOY_DIR/model"

    # Install pip dependencies for Linux x86_64 (Azure runtime)
    print_msg "Installing dependencies for Linux x86_64..." "$YELLOW"
    pip install -r "$DEPLOY_DIR/requirements.txt" \
        --target "$DEPLOY_DIR/.python_packages/lib/site-packages" \
        --platform manylinux2014_x86_64 \
        --only-binary=:all: \
        --python-version $PYTHON_VERSION \
        --quiet 2>&1

    # Create zip
    cd "$DEPLOY_DIR"
    zip -r deploy.zip . -x "*.pyc" -x "*__pycache__*" > /dev/null 2>&1
    ZIP_SIZE=$(ls -lh deploy.zip | awk '{print $5}')
    print_msg "Package size: $ZIP_SIZE" "$YELLOW"

    # Upload zip to Blob Storage
    print_msg "Uploading to Blob Storage..." "$YELLOW"
    az storage blob upload \
        --container-name $BLOB_CONTAINER \
        --file deploy.zip \
        --name "function-app.zip" \
        --connection-string "$STORAGE_CONNECTION" \
        --overwrite \
        --output none

    # Generate SAS URL (valid 2 years)
    EXPIRY=$(date -u -v+2y '+%Y-%m-%dT%H:%MZ' 2>/dev/null || date -u -d '+2 years' '+%Y-%m-%dT%H:%MZ')
    SAS_URL=$(az storage blob generate-sas \
        --container-name $BLOB_CONTAINER \
        --name "function-app.zip" \
        --connection-string "$STORAGE_CONNECTION" \
        --permissions r \
        --expiry "$EXPIRY" \
        --full-uri \
        -o tsv)

    # Point function app at the blob package
    print_msg "Configuring function app..." "$YELLOW"
    az functionapp config appsettings set \
        --name $FUNCTION_APP \
        --resource-group $RESOURCE_GROUP \
        --settings "WEBSITE_RUN_FROM_PACKAGE=$SAS_URL" \
        --output none

    # Restart
    az functionapp restart --name $FUNCTION_APP --resource-group $RESOURCE_GROUP 2>/dev/null

    cd -
    rm -rf "$DEPLOY_DIR"

    print_msg "Deployed! Waiting 90s for cold start..." "$YELLOW"
    sleep 90

    # Test
    print_msg "Testing health endpoint..." "$BLUE"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://$FUNCTION_APP.azurewebsites.net/api/health")
    if [ "$HTTP_CODE" == "200" ]; then
        print_msg "Deployment successful!" "$GREEN"
        curl -s "https://$FUNCTION_APP.azurewebsites.net/api/health" | python3 -m json.tool
        print_msg "\nAPI URL: https://$FUNCTION_APP.azurewebsites.net/api" "$BLUE"
    else
        print_msg "Health check returned HTTP $HTTP_CODE (may still be starting up)" "$YELLOW"
        print_msg "Try again in a minute: curl https://$FUNCTION_APP.azurewebsites.net/api/health" "$YELLOW"
    fi
}

# =============================================================================
# STATUS
# =============================================================================
status() {
    print_msg "Checking deployment status..." "$BLUE"

    if az functionapp show --name $FUNCTION_APP --resource-group $RESOURCE_GROUP &> /dev/null; then
        print_msg "Function App Status:" "$GREEN"
        az functionapp show \
            --name $FUNCTION_APP \
            --resource-group $RESOURCE_GROUP \
            --query "{Name:name, State:state, URL:defaultHostName, Runtime:siteConfig.linuxFxVersion}" \
            --output table

        print_msg "\nTesting health endpoint..." "$BLUE"
        curl -s "https://$FUNCTION_APP.azurewebsites.net/api/health" | python3 -m json.tool 2>/dev/null \
            || print_msg "Health endpoint not responding (function may be cold)" "$YELLOW"
    else
        print_msg "Function App not found. Run 'setup' first." "$RED"
    fi
}

# =============================================================================
# LOGS
# =============================================================================
logs() {
    print_msg "Fetching Function App logs..." "$BLUE"
    az functionapp log stream \
        --name $FUNCTION_APP \
        --resource-group $RESOURCE_GROUP
}

# =============================================================================
# DELETE
# =============================================================================
delete() {
    print_msg "WARNING: This will delete all Azure resources for this project!" "$RED"
    read -p "Are you sure? (yes/no): " confirm

    if [ "$confirm" == "yes" ]; then
        print_msg "Deleting Resource Group: $RESOURCE_GROUP" "$YELLOW"
        az group delete \
            --name $RESOURCE_GROUP \
            --yes \
            --no-wait
        rm -f .azure_config
        print_msg "Deletion initiated. Resources will be removed in a few minutes." "$GREEN"
    else
        print_msg "Deletion cancelled." "$BLUE"
    fi
}

# =============================================================================
# LOCAL
# =============================================================================
local_run() {
    print_msg "Starting local Flask API + Web App..." "$BLUE"
    source venv/bin/activate
    python P9_02_azure_function/app.py &
    sleep 5
    python P9_03_web_app/app.py
}

# =============================================================================
# HELP
# =============================================================================
show_help() {
    echo "Azure Deployment Script for Project 9"
    echo ""
    echo "Usage: ./deploy_to_azure.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup     Create Azure resources (Resource Group, Storage, Function App)"
    echo "  deploy    Build package with deps, upload to Blob, deploy function"
    echo "  status    Check deployment status and test endpoints"
    echo "  logs      Stream function app logs"
    echo "  delete    Delete all Azure resources (cleanup)"
    echo "  local     Run Flask API + Web App locally"
    echo "  help      Show this help message"
    echo ""
    echo "Workflow:"
    echo "  1. ./deploy_to_azure.sh setup     # Create Azure resources"
    echo "  2. Run training notebook           # Train and save models"
    echo "  3. ./deploy_to_azure.sh deploy    # Build, upload, deploy"
    echo "  4. ./deploy_to_azure.sh status    # Verify deployment"
    echo "  5. ./deploy_to_azure.sh delete    # Cleanup when done"
}

# Main
check_azure_cli
check_azure_login

case "${1:-help}" in
    setup)  setup ;;
    deploy) deploy ;;
    status) status ;;
    logs)   logs ;;
    delete) delete ;;
    local)  local_run ;;
    help|*) show_help ;;
esac
