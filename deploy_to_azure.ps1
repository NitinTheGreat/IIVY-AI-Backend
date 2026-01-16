$appName = "AIFunctionsApp2025"
$resourceGroup = "IIVY"
$json = Get-Content "local.settings.json" -Raw | ConvertFrom-Json

Write-Host "Reading local.settings.json..."
$settingsToUpdate = @()

foreach ($key in $json.Values.PSObject.Properties.Name) {
    $value = $json.Values.$key
    
    # Skip local-only or auto-managed keys
    if ($key -in @("AzureWebJobsStorage", "FUNCTIONS_WORKER_RUNTIME", "IsEncrypted")) {
        continue
    }

    # Override IS_LOCAL_DEV to false
    if ($key -eq "IS_LOCAL_DEV") {
        $value = "false"
    }

    $settingsToUpdate += "$key=$value"
}

Write-Host "Syncing App Settings to $appName..."
# Update settings in batches or all at once. quoted properly.
# We use logic to split if too many, but here ~15 settings is fine for one command.

if ($settingsToUpdate.Count -gt 0) {
    az functionapp config appsettings set --name $appName --resource-group $resourceGroup --settings $settingsToUpdate
}

Write-Host "App Settings synced."
Write-Host "Starting Deployment..."

func azure functionapp publish $appName --python
