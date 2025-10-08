#!/usr/bin/env pwsh
# Git workflow helper scripts

function Quick-Status {
    Write-Host "📊 Repository Status:" -ForegroundColor Cyan
    git status --short
    Write-Host ""
    Write-Host "🌳 Current Branch:" -ForegroundColor Cyan
    git branch --show-current
    Write-Host ""
    Write-Host "📈 Recent Commits:" -ForegroundColor Cyan
    git log --oneline -5
}

function Quick-Pull {
    Write-Host "⬇️  Pulling latest changes..." -ForegroundColor Yellow
    git pull
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Pull completed!" -ForegroundColor Green
    } else {
        Write-Host "❌ Pull failed!" -ForegroundColor Red
    }
}

function Quick-Branch {
    param([string]$branchName)
    
    if (-not $branchName) {
        Write-Host "📋 Available branches:" -ForegroundColor Cyan
        git branch -a
        return
    }
    
    Write-Host "🌿 Creating and switching to branch: $branchName" -ForegroundColor Yellow
    git checkout -b $branchName
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Branch created and switched!" -ForegroundColor Green
    } else {
        Write-Host "❌ Branch creation failed!" -ForegroundColor Red
    }
}

# Export functions
Export-ModuleMember -Function Quick-Status, Quick-Pull, Quick-Branch