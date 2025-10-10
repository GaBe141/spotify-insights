#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup pre-commit hooks for Audora Music Discovery System

.DESCRIPTION
    This script installs pre-commit and configures git hooks to automatically
    run Ruff, MyPy, and Black before each commit.

.EXAMPLE
    .\setup-pre-commit.ps1
#>

Write-Host "🎵 Setting up Pre-commit Hooks for Audora" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.11+ first." -ForegroundColor Red
    exit 1
}

# Check if we're in a git repository
if (-not (Test-Path ".git")) {
    Write-Host "❌ Not a git repository. Please run this script from the repository root." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Git repository detected" -ForegroundColor Green
Write-Host ""

# Install pre-commit
Write-Host "📦 Installing pre-commit..." -ForegroundColor Yellow
try {
    python -m pip install --upgrade pre-commit
    Write-Host "✅ Pre-commit installed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install pre-commit" -ForegroundColor Red
    exit 1
}

# Install the required tools
Write-Host ""
Write-Host "📦 Installing code quality tools..." -ForegroundColor Yellow
try {
    python -m pip install --upgrade black ruff mypy bandit
    Write-Host "✅ Tools installed: Black, Ruff, MyPy, Bandit" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Some tools may not have installed correctly" -ForegroundColor Yellow
}

# Install pre-commit hooks
Write-Host ""
Write-Host "🔧 Installing pre-commit hooks..." -ForegroundColor Yellow
try {
    python -m pre_commit install
    Write-Host "✅ Pre-commit hooks installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install pre-commit hooks" -ForegroundColor Red
    exit 1
}

# Run pre-commit on all files to verify setup
Write-Host ""
Write-Host "🧪 Testing pre-commit setup (this may take a moment)..." -ForegroundColor Yellow
Write-Host "   Running on all files to download hook environments..." -ForegroundColor Gray
Write-Host "   Note: First run may show many auto-fixes - this is normal!" -ForegroundColor Gray
try {
    python -m pre_commit run --all-files 2>&1 | Out-Null
    Write-Host "✅ Pre-commit hooks are ready" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Initial run completed with some issues (normal for first run)" -ForegroundColor Yellow
    Write-Host "   Auto-fixable issues have been corrected." -ForegroundColor Gray
}

# Show installed hooks
Write-Host ""
Write-Host "📋 Installed hooks:" -ForegroundColor Cyan
Write-Host "   • Black (code formatting)" -ForegroundColor White
Write-Host "   • Ruff (fast linting + formatting)" -ForegroundColor White
Write-Host "   • MyPy (type checking)" -ForegroundColor White
Write-Host "   • Bandit (security checks)" -ForegroundColor White
Write-Host "   • Pre-commit hooks (trailing whitespace, YAML/JSON validation, etc.)" -ForegroundColor White

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Usage:" -ForegroundColor Cyan
Write-Host "   The hooks will run automatically before each commit." -ForegroundColor White
Write-Host ""
Write-Host "🔧 Manual commands:" -ForegroundColor Cyan
Write-Host "   • Run on all files:      python -m pre_commit run --all-files" -ForegroundColor White
Write-Host "   • Run specific hook:     python -m pre_commit run black --all-files" -ForegroundColor White
Write-Host "   • Update hooks:          python -m pre_commit autoupdate" -ForegroundColor White
Write-Host "   • Skip hooks (not recommended): git commit --no-verify" -ForegroundColor White
Write-Host ""
Write-Host "🎉 Your code will now be automatically checked before each commit!" -ForegroundColor Green
