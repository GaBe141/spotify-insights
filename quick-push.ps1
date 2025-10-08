param(
    [string]$message = "Quick update"
)

Write-Host "🚀 Starting quick push..." -ForegroundColor Cyan

# Check if we're in a git repository
if (-not (Test-Path ".git")) {
    Write-Host "❌ Not in a git repository!" -ForegroundColor Red
    exit 1
}

# Stage all changes
Write-Host "📦 Staging all changes..." -ForegroundColor Yellow
git add .

# Check if there are changes to commit
$status = git status --porcelain
if (-not $status) {
    Write-Host "✅ No changes to commit!" -ForegroundColor Green
    exit 0
}

# Commit with message
Write-Host "💾 Committing changes..." -ForegroundColor Yellow
git commit -m $message

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Commit failed!" -ForegroundColor Red
    exit 1
}

# Push to remote
Write-Host "⬆️  Pushing to remote..." -ForegroundColor Yellow
git push

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Changes successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "🎉 Quick push completed!" -ForegroundColor Magenta
} else {
    Write-Host "❌ Push failed!" -ForegroundColor Red
    exit 1
}