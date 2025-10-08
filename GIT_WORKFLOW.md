# 🚀 Git Workflow Enhancements

This project now includes several enhancements to make your VS Code to GitHub workflow lightning fast!

## ⚡ Quick Commands

### PowerShell Scripts

1. **Quick Push** - Stage, commit, and push in one command:
   ```powershell
   .\quick-push.ps1 "Your commit message"
   ```

2. **Git Helpers** - Load helpful functions:
   ```powershell
   . .\git-helpers.ps1
   Quick-Status    # Show repo status
   Quick-Pull      # Pull latest changes
   Quick-Branch "feature-name"  # Create and switch to new branch
   ```

## ⌨️ Keyboard Shortcuts

- `Ctrl+Shift+G C` - Commit staged changes
- `Ctrl+Shift+G P` - Push to remote
- `Ctrl+Shift+G S` - Sync (pull + push)
- `Ctrl+Shift+G A` - Stage all changes
- `Ctrl+Shift+G U` - Unstage all changes

## 🛠️ VS Code Settings Added

- **Auto-fetch**: Automatically fetch from remote every few minutes
- **Smart commit**: Enable staging files when committing
- **Auto-push**: Push after successful commit
- **Clean sync**: Use rebase when syncing

## 📦 Extensions Installed

- **Git Graph**: Visual commit history
- **GitLens**: Enhanced git blame and history

## 🖥️ GitHub CLI Commands

```bash
gh repo view              # View repo info
gh pr create             # Create pull request
gh pr list               # List pull requests
gh issue create          # Create new issue
gh issue list            # List issues
```

## 🎯 Typical Workflow Now

1. Make your changes
2. `.\quick-push.ps1 "fix: updated feature"`
3. Done! ✅

Or use the keyboard shortcuts for more control over the process.

## 🔧 Configuration Files

- `.vscode/settings.json` - VS Code git settings
- `.vscode/keybindings.json` - Custom keyboard shortcuts
- `quick-push.ps1` - One-command push script
- `git-helpers.ps1` - Additional helper functions