# Pre-commit Hooks Guide

## Overview

Audora uses pre-commit hooks to automatically enforce code quality standards before commits. This ensures that all code entering the repository meets our quality and security standards.

## What Gets Checked

### 1. **Black** - Code Formatting
- Automatically formats Python code to a consistent style
- Line length: 100 characters
- Ensures readable, consistent code across the project

### 2. **Ruff** - Fast Python Linting
- Checks for common Python errors and anti-patterns
- Auto-fixes many issues automatically
- Enforces PEP 8 style guide
- Checks for unused imports, undefined names, etc.

### 3. **MyPy** - Static Type Checking
- Verifies type hints and catches type-related bugs
- Helps prevent runtime errors
- Improves code documentation and IDE support

### 4. **Bandit** - Security Scanning
- Identifies common security issues
- Checks for hardcoded passwords, SQL injection risks, etc.
- Helps maintain secure code practices

### 5. **Additional Checks**
- Trailing whitespace removal
- End-of-file fixing
- YAML/JSON syntax validation
- Large file detection
- Merge conflict markers
- Debug statement detection

## Setup

### Initial Installation

Run the setup script:

```powershell
.\setup-pre-commit.ps1
```

Or manually:

```bash
# Install pre-commit
pip install pre-commit

# Install the tools
pip install black ruff mypy bandit

# Install the hooks
pre-commit install
```

### Verify Installation

```bash
pre-commit run --all-files
```

## Usage

### Automatic (Recommended)

Once installed, the hooks run automatically when you commit:

```bash
git add .
git commit -m "Your commit message"
```

If any checks fail:
1. The commit will be blocked
2. Auto-fixable issues will be fixed automatically
3. You'll need to review changes and re-stage them
4. Commit again

### Manual Runs

Run all hooks on all files:
```bash
pre-commit run --all-files
```

Run specific hook:
```bash
pre-commit run black --all-files
pre-commit run ruff --all-files
pre-commit run mypy --all-files
```

Run on specific files:
```bash
pre-commit run --files path/to/file.py
```

### Skipping Hooks (Not Recommended)

Only in emergencies:
```bash
git commit --no-verify -m "Emergency fix"
```

**Note:** CI will still run all checks, so issues will be caught anyway.

## Configuration Files

### `.pre-commit-config.yaml`
Main configuration for pre-commit hooks and tool versions.

### `pyproject.toml`
Contains configuration for:
- Black (formatting options)
- Ruff (linting rules)
- MyPy (type checking settings)
- Bandit (security rules)

## Common Workflows

### When Starting Work

```bash
# Update hooks to latest versions
pre-commit autoupdate

# Run checks on all files
pre-commit run --all-files
```

### Before Committing

```bash
# Stage your changes
git add .

# Manually run checks (optional, will run automatically on commit)
pre-commit run

# Commit (hooks will run automatically)
git commit -m "Your message"
```

### Fixing Issues

If Black or Ruff auto-fix files:
```bash
# Review the changes
git diff

# Stage the fixed files
git add .

# Commit again
git commit -m "Your message"
```

If MyPy finds type errors:
```bash
# Fix the type issues in your code
# Then stage and commit again
git add .
git commit -m "Your message"
```

## Troubleshooting

### Hook Installation Fails

```bash
# Reinstall pre-commit
pip install --upgrade --force-reinstall pre-commit
pre-commit install
```

### Hooks Don't Run

```bash
# Check if hooks are installed
ls .git/hooks/pre-commit

# Reinstall
pre-commit install --install-hooks
```

### MyPy Errors on Third-Party Libraries

Add to `pyproject.toml`:
```toml
[[tool.mypy.overrides]]
module = ["problematic_library.*"]
ignore_missing_imports = true
```

### Too Many Errors on First Run

Normal! Run multiple times:
```bash
# Let auto-fixers do their work
pre-commit run --all-files

# Review changes
git diff

# Stage and commit fixes separately
git add .
git commit -m "Apply pre-commit auto-fixes"
```

## Best Practices

1. **Run hooks before pushing**: `pre-commit run --all-files`
2. **Update hooks monthly**: `pre-commit autoupdate`
3. **Don't skip hooks** unless absolutely necessary
4. **Review auto-fixes** before committing
5. **Fix MyPy errors** rather than adding `# type: ignore` everywhere
6. **Keep hooks fast** - exclude large directories if needed

## CI Integration

Pre-commit checks also run in GitHub Actions on:
- All pushes to `main` and `develop`
- All pull requests

This ensures code quality even if someone skips local hooks.

## Customization

### Exclude Files

Edit `.pre-commit-config.yaml`:
```yaml
- id: mypy
  exclude: ^(tests/|scripts/|notebooks/)
```

### Adjust Rules

Edit `pyproject.toml`:
```toml
[tool.ruff]
ignore = ["E501"]  # Ignore line too long
```

### Change Line Length

Edit `pyproject.toml`:
```toml
[tool.black]
line-length = 120
```

## Support

For issues or questions:
1. Check the [pre-commit documentation](https://pre-commit.com)
2. Review tool-specific docs:
   - [Black](https://black.readthedocs.io/)
   - [Ruff](https://docs.astral.sh/ruff/)
   - [MyPy](https://mypy.readthedocs.io/)
   - [Bandit](https://bandit.readthedocs.io/)
3. Check project documentation in `docs/`

## Summary

Pre-commit hooks help maintain code quality by:
- ✅ Catching errors before they enter the codebase
- ✅ Enforcing consistent code style
- ✅ Improving security
- ✅ Reducing code review time
- ✅ Making the codebase more maintainable

**Always commit with confidence knowing your code has been checked!** 🎉
