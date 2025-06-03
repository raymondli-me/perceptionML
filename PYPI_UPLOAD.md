# PyPI Upload Instructions for PerceptionML

## Prerequisites

1. **PyPI Account**: Create accounts at:
   - Production: https://pypi.org/account/register/
   - Test: https://test.pypi.org/account/register/

2. **API Tokens**: Generate tokens for secure upload:
   - Go to Account Settings → API tokens
   - Create a token with upload permissions
   - Save securely (you won't see it again!)

## Setup Authentication

### Option 1: Using .pypirc (Traditional)
Create `~/.pypirc`:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-PRODUCTION-TOKEN

[testpypi]
username = __token__
password = pypi-YOUR-TEST-TOKEN
```

### Option 2: Environment Variables (Recommended for CI/CD)
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR-TOKEN
```

## Upload Process

### 1. Clean Previous Builds
```bash
rm -rf dist/ build/ *.egg-info
```

### 2. Update Version
Edit `pyproject.toml`:
```toml
version = "0.1.1"  # Increment version
```

### 3. Build Package
```bash
python -m build
```

### 4. Verify Package
```bash
# Check package integrity
twine check dist/*

# Test install locally
pip install dist/perceptionml-*.whl
```

### 5. Upload to Test PyPI (Recommended First)
```bash
twine upload --repository testpypi dist/*
```

Test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ perceptionml
```

### 6. Upload to Production PyPI
```bash
twine upload dist/*
```

## Version Management

Follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking API changes
- MINOR: New features, backwards compatible
- PATCH: Bug fixes

## Troubleshooting

### "Invalid or non-existent authentication"
- Ensure token starts with `pypi-`
- Check token permissions
- Verify .pypirc formatting

### "Package already exists"
- Version must be unique
- Delete old builds: `rm -rf dist/`
- Increment version number

### Build Warnings
- License warning: Already fixed (Apache-2.0)
- Missing files: Check MANIFEST.in

## GitHub Release Process

After PyPI upload:
```bash
git tag v0.1.0
git push origin v0.1.0
```

Create release on GitHub with:
- Tag: v0.1.0
- Title: PerceptionML v0.1.0
- Description: Changelog and key features
- Attach: dist/*.whl and dist/*.tar.gz

## Continuous Deployment

For GitHub Actions, add secrets:
- PYPI_API_TOKEN
- TEST_PYPI_API_TOKEN

Example workflow in `.github/workflows/publish.yml`:
```yaml
name: Publish to PyPI
on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Build and publish
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: |
        pip install build twine
        python -m build
        twine upload dist/*
```