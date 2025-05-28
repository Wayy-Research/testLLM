# Publishing testLLM to PyPI

This guide covers how to publish testLLM to PyPI.

## Prerequisites

1. **Create PyPI account**: https://pypi.org/account/register/
2. **Install build tools**:
   ```bash
   pip install build twine
   ```
3. **Get API token** from PyPI account settings

## Publishing Steps

### 1. Update Version
Edit `testllm/__version__.py`:
```python
__version__ = "0.1.1"  # Increment version
__version_info__ = (0, 1, 1)
```

### 2. Update Changelog
Add new version section to `CHANGELOG.md`

### 3. Build Package
```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build package
python -m build
```

This creates:
- `dist/testllm-X.X.X.tar.gz` (source distribution)
- `dist/testllm-X.X.X-py3-none-any.whl` (wheel)

### 4. Test Package Locally
```bash
# Install locally
pip install dist/testllm-*.whl

# Test basic import
python -c "import testllm; print(testllm.__version__)"

# Test pytest plugin
pytest --testllm examples/test_agent.py
```

### 5. Upload to Test PyPI (Optional)
```bash
# Upload to test.pypi.org first
twine upload --repository testpypi dist/*

# Test install from test PyPI
pip install --index-url https://test.pypi.org/simple/ testllm
```

### 6. Upload to PyPI
```bash
# Upload to production PyPI
twine upload dist/*
```

### 7. Verify Upload
- Check package page: https://pypi.org/project/testllm/
- Test installation: `pip install testllm`

## Automation (Future)

Consider GitHub Actions for automated publishing:

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI
on:
  release:
    types: [published]
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install build twine
      - run: python -m build
      - run: twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
```

## Version Management

Follow semantic versioning:
- **Patch** (0.1.X): Bug fixes
- **Minor** (0.X.0): New features, backwards compatible
- **Major** (X.0.0): Breaking changes

## Release Checklist

- [ ] Update version in `__version__.py`
- [ ] Update CHANGELOG.md
- [ ] Test package locally
- [ ] Build package (`python -m build`)
- [ ] Upload to Test PyPI (optional)
- [ ] Upload to PyPI (`twine upload dist/*`)
- [ ] Create GitHub release
- [ ] Update documentation links

## Package Information

- **Package name**: `testllm`
- **PyPI URL**: https://pypi.org/project/testllm/
- **Install command**: `pip install testllm`