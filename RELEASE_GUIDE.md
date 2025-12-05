# Release Guide

This document explains how to create a new release using the automated CI/CD workflow.

## Automated Release Process

The workflow automatically:
1. ✅ Runs tests on Python 3.8, 3.9, 3.10, and 3.11
2. ✅ Builds the wheel (.whl) file if tests pass
3. ✅ Creates a GitHub Release with the wheel attached
4. ✅ Generates release notes automatically

## How to Create a New Release

### Option 1: Using Git Tags (Recommended)

1. **Update version in `setup.py`**:
   ```python
   version='0.5.1',  # Update this
   ```

2. **Commit the version change**:
   ```bash
   git add setup.py
   git commit -m "Bump version to 0.5.1"
   ```

3. **Create and push a git tag**:
   ```bash
   git tag v0.5.1
   git push origin v0.5.1
   ```

4. **Done!** The workflow will automatically:
   - Run all tests
   - Build the package
   - Create a GitHub Release at: `https://github.com/MSWinds/rankfusion/releases`

### Option 2: Manual Trigger

1. Go to: `https://github.com/MSWinds/rankfusion/actions`
2. Click on "Build and Release" workflow
3. Click "Run workflow" button
4. Select branch and run

**Note**: Manual triggers build the package but won't create a release (only tags create releases).

## What Happens if Tests Fail?

- ❌ If any test fails, the build and release are **automatically cancelled**
- You'll get a notification email from GitHub Actions
- Fix the issue, commit, and try again

## Checking Workflow Status

- View all workflow runs: `https://github.com/MSWinds/rankfusion/actions`
- Each run shows:
  - Test results for all Python versions
  - Build logs
  - Any errors

## Tips

- **Semantic Versioning**: Use `v0.5.1` for patches, `v0.6.0` for features, `v1.0.0` for breaking changes
- **Pre-releases**: Tag as `v0.6.0-beta` for testing (will still create a release)
- **Keep changelog**: Update README.md with changes before releasing

## Troubleshooting

**Problem**: "Tests failed on Python 3.11"
- **Solution**: Check the test logs in GitHub Actions, fix the code, commit and retag

**Problem**: "Release already exists"
- **Solution**: Delete the release and tag from GitHub, then push the tag again

**Problem**: "GITHUB_TOKEN permission denied"
- **Solution**: Check repository Settings → Actions → General → Workflow permissions (should be "Read and write")
