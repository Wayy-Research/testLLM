#!/usr/bin/env python3
"""
Script to bump version and create a release.
Usage: python scripts/bump_version.py [major|minor|patch]
"""

import sys
import re
import subprocess
from pathlib import Path

def get_current_version():
    """Get current version from __version__.py"""
    version_file = Path(__file__).parent.parent / "testllm" / "__version__.py"
    with open(version_file, 'r') as f:
        content = f.read()
    
    version_match = re.search(r'__version__ = [\'"]([^\'"]*)[\'"]', content)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

def bump_version(current_version, bump_type):
    """Bump version based on type"""
    major, minor, patch = map(int, current_version.split('.'))
    
    if bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    elif bump_type == 'patch':
        patch += 1
    else:
        raise ValueError("bump_type must be 'major', 'minor', or 'patch'")
    
    return f"{major}.{minor}.{patch}"

def update_version_file(new_version):
    """Update __version__.py with new version"""
    version_file = Path(__file__).parent.parent / "testllm" / "__version__.py"
    
    with open(version_file, 'r') as f:
        content = f.read()
    
    # Update version
    content = re.sub(
        r'__version__ = [\'"][^\'"]*[\'"]',
        f'__version__ = "{new_version}"',
        content
    )
    
    # Update version_info tuple
    major, minor, patch = map(int, new_version.split('.'))
    content = re.sub(
        r'__version_info__ = \([^)]*\)',
        f'__version_info__ = ({major}, {minor}, {patch})',
        content
    )
    
    with open(version_file, 'w') as f:
        f.write(content)

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py [major|minor|patch]")
        sys.exit(1)
    
    bump_type = sys.argv[1]
    if bump_type not in ['major', 'minor', 'patch']:
        print("Error: bump_type must be 'major', 'minor', or 'patch'")
        sys.exit(1)
    
    current_version = get_current_version()
    new_version = bump_version(current_version, bump_type)
    
    print(f"Bumping version from {current_version} to {new_version}")
    
    # Update version file
    update_version_file(new_version)
    
    # Git commands
    print("Updating git...")
    subprocess.run(["git", "add", "testllm/__version__.py"], check=True)
    subprocess.run(["git", "commit", "-m", f"Bump version to {new_version}"], check=True)
    subprocess.run(["git", "tag", f"v{new_version}"], check=True)
    
    print(f"Version bumped to {new_version}")
    print("To release:")
    print("  git push origin main")
    print(f"  git push origin v{new_version}")
    print("Or create a GitHub release from the tag to trigger PyPI deployment")

if __name__ == "__main__":
    main()