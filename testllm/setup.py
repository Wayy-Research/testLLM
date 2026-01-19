"""
testLLM Setup - CLI helper for API key configuration.

Guides users through setting up their evaluator API key with minimal friction.
"""

import os
import sys
import webbrowser
from pathlib import Path


GOOGLE_AI_STUDIO_URL = "https://aistudio.google.com/apikey"

ENV_VARS = {
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
}


def check_existing_keys() -> dict:
    """Check which API keys are already configured."""
    found = {}
    for provider, keys in ENV_VARS.items():
        for key in keys:
            if os.getenv(key):
                found[provider] = key
                break
    return found


def find_env_file() -> Path:
    """Find or determine the .env file location."""
    # Check current directory first
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return cwd_env

    # Check if we're in a project with pyproject.toml
    if (Path.cwd() / "pyproject.toml").exists():
        return cwd_env

    # Default to current directory
    return cwd_env


def save_api_key(key: str, env_var: str = "GOOGLE_API_KEY") -> Path:
    """Save API key to .env file."""
    env_file = find_env_file()

    # Read existing content
    existing_content = ""
    if env_file.exists():
        existing_content = env_file.read_text()

    # Check if key already exists
    lines = existing_content.splitlines()
    new_lines = []
    key_found = False

    for line in lines:
        if line.startswith(f"{env_var}="):
            new_lines.append(f"{env_var}={key}")
            key_found = True
        else:
            new_lines.append(line)

    if not key_found:
        new_lines.append(f"{env_var}={key}")

    # Write back
    env_file.write_text("\n".join(new_lines) + "\n")

    return env_file


def setup_interactive() -> bool:
    """
    Interactive setup wizard for testLLM.

    Returns True if setup was successful.
    """
    print("\n🔧 testLLM Setup\n")
    print("=" * 50)

    # Check for existing keys
    existing = check_existing_keys()
    if existing:
        print("✅ Found existing API key(s):")
        for provider, key in existing.items():
            print(f"   - {provider.title()}: {key}")
        print("\nYou're ready to use testLLM!")
        return True

    print("No API key found. Let's set one up (~1 minute):\n")
    print("Google AI Studio offers a free API with no credit card required.")
    print("")

    # Ask to open browser
    response = input("Open Google AI Studio in your browser? [Y/n]: ").strip().lower()
    if response != 'n':
        print(f"\n📱 Opening {GOOGLE_AI_STUDIO_URL}")
        try:
            webbrowser.open(GOOGLE_AI_STUDIO_URL)
        except Exception:
            print(f"   Could not open browser. Please visit:")
            print(f"   {GOOGLE_AI_STUDIO_URL}")
    else:
        print(f"\nPlease visit: {GOOGLE_AI_STUDIO_URL}")

    print("\nSteps:")
    print("  1. Sign in with your Google account")
    print("  2. Click 'Create API Key'")
    print("  3. Copy the key\n")

    # Get the key
    api_key = input("Paste your API key here: ").strip()

    if not api_key:
        print("\n❌ No API key provided. Setup cancelled.")
        return False

    # Validate key format (basic check)
    if len(api_key) < 20:
        print("\n⚠️  That doesn't look like a valid API key (too short).")
        confirm = input("Save anyway? [y/N]: ").strip().lower()
        if confirm != 'y':
            return False

    # Save the key
    env_file = save_api_key(api_key, "GOOGLE_API_KEY")

    # Also set it in current environment
    os.environ["GOOGLE_API_KEY"] = api_key

    print(f"\n✅ API key saved to {env_file}")
    print("\nYou're ready to use testLLM!")
    print("\nTry running your tests now, or use:")
    print("  from testllm import semantic_assert")
    print("  semantic_assert('Hello!', ['Response should be friendly'])")

    return True


def setup_check() -> bool:
    """
    Check if testLLM is configured. Returns True if ready to use.

    Prints helpful message if not configured.
    """
    existing = check_existing_keys()
    if existing:
        return True

    print("\n⚠️  testLLM is not configured yet.")
    print("\nQuick setup (1 minute, no credit card):")
    print(f"  1. Visit: {GOOGLE_AI_STUDIO_URL}")
    print("  2. Sign in with Google and click 'Create API Key'")
    print("  3. Set the environment variable:")
    print("     export GOOGLE_API_KEY='your-key-here'")
    print("\nOr run: python -m testllm.setup")

    return False


def main():
    """CLI entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(0 if setup_check() else 1)
    else:
        success = setup_interactive()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
