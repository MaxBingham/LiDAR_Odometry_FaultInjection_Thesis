#!/bin/bash
# Install Git hooks from scripts/ directory to .git/hooks/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

if [ ! -d "$HOOKS_DIR" ]; then
    echo "Error: .git/hooks directory not found"
    echo "Are you in a Git repository?"
    exit 1
fi

echo "Installing Git hooks..."

# Install post-checkout hook
if [ -f "$SCRIPT_DIR/post-checkout" ]; then
    cp "$SCRIPT_DIR/post-checkout" "$HOOKS_DIR/post-checkout"
    chmod +x "$HOOKS_DIR/post-checkout"
    echo "✅ Installed post-checkout hook"
else
    echo "❌ post-checkout hook not found in scripts/"
fi

echo ""
echo "Git hooks installed successfully!"
echo "The post-checkout hook will now run automatically after every 'git checkout'."
echo ""
echo "To temporarily skip the hook, use: git checkout --no-verify <branch>"
