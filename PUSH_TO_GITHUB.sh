#!/bin/bash
# Quick script to push RadEx to GitHub
# Run this script: bash PUSH_TO_GITHUB.sh

set -e  # Exit on error

echo "========================================="
echo "  RadEx GitHub Push Script"
echo "========================================="
echo ""

# Step 1: Configure git user (update with your details)
echo "📝 Step 1: Configuring Git..."
read -p "Enter your name (e.g., David Ajay): " GIT_NAME
read -p "Enter your email: " GIT_EMAIL

git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"
echo "✓ Git configured"
echo ""

# Step 2: Check current status
echo "📊 Step 2: Checking repository status..."
cd /Users/david/Downloads/radexplain
git status --short
echo ""

# Step 3: Verify remote
echo "🔗 Step 3: Checking remote URL..."
CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "none")
echo "Current remote: $CURRENT_REMOTE"

if [ "$CURRENT_REMOTE" == "https://github.com/David-Ajay/radexplain.git" ]; then
    echo "⚠️  Remote URL needs to be updated to 'RadEx'"
    read -p "Update remote URL to https://github.com/David-Ajay/RadEx.git? (y/n): " UPDATE_REMOTE
    if [ "$UPDATE_REMOTE" == "y" ]; then
        git remote remove origin
        git remote add origin https://github.com/David-Ajay/RadEx.git
        echo "✓ Remote URL updated"
    fi
elif [ "$CURRENT_REMOTE" == "none" ]; then
    echo "Adding remote..."
    git remote add origin https://github.com/David-Ajay/RadEx.git
    echo "✓ Remote added"
fi
echo ""

# Step 4: Verify what will be pushed
echo "📦 Step 4: Verifying files to push..."
FILE_COUNT=$(git ls-files | wc -l | tr -d ' ')
echo "Files tracked: $FILE_COUNT"
GIT_SIZE=$(du -sh .git | cut -f1)
echo "Repository size: $GIT_SIZE"
echo ""

# Check for large files
echo "Checking for large files..."
LARGE_FILES=$(git ls-files | xargs ls -lh 2>/dev/null | awk '$5 ~ /M$/ && $5 > 50 {print $9, $5}' || true)
if [ ! -z "$LARGE_FILES" ]; then
    echo "⚠️  Warning: Found large files:"
    echo "$LARGE_FILES"
    read -p "Continue anyway? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        echo "Aborted."
        exit 1
    fi
else
    echo "✓ No large files detected"
fi
echo ""

# Step 5: Push to GitHub
echo "🚀 Step 5: Ready to push!"
echo "========================================="
echo "Repository: https://github.com/David-Ajay/RadEx"
echo "Branch: main"
echo "========================================="
echo ""
read -p "Push to GitHub now? (y/n): " DO_PUSH

if [ "$DO_PUSH" == "y" ]; then
    echo ""
    echo "Pushing to GitHub..."
    echo "(You may be prompted for GitHub credentials)"
    echo ""
    
    git push -u origin main
    
    echo ""
    echo "========================================="
    echo "  ✅ SUCCESS! Project pushed to GitHub"
    echo "========================================="
    echo ""
    echo "View your repository at:"
    echo "https://github.com/David-Ajay/RadEx"
    echo ""
    echo "Next steps:"
    echo "1. Visit the repository and verify files"
    echo "2. Add repository description and topics"
    echo "3. Enable Issues and Discussions"
    echo "4. Create a release (git tag v1.0.0)"
    echo ""
else
    echo ""
    echo "Push cancelled. You can push manually with:"
    echo "  cd /Users/david/Downloads/radexplain"
    echo "  git push -u origin main"
    echo ""
fi
