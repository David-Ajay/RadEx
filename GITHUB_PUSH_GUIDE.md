# 📤 GitHub Push Guide for RadEx

This guide will help you push the RadEx project to GitHub properly, excluding the large dataset files.

## ✅ Pre-Push Checklist

- [x] `.gitignore` updated to exclude data files (44GB excluded)
- [x] README.md updated with comprehensive documentation
- [x] LICENSE file added (MIT License)
- [x] CONTRIBUTING.md added
- [x] GitHub Actions workflow added
- [x] Repository size manageable (~600MB .git folder)

---

## 🚀 Step-by-Step Push Instructions

### Step 1: Configure Git (If Not Already Done)

```bash
cd /Users/david/Downloads/radexplain

# Set your name and email
git config --global user.name "David Ajay"
git config --global user.email "your-email@example.com"

# Verify configuration
git config --global --list
```

### Step 2: Create GitHub Repository

1. Go to: https://github.com/new
2. **Repository name**: `RadEx`
3. **Description**: "An Explainable Multi-Organ AI Framework for Chest X-Ray Diagnosis, Report Generation and Conversational Assistance"
4. **Visibility**: Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### Step 3: Review What Will Be Pushed

```bash
cd /Users/david/Downloads/radexplain

# Check current status
git status

# See what's being tracked (should be ~155 files)
git ls-files | wc -l

# Verify data is NOT tracked
git ls-files | grep "data/" | head -20

# Check repository size (should be ~600MB, not 47GB)
du -sh .git
```

### Step 4: Stage and Commit Changes

```bash
cd /Users/david/Downloads/radexplain

# Stage all changes
git add .

# Check what will be committed
git status

# Commit the changes
git commit -m "docs: Update README, add LICENSE and contributing guidelines

- Comprehensive README with installation, usage, and performance metrics
- MIT License
- Contributing guidelines
- GitHub Actions CI workflow
- Improved .gitignore to exclude datasets and large files"
```

### Step 5: Add Remote and Push

```bash
cd /Users/david/Downloads/radexplain

# Add GitHub remote (replace with your actual repository URL)
git remote add origin https://github.com/David-Ajay/RadEx.git

# Verify remote was added
git remote -v

# Push to GitHub (first time)
git push -u origin main
```

**Note**: You'll be prompted for GitHub credentials:
- **Username**: David-Ajay
- **Password**: Use a **Personal Access Token** (not your password)

### Step 6: Create Personal Access Token (If Needed)

If you don't have a token:

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name: "RadEx Push Token"
4. Select scopes:
   - ✅ `repo` (Full control of private repositories)
5. Click "Generate token"
6. **COPY THE TOKEN** (you won't see it again!)
7. Use this token as your password when pushing

---

## 🔐 Alternative: Use GitHub CLI (Easier Authentication)

### Option A: Using GitHub CLI (Recommended)

```bash
# Install GitHub CLI (if not installed)
brew install gh

# Authenticate
gh auth login
# Choose: GitHub.com → HTTPS → Yes to authenticate → Login with browser

# Create repository directly from terminal
cd /Users/david/Downloads/radexplain
gh repo create David-Ajay/RadEx --public --source=. --remote=origin

# Push
git push -u origin main
```

---

## 📊 What Gets Pushed vs. Excluded

### ✅ Included (~600MB)
- Source code (`src/radexplain/`)
- Configuration files (`configs/`)
- Scripts (`scripts/`)
- Documentation (`README.md`, `DATA_SETUP.md`, etc.)
- Requirements (`requirements.txt`, `pyproject.toml`)
- Small assets (`assets/xray_hero.png`)
- Checkpoint configs (`checkpoints/*/postprocess.json`)
- Directory structure (empty `.gitkeep` files)

### ❌ Excluded (~44GB+)
- Raw datasets (`data/raw/`)
- Processed images (`data/processed/`)
- Label CSV files (`data/labels/*.csv`)
- Database files (`data/*.db`)
- Model weights (`checkpoints/*/*.pt`)
- Training outputs (`outputs/*/history.json`, `*.png`)
- Python cache (`__pycache__/`, `*.pyc`)
- Virtual environment (`.venv/`)
- macOS files (`.DS_Store`)
- Environment variables (`.env`)

---

## 🔍 Verify Push Success

After pushing, check:

1. **GitHub Repository**: https://github.com/David-Ajay/RadEx
2. **Verify files are there**: README.md, LICENSE, src/, etc.
3. **Check repository size**: Should show ~1-2 MB (compressed)
4. **Verify data is excluded**: No `data/raw/` or `data/processed/` folders

---

## 🎉 Post-Push Tasks

### 1. Add Repository Description

On GitHub repository page:
- Click ⚙️ (Settings) → About → Edit
- Add description and topics

**Suggested Topics**:
- `chest-xray`
- `medical-imaging`
- `deep-learning`
- `pytorch`
- `computer-vision`
- `radiology`
- `explainable-ai`
- `multi-label-classification`
- `healthcare-ai`

### 2. Create Release

```bash
# Tag the current version
git tag -a v1.0.0 -m "Initial release: RadEx v1.0.0"
git push origin v1.0.0
```

Then on GitHub:
- Go to Releases → "Create a new release"
- Choose tag: v1.0.0
- Title: "RadEx v1.0.0 - Initial Release"
- Add release notes

### 3. Add Badges to README

GitHub will auto-generate badge URLs after first push.

### 4. Enable GitHub Pages (Optional)

If you want to host documentation:
- Settings → Pages → Source: main branch → /docs folder

---

## 🐛 Troubleshooting

### Problem: "fatal: repository not found"
**Solution**: Check repository name and URL are correct

### Problem: "failed to push some refs"
**Solution**: Pull first: `git pull origin main --rebase`

### Problem: "authentication failed"
**Solution**: Use Personal Access Token instead of password

### Problem: Push is too slow
**Solution**: Your connection speed; ~600MB upload will take time
- On 10 Mbps upload: ~8 minutes
- On 50 Mbps upload: ~2 minutes

### Problem: "remote: error: File is too large"
**Solution**: Check if large files are accidentally tracked
```bash
git ls-files | xargs ls -lh | sort -k5 -h -r | head -20
```

### Problem: Need to remove large files from history
**Solution**: Use git-filter-repo (DO THIS BEFORE FIRST PUSH)
```bash
pip install git-filter-repo
git filter-repo --path data/ --invert-paths
```

---

## 📝 Maintenance Commands

### Update .gitignore Later
```bash
# If you accidentally tracked large files
git rm -r --cached data/raw/
git commit -m "Remove tracked dataset files"
git push origin main
```

### Check Repository Health
```bash
# See largest files in git history
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sed -n 's/^blob //p' | sort -n -k 2 -r | head -20

# Check total repository size
git count-objects -vH
```

---

## 🎓 Summary

**Quick Commands (After Setting Up GitHub Repo)**:

```bash
cd /Users/david/Downloads/radexplain
git add .
git commit -m "docs: Comprehensive documentation and project setup"
git remote add origin https://github.com/David-Ajay/RadEx.git
git push -u origin main
```

Or with GitHub CLI:

```bash
cd /Users/david/Downloads/radexplain
gh auth login
gh repo create David-Ajay/RadEx --public --source=. --remote=origin
git add .
git commit -m "docs: Comprehensive documentation and project setup"
git push -u origin main
```

---

## ✅ Done!

Your RadEx project is now on GitHub! 🎉

**Next Steps**:
1. Share the repository link
2. Add collaborators if needed
3. Enable issue tracking
4. Set up project board for tasks
5. Add contribution guidelines

**Repository URL**: https://github.com/David-Ajay/RadEx

---

**Need Help?** Open an issue or check GitHub documentation: https://docs.github.com
