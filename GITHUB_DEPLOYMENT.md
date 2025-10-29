# 🚀 GitHub Deployment Guide

Complete step-by-step instructions to upload PropertyRAG to GitHub.

---

## 📋 Prerequisites Checklist

Before uploading to GitHub, ensure:
- ✅ Git is installed on your system
- ✅ You have a GitHub account
- ✅ Virtual environment (`venv/`) is NOT being tracked
- ✅ `.env` file is NOT being tracked (contains secrets)
- ✅ All log files are excluded
- ✅ `.gitignore` file is present

---

## 🔧 Step 1: Verify Git Installation

**Check if Git is installed**:
```bash
git --version
```

**If not installed, install Git**:
- **Windows**: Download from https://git-scm.com/download/win
- **macOS**: `brew install git` or download from https://git-scm.com/
- **Linux**: `sudo apt-get install git` (Ubuntu/Debian) or `sudo yum install git` (CentOS/RHEL)

---

## 🌐 Step 2: Create GitHub Repository

1. **Go to GitHub**: https://github.com/
2. **Click** the `+` icon in the top-right corner
3. **Select** "New repository"
4. **Fill in details**:
   - **Repository name**: `PropertyRAG` (or your preferred name)
   - **Description**: "AI-Powered Real Estate Intelligence Platform with RAG"
   - **Visibility**: 
     - ✅ **Public** (recommended for portfolio/open-source)
     - Or **Private** (if you want to keep it private)
   - **DO NOT** initialize with README (we already have one)
   - **DO NOT** add .gitignore (we already have one)
   - **DO NOT** choose a license (we already have one)
5. **Click** "Create repository"

**Copy the repository URL** (you'll need it in Step 4):
```
https://github.com/YOUR_USERNAME/PropertyRAG.git
```

---

## 📁 Step 3: Prepare Your Local Repository

**Navigate to your project directory**:
```bash
cd c:\Users\hanza\OneDrive\Desktop\PropertyRAG\project
```

**Verify files are clean**:
```bash
# Check what files exist
dir  # Windows
# or
ls -la  # macOS/Linux

# Ensure these are present:
# ✅ .gitignore
# ✅ .env.example
# ✅ README.md
# ✅ LICENSE
# ✅ requirements.txt
# ✅ main.py, app.py, advanced_rag.py, ingest.py

# Ensure these are NOT present or will be ignored:
# ❌ .env (secret keys)
# ❌ venv/ (virtual environment)
# ❌ __pycache__/ (Python cache)
# ❌ *.log files
# ❌ *.pyc files
```

---

## 🎯 Step 4: Initialize Git and Push to GitHub

### 4.1 Initialize Git Repository

```bash
# Initialize git repository
git init
```

**Expected Output**:
```
Initialized empty Git repository in C:/Users/hanza/OneDrive/Desktop/PropertyRAG/project/.git/
```

### 4.2 Configure Git (First Time Only)

**Set your name and email** (will appear in commits):
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Verify configuration**:
```bash
git config --global user.name
git config --global user.email
```

### 4.3 Add All Files

```bash
# Add all files (respecting .gitignore)
git add .
```

**Verify what will be committed**:
```bash
# Check status
git status
```

**Expected Output** (verify these files are staged):
```
On branch main

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   .env.example
        new file:   .gitignore
        new file:   LICENSE
        new file:   README.md
        new file:   advanced_rag.py
        new file:   app.py
        new file:   data/property_data_cleaned.csv
        new file:   ingest.py
        new file:   main.py
        new file:   requirements.txt
        new file:   test_production_api.py
        new file:   PRODUCTION_UPGRADE_SUMMARY.md
        new file:   SETUP_INSTRUCTIONS.md
```

**❗ IMPORTANT**: Ensure `.env` is NOT listed (it should be ignored)

### 4.4 Create Initial Commit

```bash
git commit -m "Initial commit: Production-grade PropertyRAG system

- Advanced RAG system with Gemini + Pinecone
- Modern Streamlit UI with dark mode
- Sophisticated query understanding and intent classification
- Multi-turn conversation support
- Comprehensive documentation"
```

**Expected Output**:
```
[main (root-commit) abc1234] Initial commit: Production-grade PropertyRAG system
 13 files changed, 5000+ insertions(+)
 create mode 100644 .env.example
 create mode 100644 .gitignore
 create mode 100644 LICENSE
 ...
```

### 4.5 Add Remote Repository

**Replace `YOUR_USERNAME` with your actual GitHub username**:
```bash
git remote add origin https://github.com/YOUR_USERNAME/PropertyRAG.git
```

**Verify remote**:
```bash
git remote -v
```

**Expected Output**:
```
origin  https://github.com/YOUR_USERNAME/PropertyRAG.git (fetch)
origin  https://github.com/YOUR_USERNAME/PropertyRAG.git (push)
```

### 4.6 Rename Branch to Main (if needed)

```bash
# Check current branch name
git branch

# Rename to main if it's master
git branch -M main
```

### 4.7 Push to GitHub

```bash
git push -u origin main
```

**Expected Output**:
```
Enumerating objects: 15, done.
Counting objects: 100% (15/15), done.
Delta compression using up to 8 threads
Compressing objects: 100% (13/13), done.
Writing objects: 100% (15/15), 50.23 KiB | 5.02 MiB/s, done.
Total 15 (delta 2), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (2/2), done.
To https://github.com/YOUR_USERNAME/PropertyRAG.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

---

## ✅ Step 5: Verify Upload on GitHub

1. **Go to your repository**: https://github.com/YOUR_USERNAME/PropertyRAG
2. **Verify files are present**:
   - ✅ README.md is displayed
   - ✅ All Python files are visible
   - ✅ .gitignore is present
   - ✅ LICENSE is present
3. **Check that secrets are NOT uploaded**:
   - ❌ .env file should NOT be visible
   - ❌ venv/ should NOT be visible
   - ❌ Log files should NOT be visible

---

## 🔄 Making Future Updates

### After making changes to your code:

```bash
# 1. Check what changed
git status

# 2. Add changes
git add .
# Or add specific files:
# git add main.py app.py

# 3. Commit with descriptive message
git commit -m "Add feature: Multi-turn conversation support"

# 4. Push to GitHub
git push origin main
```

### Common Git Commands:

```bash
# View commit history
git log --oneline

# View changes before committing
git diff

# Undo uncommitted changes
git checkout -- filename.py

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Create a new branch
git checkout -b feature/new-feature

# Switch branches
git checkout main
```

---

## 🔐 Security Best Practices

### ✅ DO:
- ✅ Keep `.env` file in `.gitignore`
- ✅ Use `.env.example` for template
- ✅ Rotate API keys if accidentally committed
- ✅ Use environment variables for secrets
- ✅ Review files before `git add .`

### ❌ DON'T:
- ❌ Commit API keys or passwords
- ❌ Commit `.env` file
- ❌ Commit large binary files (videos, models)
- ❌ Commit virtual environments (`venv/`)
- ❌ Push directly to main (use branches for large changes)

---

## 🚨 Emergency: If You Accidentally Committed Secrets

### If `.env` was committed but NOT pushed:

```bash
# Remove from staging
git reset HEAD .env

# Add to .gitignore
echo ".env" >> .gitignore

# Commit
git add .gitignore
git commit -m "Add .env to gitignore"
```

### If `.env` was committed AND pushed:

**⚠️ CRITICAL**: Your API keys are now public!

1. **Immediately rotate API keys**:
   - Gemini: https://makersuite.google.com/app/apikey (delete old key, create new)
   - Pinecone: https://app.pinecone.io/ (API Keys → Delete → Create new)

2. **Remove from Git history**:
```bash
# Install BFG Repo-Cleaner
# Download from: https://rtyley.github.io/bfg-repo-cleaner/

# Remove .env from history
bfg --delete-files .env

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push (rewrites history)
git push --force
```

3. **Update .env.example** (without real keys)

---

## 📊 GitHub Repository Settings

### Recommended Settings:

1. **Go to**: Settings → General
2. **Features**:
   - ✅ Enable Issues (for bug tracking)
   - ✅ Enable Discussions (for community)
   - ✅ Enable Wiki (for extended docs)
3. **Pull Requests**:
   - ✅ Enable "Automatically delete head branches"
4. **GitHub Pages** (optional):
   - Deploy documentation to GitHub Pages

---

## 🎨 Enhance Your Repository

### Add Badges to README

Already included in README.md:
- ![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
- ![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
- ![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)

### Create GitHub Topics

1. Go to repository homepage
2. Click "⚙️" (gear icon) next to "About"
3. Add topics:
   - `rag`
   - `ai`
   - `real-estate`
   - `gemini`
   - `pinecone`
   - `fastapi`
   - `streamlit`
   - `nlp`
   - `vector-search`
   - `python`

### Add Repository Description

In the "About" section:
```
🏡 AI-Powered Real Estate Intelligence Platform with advanced RAG, multi-turn conversations, and production-grade features
```

---

## 📝 Complete Command Reference

### Initial Setup (One-Time):
```bash
cd c:\Users\hanza\OneDrive\Desktop\PropertyRAG\project
git init
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git add .
git commit -m "Initial commit: Production-grade PropertyRAG system"
git remote add origin https://github.com/YOUR_USERNAME/PropertyRAG.git
git branch -M main
git push -u origin main
```

### Future Updates:
```bash
git status
git add .
git commit -m "Your commit message"
git push origin main
```

---

## 🎉 Success!

Your PropertyRAG system is now on GitHub! 🎊

**Next Steps**:
1. ✅ Share the repository URL with others
2. ✅ Add a GitHub Star ⭐ (self-promotion!)
3. ✅ Enable GitHub Actions for CI/CD (optional)
4. ✅ Create a demo video and add to README
5. ✅ Write a blog post about your project

**Repository URL**: https://github.com/YOUR_USERNAME/PropertyRAG

---

## 🆘 Troubleshooting

### Error: "fatal: not a git repository"
**Solution**: Run `git init` first

### Error: "remote origin already exists"
**Solution**: 
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/PropertyRAG.git
```

### Error: "failed to push some refs"
**Solution**:
```bash
# Pull first (if remote has changes)
git pull origin main --rebase

# Then push
git push origin main
```

### Error: "Permission denied (publickey)"
**Solution**: Set up SSH keys or use HTTPS with personal access token
- HTTPS: https://github.com/YOUR_USERNAME/PropertyRAG.git
- Token: https://github.com/settings/tokens

### Error: "Large files detected"
**Solution**: Add to .gitignore and use Git LFS for large files
```bash
git lfs install
git lfs track "*.csv"
```

---

## 📞 Need Help?

- **Git Documentation**: https://git-scm.com/doc
- **GitHub Guides**: https://guides.github.com/
- **GitHub Support**: https://support.github.com/

---

**Happy Coding! 🚀**
