# GitHub Setup Instructions

## Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `g-well` (or any name you prefer)
3. Description: "Plant disease detection app with real-time AI inference"
4. Choose: Public or Private
5. DO NOT initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Connect and Push

After creating the repository, run these commands:

```bash
cd /Users/suraga/Desktop/Gradio

# Set the remote (replace with your actual repo name if different)
git remote add origin https://github.com/surnaik01/g-well.git

# Or if you used a different name:
# git remote add origin https://github.com/surnaik01/YOUR-REPO-NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Alternative: Using SSH (if you have SSH keys set up)

```bash
git remote add origin git@github.com:surnaik01/g-well.git
git branch -M main
git push -u origin main
```

## What's Included

- ✅ All source code (app.py, index.html, demo.html)
- ✅ Requirements.txt
- ✅ README.md
- ✅ .gitignore (excludes models, cache, etc.)
- ✅ Training scripts

## What's Excluded (via .gitignore)

- Model files (*.pth, *.pt)
- Python cache (__pycache__)
- Virtual environments
- Log files
- Environment variables

## Note

The model file (models/plant_disease_model.pth) is excluded because it can be large.
Users can generate it by running: `python train_model.py`
