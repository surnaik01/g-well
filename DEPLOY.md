# Deploy to GitHub - Quick Guide

## Method 1: Using GitHub CLI (Fastest)

1. Authenticate (one-time):
   ```bash
   gh auth login
   ```
   Follow the prompts (choose GitHub.com, HTTPS, authenticate in browser)

2. Create and push:
   ```bash
   cd /Users/suraga/Desktop/Gradio
   gh repo create g-well --public --source=. --remote=origin --description "Plant disease detection app with real-time AI inference" --push
   ```

## Method 2: Manual (No CLI needed)

1. Create repo at: https://github.com/new
   - Name: `g-well`
   - Public
   - Don't initialize with README

2. Push code:
   ```bash
   cd /Users/suraga/Desktop/Gradio
   git remote add origin https://github.com/surnaik01/g-well.git
   git branch -M main
   git push -u origin main
   ```

Done! Your repo will be at: https://github.com/surnaik01/g-well
