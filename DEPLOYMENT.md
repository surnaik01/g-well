# Deployment Guide

## GitHub Pages (Static Website)

The landing pages (index.html, demo.html) are deployed on GitHub Pages.

**URL:** https://surnaik01.github.io/g-well/

### Enable GitHub Pages:
1. Go to: https://github.com/surnaik01/g-well/settings/pages
2. Source: "Deploy from a branch"
3. Branch: `gh-pages`
4. Folder: `/ (root)`
5. Click "Save"

## Gradio App Deployment

The Gradio app (app.py) needs a Python server. Here are deployment options:

### Option 1: Gradio Share (Easiest)
```bash
python app.py
# When it starts, it will show a shareable link like:
# https://xxxxx.gradio.app
```

### Option 2: Hugging Face Spaces (Free)
1. Go to: https://huggingface.co/spaces
2. Create new Space
3. Upload app.py and requirements.txt
4. It will auto-deploy

### Option 3: Railway/Render (Free tier available)
- Connect your GitHub repo
- Set start command: `python app.py`
- Auto-deploys on push

### Option 4: Local Development
```bash
python app.py
# Access at: http://localhost:7860
```

## Update Demo Page

Once you have a hosted Gradio URL, update `demo.html`:
```html
<iframe src="YOUR_GRADIO_URL"></iframe>
```

