# Deploy Full G-well App

## Option 1: Hugging Face Spaces (Easiest & Free) ⭐ RECOMMENDED

### Steps:
1. Go to: https://huggingface.co/spaces
2. Click "Create new Space"
3. Settings:
   - Space name: `g-well`
   - SDK: **Gradio**
   - Visibility: Public
   - Hardware: CPU (free)
4. Click "Create Space"
5. Upload `app.py` and `requirements.txt` via web UI or Git
6. Your app will auto-deploy!

**Live URL:** `https://surnaik01-g-well.hf.space`

---

## Option 2: Railway (Free Tier Available)

1. Go to: https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select `surnaik01/g-well`
5. Railway will auto-detect it's a Python app
6. Set start command: `python app.py`
7. Deploy!

---

## Option 3: Render (Free Tier)

1. Go to: https://render.com
2. Sign up with GitHub
3. New → Web Service
4. Connect `surnaik01/g-well` repo
5. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
6. Deploy!

---

## Option 4: Gradio Share (Temporary)

When running locally:
```bash
python app.py
# Look for: "Running on public URL: https://xxxxx.gradio.app"
# This link is valid for 72 hours
```

---

## After Deployment

1. Update `demo.html` iframe URL to your deployed app
2. Commit and push to update GitHub Pages
3. Your full app will be live!

**Landing Page:** https://surnaik01.github.io/g-well/
**Demo App:** (Your deployed Gradio URL)
