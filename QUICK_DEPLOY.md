# 🚀 Quick Deploy Guide

## Deploy to Hugging Face Spaces (5 minutes)

### Step 1: Create Space
1. Go to: **https://huggingface.co/spaces**
2. Click **"Create new Space"**
3. Fill in:
   - **Space name:** `g-well`
   - **SDK:** Select **"Gradio"**
   - **Visibility:** Public
   - **Hardware:** CPU Basic (free)
4. Click **"Create Space"**

### Step 2: Upload Files
You have 2 options:

#### Option A: Web UI (Easiest)
1. In your new space, click **"Files and versions"** tab
2. Click **"Add file"** → **"Upload files"**
3. Upload these files:
   - `app.py`
   - `requirements.txt`
4. Wait 2-3 minutes for auto-deployment

#### Option B: Git (Recommended)
```bash
# Get your HF space Git URL (shown on space page)
git clone https://huggingface.co/spaces/surnaik01/g-well
cd g-well

# Copy files
cp /Users/suraga/Desktop/Gradio/app.py .
cp /Users/suraga/Desktop/Gradio/requirements.txt .

# Push
git add .
git commit -m "Deploy G-well"
git push
```

### Step 3: Your App is Live!
Your app will be at: **https://surnaik01-g-well.hf.space**

### Step 4: Update Demo Page
Once deployed, update `demo.html`:
```html
<iframe src="https://surnaik01-g-well.hf.space"></iframe>
```

Then commit and push to update GitHub Pages.

---

## Alternative: Railway (Also Free)

1. Go to: **https://railway.app**
2. Sign up with GitHub
3. **New Project** → **Deploy from GitHub repo**
4. Select `surnaik01/g-well`
5. Railway auto-detects Python
6. Set start command: `python app.py`
7. Deploy!

---

## What Gets Deployed

✅ **Gradio App** (app.py) - Full AI model with real-time inference
✅ **Landing Pages** (index.html, demo.html) - Already on GitHub Pages
✅ **All Features** - Disease detection, validation, recommendations

---

## Result

- **Landing Page:** https://surnaik01.github.io/g-well/
- **Demo App:** https://surnaik01-g-well.hf.space (after HF deployment)

🎉 Your complete G-well app will be live!

