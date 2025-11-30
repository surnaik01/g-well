# Deploy G-well to Hugging Face Spaces

## Quick Deploy (Recommended)

1. **Go to Hugging Face Spaces**: https://huggingface.co/spaces
2. **Click "Create new Space"**
3. **Fill in the form:**
   - Space name: `g-well` (or your preferred name)
   - SDK: **Gradio**
   - Visibility: Public or Private
   - Hardware: CPU (free) or GPU (if needed)
4. **Click "Create Space"**

## Upload Files

After creating the space, you can either:

### Option A: Upload via Web UI
1. Click "Files and versions" tab
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `README.md` (optional)

### Option B: Use Git (Recommended)
```bash
# Clone your HF space
git clone https://huggingface.co/spaces/surnaik01/g-well
cd g-well

# Copy files from your local project
cp /Users/suraga/Desktop/Gradio/app.py .
cp /Users/suraga/Desktop/Gradio/requirements.txt .
cp /Users/suraga/Desktop/Gradio/README.md .

# Commit and push
git add .
git commit -m "Deploy G-well app"
git push
```

## Your App Will Be Live At

**https://huggingface.co/spaces/surnaik01/g-well**

## Update Demo Page

Once deployed, update `demo.html`:
```html
<iframe src="https://surnaik01-g-well.hf.space"></iframe>
```

## Notes

- The model will be created automatically on first run
- Hugging Face provides free CPU/GPU resources
- Auto-deploys on every push
- Public spaces are free forever

