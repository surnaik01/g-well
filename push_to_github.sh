#!/bin/bash
# Script to create and push to GitHub

REPO_NAME="g-well"
USERNAME="surnaik01"

echo "Creating repository on GitHub..."
echo ""
echo "Please create a Personal Access Token at:"
echo "https://github.com/settings/tokens"
echo ""
echo "Select scopes: repo (all)"
echo ""
read -p "Enter your GitHub Personal Access Token: " TOKEN

if [ -z "$TOKEN" ]; then
    echo "Token required. Exiting."
    exit 1
fi

# Create repository via API
echo "Creating repository..."
curl -X POST \
  -H "Authorization: token $TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/user/repos \
  -d "{\"name\":\"$REPO_NAME\",\"description\":\"Plant disease detection app with real-time AI inference\",\"public\":true}" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Repository created!"
    
    # Set remote and push
    git remote remove origin 2>/dev/null
    git remote add origin https://$TOKEN@github.com/$USERNAME/$REPO_NAME.git
    git branch -M main
    git push -u origin main
    
    echo "✅ Code pushed to GitHub!"
    echo "Repository: https://github.com/$USERNAME/$REPO_NAME"
else
    echo "❌ Failed to create repository"
    exit 1
fi
