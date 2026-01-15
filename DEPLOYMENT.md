# 🚀 How to Deploy to Hugging Face Spaces

This guide will help you deploy your **Aadhaar Insights Dashboard** to the web using Hugging Face Spaces. it is free and very easy!

## Step 1: Create a Hugging Face Account
If you don't have one, sign up at [huggingface.co/join](https://huggingface.co/join).

## Step 2: Create a New Space
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. **Owner**: Select your username.
3. **Space Name**: Enter `aadhaar-insights` (or any name you like).
4. **License**: Choose `MIT`.
5. **SDK**: Select **Streamlit**.
6. **Hardware**: Select **CPU basic (Free)**.
7. Click **Create Space**.

## Step 3: Upload Your Files
You have two ways to upload files. The easiest way for you right now is via the browser.

### Option A: Upload via Browser (Easiest)
1. In your new Space, click on the **Files** tab.
2. Click on **+ Add file** > **Upload files**.
3. Drag and drop the following files from your `aadhaar-insights` folder:
   - `app.py`
   - `requirements.txt`
   - `aadhaar_data.csv`
   - `india_states.geojson`
   - `README.md` (Note: This might replace the default readme, which is fine, but ensure the metadata at the top is preserved or just edit the existing one).
   
   > **Important**: You do NOT need to upload the usage `.venv` folder or `.git` folder.

4. In the "Commit changes" box, type "Initial deploy" and click **Commit changes to main**.

### Option B: Upload via Git (If you have Git installed)
1. Clone your space locally:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/aadhaar-insights-dashboard
   ```
2. Copy your project files (`app.py`, `requirements.txt`, etc.) into this new folder.
3. Push changes:
   ```bash
   git add .
   git commit -m "Initial deploy"
   git push
   ```

## Step 4: Watch it Build!
1. Click on the **App** tab in your Space.
2. You will see a "Building" status. Since we removed heavy libraries, it should be fast!
3. Once done, your app will be live on the web! 🎉
