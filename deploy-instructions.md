# Full Functionality Deployment Instructions

## For Render.com with Java Support

### Option 1: Using Dockerfile (Recommended)
1. **Copy the full requirements:**
   ```bash
   cp requirements-java.txt requirements.txt
   ```

2. **Deploy using the provided Dockerfile:**
   - The Dockerfile installs Java automatically
   - Includes tabula-py with full functionality
   - Configured for production deployment

3. **Set environment variables in Render:**
   ```
   OPENAI_API_KEY=your-openai-key
   ```

### Option 2: Using Native Build
1. **Add Java buildpack to Render:**
   - In Render dashboard: Settings → Build & Deploy
   - Add buildpack: `heroku/jvm`
   - Set build command: `pip install -r requirements.txt`

2. **Create aptfile for system dependencies:**
   ```
   default-jre
   default-jdk
   ```

3. **Use requirements-java.txt:**
   ```bash
   cp requirements-java.txt requirements.txt
   ```

### Option 3: Heroku with Java Support
1. **Add Java buildpack:**
   ```bash
   heroku buildpacks:add heroku/jvm
   heroku buildpacks:add heroku/python
   ```

2. **Deploy with full requirements:**
   ```bash
   cp requirements-java.txt requirements.txt
   git add .
   git commit -m "Add Java support"
   git push heroku main
   ```

## Expected First Run Behavior:
- App starts successfully
- First PDF upload triggers tabula-py JAR download (30-60 seconds)
- Subsequent uploads are fast (uses cached JARs)
- Full table extraction capabilities available

## Verification:
Upload a PDF with tables and verify:
- ✅ Text extraction works
- ✅ Table extraction from tabula-py works
- ✅ No Java-related error messages
- ✅ All ResearchMiner features operational

## Troubleshooting:
If Java still not available, the system gracefully falls back to pdfplumber-only table extraction while maintaining all other functionality.