# Deployment Reference

Complete guide to running and deploying marimo applications.

## Running Locally

### Development Mode

```bash
# Edit notebook in browser (development interface)
marimo edit my_notebook.py

# Edit with custom port
marimo edit my_notebook.py --port 8080

# Edit without token authentication (less secure)
marimo edit my_notebook.py --no-token

# Edit with custom host (for remote access)
marimo edit my_notebook.py --host 0.0.0.0
```

### Production Mode

```bash
# Run as web app (read-only for users)
marimo run my_notebook.py

# Run with custom port
marimo run my_notebook.py --port 8080

# Run in headless mode
marimo run my_notebook.py --headless

# Run multiple notebooks
marimo run notebook1.py notebook2.py notebook3.py
```

### Script Execution

```bash
# Run as Python script (non-interactive)
python my_notebook.py

# Run with arguments
python my_notebook.py --arg1 value1 --arg2 value2
```

## Export Formats

### Static HTML Export

Creates a snapshot of the notebook with current output. Not interactive.

```bash
# Basic HTML export
marimo export html my_notebook.py -o output.html

# Include code in output
marimo export html my_notebook.py -o output.html --include-code
```

**Use cases:**
- Share results via email
- Embed in documentation
- Archive analysis results

### WASM-Powered Interactive Export

Creates a fully self-contained HTML file that runs Python in the browser via WebAssembly (Pyodide).

```bash
# Basic WASM export (readonly, code hidden)
uv run marimo export html-wasm notebook.py -o notebook.wasm.html

# Show code in output
uv run marimo export html-wasm notebook.py -o output.html --show-code

# Editable mode (users can modify code in browser)
uv run marimo export html-wasm notebook.py -o output.html --mode edit

# With Cloudflare Worker config
uv run marimo export html-wasm notebook.py -o output/ --include-cloudflare

# Watch mode (auto-rebuild on file change)
uv run marimo export html-wasm notebook.py -o output.html --watch
```

**WASM Features:**
- ✅ Fully interactive - UI elements work
- ✅ No Python server needed
- ✅ Runs entirely in browser
- ✅ Easy deployment to static hosts
- ⚠️ Limited to Pyodide-compatible packages
- ⚠️ Must be served over HTTP (not `file://`)
- ⚠️ Slower initial load (downloads Python runtime)
- ⚠️ Cannot access local file system

**Pyodide Package Compatibility:**

Check if your packages work with Pyodide:
- https://pyodide.org/en/stable/usage/packages-in-pyodide.html

Common supported packages:
- numpy, pandas, matplotlib, scipy
- scikit-learn, statsmodels
- plotly, altair
- requests, beautifulsoup4

Not supported:
- Packages with C extensions not compiled for WASM
- Packages requiring system access
- Most database drivers

### Script Export

Convert to standalone Python script.

```bash
# Export as .py script
marimo export script my_notebook.py -o script.py

# The output is a regular Python script with cell decorators removed
```

### Markdown Export

```bash
# Export to markdown
marimo export md my_notebook.py -o output.md

# Useful for documentation or version control
```

### Jupyter Notebook Export

```bash
# Export to Jupyter .ipynb format
marimo export ipynb my_notebook.py -o output.ipynb

# Useful for sharing with Jupyter users
```

## Testing WASM Exports Locally

WASM exports must be served over HTTP (not opened as local files).

### Option 1: Python HTTP Server

```bash
# Navigate to directory with exported file
cd output_directory

# Start simple HTTP server
python -m http.server 8000

# Open in browser
# http://localhost:8000/notebook.wasm.html
```

### Option 2: Cloudflare Worker

If you used `--include-cloudflare` flag:

```bash
cd output_directory
npx wrangler dev
```

### Option 3: VS Code Live Server

Install "Live Server" extension and right-click the HTML file.

## Deployment Options

### Static Hosting (WASM Exports)

#### GitHub Pages

```bash
# 1. Export notebook
uv run marimo export html-wasm notebook.py -o docs/index.html

# 2. Commit and push
git add docs/index.html
git commit -m "Deploy notebook"
git push

# 3. Enable GitHub Pages in repository settings
# Settings > Pages > Source: docs folder
```

#### Netlify

```bash
# 1. Export notebook
uv run marimo export html-wasm notebook.py -o dist/index.html

# 2. Deploy
netlify deploy --dir=dist --prod
```

#### Vercel

```bash
# 1. Export notebook
uv run marimo export html-wasm notebook.py -o public/index.html

# 2. Deploy
vercel --prod
```

### Server Deployment

#### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install marimo

# Copy notebook
COPY my_notebook.py .

# Expose port
EXPOSE 8080

# Run in production mode
CMD ["marimo", "run", "my_notebook.py", "--host", "0.0.0.0", "--port", "8080"]
```

Build and run:

```bash
docker build -t my-marimo-app .
docker run -p 8080:8080 my-marimo-app
```

#### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: marimo-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: marimo
  template:
    metadata:
      labels:
        app: marimo
    spec:
      containers:
      - name: marimo
        image: my-marimo-app:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: marimo-service
spec:
  selector:
    app: marimo
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

#### Cloud Run (GCP)

```bash
# Build and deploy
gcloud run deploy marimo-app \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Heroku

```bash
# Create Procfile
echo "web: marimo run my_notebook.py --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
heroku create my-marimo-app
git push heroku main
```

#### AWS EC2

```bash
# On EC2 instance
# Install marimo
pip install marimo

# Run with systemd
sudo tee /etc/systemd/system/marimo.service > /dev/null <<EOF
[Unit]
Description=Marimo App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/app
ExecStart=/usr/local/bin/marimo run my_notebook.py --host 0.0.0.0 --port 8080
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable marimo
sudo systemctl start marimo
```

## Environment Configuration

### Configuration File

Create `.marimo.toml` in project root:

```toml
[runtime]
# Auto-instantiate UI elements
auto_instantiate = true

# On module change behavior
on_module_change = "autorun"  # "autorun", "lazy", or "disabled"

[display]
# Theme
theme = "dark"  # "light" or "dark"

# Cell output area
output_renderer = "rich"

[completion]
# Code completion
activate_on_typing = true

[formatting]
# Auto-format on save
line_length = 88

[server]
# Server settings
browser = "default"
```

### Environment Variables

```bash
# Set via environment
export MARIMO_OUTPUT_RENDERER=rich
export MARIMO_THEME=dark

# Run notebook
marimo run notebook.py
```

## MCP Server Mode

Run notebook as MCP server for AI tool integration.

```bash
# Install MCP dependencies
pip install "marimo[mcp]"

# Run as MCP server
marimo edit notebook.py --mcp --no-token

# Add to Claude Code
claude mcp add --transport http marimo http://localhost:PORT/mcp/server
```

**Exposed tools:**
- `get_active_notebooks`
- `get_lightweight_cell_map`
- `get_cell_runtime_data`
- `get_tables_and_variables`
- `get_database_tables`
- `get_notebook_errors`

## Performance Optimization

### Lazy Loading

```python
# Only render when in viewport
mo.lazy(expensive_component)
```

### Caching

```python
@mo.cache
def expensive_computation(params):
    # Cached by params
    return results
```

### Memory Management

```python
# Explicit cleanup
large_object = load_large_data()
process(large_object)
del large_object
```

### Async Operations

```python
async def fetch_data():
    data = await api.get()
    return data

# marimo handles async cells automatically
result = await fetch_data()
```

## Security Considerations

### Production Deployment

1. **Always use authentication** (don't use `--no-token` in production)
2. **Use HTTPS** for production deployments
3. **Limit host access** (don't expose to 0.0.0.0 unless necessary)
4. **Set resource limits** in containerized environments
5. **Sanitize user inputs** in forms
6. **Review file upload permissions**

### WASM Exports

- WASM exports run client-side (no server security needed)
- Cannot access user's file system
- Cannot make unauthorized network requests
- Safe to share publicly

## Monitoring and Logging

### Application Logs

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Processing started")
```

### Health Check Endpoint

```python
# In notebook
def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}
```

### Metrics

Track usage with custom metrics:

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Metrics:
    page_loads: int = 0
    last_access: datetime = None

get_metrics, set_metrics = mo.state(Metrics())

# Increment on load
set_metrics(lambda m: replace(
    m,
    page_loads=m.page_loads + 1,
    last_access=datetime.now()
))
```

## Export Decision Matrix

| Use Case | Export Type | Interactive | Server Required | Shareable |
|----------|-------------|-------------|-----------------|-----------|
| Email results | `html` | No | No | Yes |
| Demo without server | `html-wasm` | Yes | No | Yes |
| Editable demo | `html-wasm --mode edit` | Yes | No | Yes |
| Production app | `run` | Yes | Yes | Via URL |
| Documentation | `md` | No | No | Yes |
| Jupyter users | `ipynb` | No | Jupyter | Yes |
| CI/CD scripts | `script` | No | No | Yes |
