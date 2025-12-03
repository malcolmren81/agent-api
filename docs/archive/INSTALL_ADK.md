# Google ADK Installation Instructions

## Package Information

**Package**: `google-adk`
**Latest Version**: 1.16.0
**PyPI**: https://pypi.org/project/google-adk/
**Status**: ✅ Available on PyPI

## Installation

### For Cloud Run Deployment (Automatic)

The `google-adk` package is now included in `requirements.txt` and will be automatically installed during Cloud Run deployment:

```bash
cd services/agents-api
gcloud run deploy palet8-agents --source . --region=us-central1
```

The Cloud Run build process will:
1. Read `requirements.txt`
2. Install all packages including `google-adk==1.16.0`
3. Build and deploy the service

### For Local Development

#### Option 1: Virtual Environment (Recommended)

```bash
cd services/agents-api

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import google.adk; print(google.adk.__version__)"
```

#### Option 2: Docker (Alternative)

```bash
cd services/agents-api

# Build Docker image
docker build -t palet8-agents .

# Run container
docker run -p 8000:8000 palet8-agents

# Check logs
docker logs <container-id>
```

#### Option 3: System-Wide (Not Recommended on macOS)

```bash
# Only if you understand the risks
pip3 install --break-system-packages google-adk==1.16.0
```

⚠️ **Warning**: Using `--break-system-packages` can interfere with system Python. Use virtual environments instead.

## Verification

After installation, verify that google-adk is available:

```python
import google.adk
from google.adk import Agent
from google.adk.agents import SequentialAgent, ParallelAgent, LoopAgent

print(f"Google ADK version: {google.adk.__version__}")
```

Expected output:
```
Google ADK version: 1.16.0
```

**Note**: Workflow agents (SequentialAgent, ParallelAgent, LoopAgent) are in `google.adk.agents`, not `google.adk.workflow`.

## Migration Status

### Current State
- ✅ `google-adk==1.16.0` added to requirements.txt
- ⚠️ Code still uses local ADK-compliant implementation
- ✅ System is 100% ADK-compliant (see GOOGLE_ADK_COMPLIANCE.md)

### Next Steps

1. **Deploy with ADK Package** (no code changes needed yet)
   - Deploy to Cloud Run with updated requirements.txt
   - Package will be installed automatically
   - Current code will continue working (doesn't import google.adk yet)

2. **Code Migration** (optional, when ready)
   - Follow migration checklist in GOOGLE_ADK_COMPLIANCE.md
   - Update imports to use `from google.adk import ...`
   - Replace local classes with official ADK classes
   - Test thoroughly

3. **Benefits of Migration**
   - Official Google support
   - Automatic updates
   - Community contributions
   - Potential performance improvements

## Current Architecture

Our current implementation uses ADK-compliant local classes. The system works perfectly as-is and **no immediate code changes are required**.

The google-adk package is included in requirements.txt for:
- Future migration convenience
- Dependency resolution
- Documentation purposes
- Cloud Run deployment preparation

## Troubleshooting

### Error: "No module named 'google.adk'"

This is expected if the code hasn't been migrated yet. Our current code uses local implementations in `src/agents/base_agent.py`, not the google-adk package.

To use the official package, follow the migration steps in GOOGLE_ADK_COMPLIANCE.md.

### Installation fails with "externally-managed-environment"

Use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Package conflicts

Check your Python version:
```bash
python3 --version  # Should be Python 3.10+
```

If issues persist, create a fresh virtual environment:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## References

- Google ADK Documentation: https://google.github.io/adk
- PyPI Package: https://pypi.org/project/google-adk/
- Compliance Documentation: ./GOOGLE_ADK_COMPLIANCE.md

## Support

For installation issues:
1. Check that Python 3.10+ is installed
2. Use a virtual environment
3. Verify internet connection for PyPI access
4. Check firewall settings if behind corporate proxy
