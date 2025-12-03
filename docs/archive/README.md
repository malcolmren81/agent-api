# Agents API Documentation

Documentation for deploying and maintaining the palet8-agents Cloud Run service.

---

## Quick Links

- **[Deployment Guide](./DEPLOYMENT_GUIDE.md)** - Complete step-by-step deployment instructions
- **[Common Mistakes](./COMMON_MISTAKES.md)** - Quick reference of mistakes and solutions
- **[Changelog](./CHANGELOG.md)** - Deployment history and lessons learned

---

## Current Production Status

**Service:** palet8-agents
**Platform:** Google Cloud Run
**Region:** us-central1
**Current Revision:** palet8-agents-00170-wdk
**Status:** ✅ Stable
**Last Updated:** November 19, 2025

**Service URL:** https://palet8-agents-702210710671.us-central1.run.app

---

## Quick Start

### Deploy
```bash
cd /Users/malcolmyam/Desktop/alpha\ project/agent\ system/services/agents-api
gcloud run deploy palet8-agents --source . --region us-central1 --project palet8-system --allow-unauthenticated
```

### Verify
```bash
# Check which revision is serving
gcloud run services describe palet8-agents --region=us-central1 --project=palet8-system --format="value(status.latestReadyRevisionName)"

# Test health
curl https://palet8-agents-702210710671.us-central1.run.app/health
```

### Monitor
```bash
# View logs
gcloud run services logs read palet8-agents --region=us-central1 --project=palet8-system --limit=50

# Watch for errors
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=palet8-agents" --limit=100 | grep -E "❌|ERROR"
```

---

## Most Common Issues

| Issue | Solution | Doc Link |
|-------|----------|----------|
| Build fails with 500 error | Use direct Node.js download | [Mistake #1](./COMMON_MISTAKES.md#mistake-1-external-package-repository-failures) |
| Database not working | Copy `/root/.cache` from builder | [Mistake #2](./COMMON_MISTAKES.md#mistake-2-missing-prisma-query-engine-binary) |
| Container startup timeout | Run Prisma at build time | [Mistake #3](./COMMON_MISTAKES.md#mistake-3-runtime-prisma-generation-causing-startup-timeout) |
| Changes not visible | Verify revision number | [Mistake #7](./COMMON_MISTAKES.md#mistake-7-testing-old-revision-instead-of-new-deployment) |
| No Task records | Add delay + retry logic | [Mistake #5](./COMMON_MISTAKES.md#mistake-5-background-task-race-conditions) |

---

## Documentation Files

### [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
Comprehensive deployment guide covering:
- Prerequisites and setup
- Step-by-step deployment procedure
- Detailed mistake explanations with solutions
- Troubleshooting procedures
- Rollback instructions
- Monitoring and verification
- Best practices

**Read this when:**
- Deploying for the first time
- Encountering deployment issues
- Need detailed troubleshooting steps
- Want to understand why things work certain ways

---

### [COMMON_MISTAKES.md](./COMMON_MISTAKES.md)
Quick reference of the 7 most common mistakes:
- Critical mistakes (will break deployment)
- Verification mistakes (false positives)
- Non-critical mistakes (data loss but service runs)
- Pre/post-deployment checklists
- Troubleshooting decision tree
- Quick fixes table

**Read this when:**
- You need a quick answer
- Something broke and you need to fix it fast
- You want to avoid known pitfalls
- You're reviewing code before deployment

---

### [CHANGELOG.md](./CHANGELOG.md)
Historical record of all deployments:
- Revision history with status
- What changed in each deployment
- Issues encountered and resolutions
- Lessons learned
- Known good configurations

**Read this when:**
- Planning a rollback
- Understanding past issues
- Learning from previous deployments
- Documenting new changes

---

## Architecture Overview

### Multi-Stage Docker Build
```
Builder Stage (python:3.11)
├── Install system dependencies
├── Install Python packages to /root/.local
├── Generate Prisma client (creates files in /root/.cache)
└── Output: /root/.local + /root/.cache

Runtime Stage (python:3.11)
├── Install Node.js (direct download)
├── Copy /root/.local (Python packages)
├── Copy /root/.cache (Prisma binaries) ← Critical!
├── Copy application code
└── CMD: uvicorn (direct, fast startup)
```

### Key Components
- **FastAPI** - Web framework
- **Prisma** - Database ORM (Python client)
- **Uvicorn** - ASGI server
- **Node.js** - Required by Prisma
- **PostgreSQL** - Database (Cloud SQL)

### External Dependencies
- Google Gemini API
- OpenAI API
- Flux API (image generation)
- Admin API (internal)

---

## Critical Files

### Dockerfile
**Location:** `/services/agents-api/Dockerfile`

**Critical Sections:**
```dockerfile
# Builder: Set PATH for Prisma CLI
ENV PATH="/root/.local/bin:$PATH"

# Builder: Generate Prisma at build time
RUN prisma generate

# Runtime: Copy BOTH directories
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache /root/.cache  # ← Must include this!

# Runtime: Direct Node.js installation
RUN curl -fsSL https://nodejs.org/dist/v20.11.0/...

# Runtime: Simple CMD for fast startup
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Background Task Files
**routes/interactive.py** - Background task orchestration
- Line 79-82: 2-second delay before aggregation
- Line 131-173: Structured error logging

**task_aggregator.py** - Task record creation
- Line 61-89: Retry logic with exponential backoff

---

## Monitoring Checklist

After every deployment, verify these appear in logs:

```
✅ Schema migrations completed successfully
✅ Database connection completed in lifespan
✅ Database connection verified - client reports connected
✅ Database connected successfully on attempt 1
✅ APPLICATION READY: All startup tasks completed
```

**Red flags** (investigate immediately):
```
❌ BinaryNotFoundError
❌ Database connection attempt failed
⚠️ Migration warning: NotConnectedError
⚠️ BACKGROUND_TASK_WARNING
❌ BACKGROUND_TASK_FAILURE
```

---

## Emergency Procedures

### Rollback to Last Known Good
```bash
# Immediate rollback to revision 170
gcloud run services update-traffic palet8-agents \
  --to-revisions=palet8-agents-00170-wdk=100 \
  --region=us-central1 \
  --project=palet8-system

# Verify
curl https://palet8-agents-702210710671.us-central1.run.app/health
```

### Check Service Status
```bash
# Cloud Console
https://console.cloud.google.com/run/detail/us-central1/palet8-agents?project=palet8-system

# CLI
gcloud run services describe palet8-agents --region=us-central1 --project=palet8-system
```

### View Recent Errors
```bash
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=palet8-agents AND severity>=ERROR AND timestamp>=2025-11-19T00:00:00Z" \
  --limit=50 \
  --project=palet8-system
```

---

## Development Workflow

### 1. Make Changes
Edit code in `/services/agents-api/src/`

### 2. Test Locally (Optional)
```bash
# Build Docker image
docker build -t palet8-agents-test .

# Run container
docker run -p 8000:8000 -e PORT=8000 palet8-agents-test

# Test endpoints
curl http://localhost:8000/health
```

### 3. Deploy
```bash
gcloud run deploy palet8-agents --source . --region us-central1 --project palet8-system --allow-unauthenticated
```

### 4. Verify Deployment
```bash
# Get revision number
REVISION=$(gcloud run services describe palet8-agents --region=us-central1 --project=palet8-system --format="value(status.latestReadyRevisionName)")
echo "New revision: $REVISION"

# Check timestamp
gcloud run revisions describe $REVISION --region=us-central1 --project=palet8-system --format="value(metadata.creationTimestamp)"

# Test health
curl https://palet8-agents-702210710671.us-central1.run.app/health
```

### 5. Monitor
```bash
# Watch logs for 5 minutes
gcloud run services logs read palet8-agents --region=us-central1 --project=palet8-system --limit=100 --follow
```

### 6. Update Documentation
- Add entry to [CHANGELOG.md](./CHANGELOG.md)
- Update this README if architecture changed
- Document any new mistakes in [COMMON_MISTAKES.md](./COMMON_MISTAKES.md)

---

## Support Resources

- **Cloud Console:** https://console.cloud.google.com/run?project=palet8-system
- **Cloud Logging:** https://console.cloud.google.com/logs/query?project=palet8-system
- **Cloud Build:** https://console.cloud.google.com/cloud-build/builds?project=palet8-system
- **Cloud SQL:** https://console.cloud.google.com/sql/instances?project=palet8-system

---

## Contributing

When making changes:

1. Read [COMMON_MISTAKES.md](./COMMON_MISTAKES.md) to avoid known pitfalls
2. Follow [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) procedures
3. Test deployment in non-production first (if possible)
4. Verify revision number before testing
5. Monitor logs for at least 5 minutes after deployment
6. Document changes in [CHANGELOG.md](./CHANGELOG.md)

---

**Last Updated:** November 19, 2025
**Maintained By:** Development Team
**Service Owner:** palet8-system
