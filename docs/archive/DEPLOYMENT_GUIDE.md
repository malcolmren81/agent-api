# Agents API Deployment Guide

**Service:** palet8-agents
**Platform:** Google Cloud Run
**Region:** us-central1
**Project:** palet8-system

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Procedure](#deployment-procedure)
3. [Common Mistakes & Solutions](#common-mistakes--solutions)
4. [Troubleshooting](#troubleshooting)
5. [Rollback Procedure](#rollback-procedure)
6. [Monitoring & Verification](#monitoring--verification)

---

## Prerequisites

### Required Tools
- `gcloud` CLI (authenticated as marc@palet8.com)
- Docker (for local testing, optional)
- Access to palet8-system GCP project

### Environment Variables (Managed as Secrets)
The following secrets are automatically injected by Cloud Run:
- `GEMINI_API_KEY` - Google Gemini API credentials
- `FLUX_API_KEY` - Flux image generation API
- `OPENAI_API_KEY` - OpenAI API credentials
- `PALET8_API_URL` - Internal API endpoint
- `PALET8_API_KEY` - Internal API authentication
- `DATABASE_URL` - PostgreSQL connection string
- `ADMIN_API_URL` - Admin API endpoint (https://palet8-admin-api-702210710671.us-central1.run.app)

### Cloud SQL Connection
- Instance: `palet8-system:us-central1:palet8-db`
- Automatically connected via Cloud Run annotation

---

## Deployment Procedure

### Step 1: Verify Local Changes

```bash
# Navigate to service directory
cd "/Users/malcolmyam/Desktop/alpha project/agent system/services/agents-api"

# Verify Dockerfile structure
cat Dockerfile | grep -E "COPY.*builder|RUN prisma|CMD"

# Expected output should show:
# - COPY --from=builder /root/.local /root/.local
# - COPY --from=builder /root/.cache /root/.cache
# - RUN prisma generate (in builder stage)
# - CMD uvicorn src.api.main:app...
```

### Step 2: Deploy to Cloud Run

```bash
gcloud run deploy palet8-agents \
  --source . \
  --region us-central1 \
  --project palet8-system \
  --allow-unauthenticated
```

**Expected Build Time:** 3-5 minutes
**Expected Deployment Output:**
```
Building Container...done
Setting IAM Policy...done
Creating Revision...done
Done.
Service [palet8-agents] revision [palet8-agents-00XXX-xxx] has been deployed and is serving 100 percent of traffic.
Service URL: https://palet8-agents-702210710671.us-central1.run.app
```

### Step 3: Verify Correct Revision is Deployed

**⚠️ CRITICAL:** The deployment output may show an old revision number if Cloud Run reused cached layers. Always verify which revision is actually serving traffic!

```bash
# Get the actual serving revision
gcloud run services describe palet8-agents \
  --region=us-central1 \
  --project=palet8-system \
  --format="value(status.latestReadyRevisionName, status.traffic)"

# Example output:
# palet8-agents-00170-wdk [{'percent': 100, 'revisionName': 'palet8-agents-00170-wdk'}]
```

**Verify:**
1. `latestReadyRevisionName` is a NEW revision number (incremented from last deployment)
2. `traffic` shows 100% going to the new revision
3. Revision timestamp matches your deployment time

```bash
# Check revision creation time
gcloud run revisions describe palet8-agents-00XXX-xxx \
  --region=us-central1 \
  --project=palet8-system \
  --format="value(metadata.creationTimestamp)"
```

**If revision number is OLD or traffic is not routed:**
```bash
# List recent revisions to find the new one
gcloud run revisions list \
  --service=palet8-agents \
  --region=us-central1 \
  --project=palet8-system \
  --limit=5

# Manually route traffic to new revision if needed
gcloud run services update-traffic palet8-agents \
  --to-revisions=palet8-agents-00XXX-xxx=100 \
  --region=us-central1 \
  --project=palet8-system
```

### Step 4: Verify Service Health

**Only test health AFTER confirming correct revision is serving!**

```bash
# Check health endpoint
curl https://palet8-agents-702210710671.us-central1.run.app/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2025-11-19T...",
  "version": "0.1.0",
  "services": {
    "gemini_api": true,
    "openai_api": true,
    "flux_api": true,
    "product_generator": true,
    "database": true
  }
}
```

**⚠️ Common Mistake:** Testing health endpoint before verifying revision number can give false positives (testing old working revision instead of new deployment).

### Step 4: Monitor Startup Logs

```bash
# Get latest revision name
REVISION=$(gcloud run services describe palet8-agents \
  --region=us-central1 \
  --project=palet8-system \
  --format="value(status.latestReadyRevisionName)")

# Check startup logs
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.revision_name=$REVISION" \
  --limit=50 \
  --project=palet8-system \
  --format="value(textPayload)" | grep "✅"

# Expected startup logs:
# ✅ Schema migrations completed successfully
# ✅ Database connection completed in lifespan
# ✅ Database connection verified - client reports connected
# ✅ DATABASE connected successfully on attempt 1
# ✅ APPLICATION READY: All startup tasks completed
```

---

## Common Mistakes & Solutions

### Mistake #1: External Package Repository Failures

**Symptom:**
```
ERROR: The requested URL returned error: 500
curl: (22) deb.nodesource.com failed
Build failed at Step 10/22
```

**Root Cause:**
Using external apt repositories (like NodeSource) that may have outages during build.

**Solution:**
✅ **Use direct binary downloads instead of apt repositories**

```dockerfile
# ❌ WRONG - Depends on external repository
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor...
    && apt-get install -y nodejs

# ✅ CORRECT - Direct download from official source
RUN curl -fsSL https://nodejs.org/dist/v20.11.0/node-v20.11.0-linux-x64.tar.xz -o /tmp/node.tar.xz \
    && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 \
    && rm /tmp/node.tar.xz
```

**Prevention:**
- Avoid dependencies on third-party package repositories
- Use official binaries or multi-stage builds with base images

---

### Mistake #2: Missing Prisma Query Engine Binary

**Symptom:**
```
BinaryNotFoundError: Expected /app/prisma-query-engine-debian-openssl-3.5.x
❌ Database connection attempt failed
Try running prisma py fetch
```

**Root Cause:**
Prisma generates client code in `/root/.local` but stores query engine binaries in `/root/.cache`. In multi-stage Docker builds, only copying `/root/.local` leaves binaries behind.

**Solution:**
✅ **Copy BOTH .local and .cache directories from builder stage**

```dockerfile
# ❌ WRONG - Only copies Python packages
COPY --from=builder /root/.local /root/.local

# ✅ CORRECT - Copies both packages AND Prisma binaries
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache /root/.cache
```

**Prevention:**
- Always verify Prisma client generation includes binaries
- Test database connections in startup logs
- Check for "BinaryNotFoundError" in logs immediately after deployment

---

### Mistake #3: Runtime Prisma Generation Causing Startup Timeout

**Symptom:**
```
ERROR: The user-provided container failed to start and listen on the port
Deployment failed
Logs show: Installing Prisma CLI... (taking too long)
```

**Root Cause:**
Running `prisma generate` at container startup (via entrypoint script) delays application from listening on the required port. Cloud Run kills containers that don't respond within timeout.

**Solution:**
✅ **Generate Prisma client at BUILD time, not RUNTIME**

```dockerfile
# ❌ WRONG - Runtime generation in CMD/entrypoint
CMD ["./deploy-with-migrations.sh"]  # Script runs: npx prisma generate

# ✅ CORRECT - Build-time generation in Dockerfile
# In builder stage:
RUN prisma generate

# In runtime stage:
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

**Prevention:**
- Keep CMD simple and fast (direct uvicorn startup)
- Run expensive operations (prisma generate, migrations) during build
- Test startup time: containers should listen on PORT within 10 seconds

---

### Mistake #4: Prisma CLI Not Found in Builder Stage

**Symptom:**
```
/bin/sh: 1: prisma: not found
The command '/bin/sh -c prisma generate' returned a non-zero code: 127
```

**Root Cause:**
The `prisma` CLI is installed via pip to `/root/.local/bin` but PATH environment variable isn't set in the builder stage.

**Solution:**
✅ **Set PATH in builder stage before running Prisma**

```dockerfile
# ❌ WRONG - PATH not set in builder
FROM python:3.11 as builder
WORKDIR /app
RUN pip install --user -r requirements.txt
RUN prisma generate  # ← Command not found!

# ✅ CORRECT - PATH set before using Prisma
FROM python:3.11 as builder
WORKDIR /app
ENV PATH="/root/.local/bin:$PATH"  # ← Set PATH first
RUN pip install --user -r requirements.txt
RUN prisma generate  # ← Now works!
```

**Prevention:**
- Always set PATH when using `pip install --user` in Docker
- Test that all CLI tools are accessible before running them

---

### Mistake #5: Background Task Race Conditions

**Symptom:**
```
⚠️ WARNING: No agent logs found for task_id=abc-123
⚠️ BACKGROUND TASK: aggregator returned None
No Task records created in database
```

**Root Cause:**
Background task (`asyncio.create_task()`) starts immediately after HTTP response is sent. If database transactions haven't committed yet, aggregator queries return no results.

**Solution:**
✅ **Add delay before querying + retry logic with exponential backoff**

```python
# ❌ WRONG - Immediate query after response sent
asyncio.create_task(aggregate_task_in_background(...))
# Inside aggregate_task_in_background:
agent_logs = await prisma.agentlog.find_many(...)  # ← May return empty!

# ✅ CORRECT - Wait for commits + retry logic
async def aggregate_task_in_background(...):
    # Wait for database commits to complete
    await asyncio.sleep(2)

    # Retry with exponential backoff
    max_retries = 3
    retry_delays = [0.5, 1.0, 2.0]

    for attempt in range(max_retries):
        agent_logs = await prisma.agentlog.find_many(...)
        if agent_logs:
            break
        elif attempt < max_retries - 1:
            await asyncio.sleep(retry_delays[attempt])
```

**Prevention:**
- Never assume database writes are instantly visible to other queries
- Always add retry logic for background tasks that depend on recent writes
- Monitor for "aggregator returned None" warnings in logs

---

### Mistake #6: Silent Background Task Failures

**Symptom:**
```
User gets images successfully
No errors in logs
BUT: No Task records in database
No visibility into what went wrong
```

**Root Cause:**
Background tasks use broad exception catching (`except Exception`) without re-raising or alerting, making failures invisible.

**Solution:**
✅ **Use structured logging with searchable patterns**

```python
# ❌ WRONG - Errors suppressed silently
try:
    result = await aggregator.create_task_from_logs(...)
except Exception as e:
    print(f"Error: {e}")  # ← Generic, unsearchable

# ✅ CORRECT - Structured, searchable logging
try:
    result = await aggregator.create_task_from_logs(...)
    if result:
        print(f"✅ BACKGROUND_TASK_SUCCESS task_id={task_id} shop={shop}")
        logger.info("Background task completed",
                    task_id=task_id,
                    metric_name="background_task_success")
    else:
        print(f"⚠️ BACKGROUND_TASK_WARNING task_id={task_id} reason=aggregator_returned_none")
        logger.warning("Task aggregation returned None",
                       task_id=task_id,
                       metric_name="background_task_no_result")
except Exception as e:
    print(f"❌ BACKGROUND_TASK_FAILURE task_id={task_id} error_type={type(e).__name__}")
    logger.error("Background task failed",
                 task_id=task_id,
                 error_type=type(e).__name__,
                 metric_name="background_task_failure",
                 exc_info=True)
```

**Prevention:**
- Always use structured logging with consistent patterns
- Include metric names for alerting
- Make errors searchable with emojis or prefixes (✅, ⚠️, ❌)

---

### Mistake #7: Dockerfile Changes Without Testing Build Stages

**Symptom:**
```
Step 17/23 successful
But runtime fails with missing files/binaries
```

**Root Cause:**
Multi-stage Docker builds require careful attention to what gets copied between stages. Changes to one stage may break another.

**Solution:**
✅ **Always verify what gets copied between stages**

```dockerfile
# Checklist for multi-stage builds:
# 1. What does the builder stage produce?
#    - /root/.local (Python packages)
#    - /root/.cache (Prisma binaries)
#    - Generated code

# 2. What does the runtime stage need?
#    - All of the above!

# 3. Are we copying everything necessary?
COPY --from=builder /root/.local /root/.local  # ✓
COPY --from=builder /root/.cache /root/.cache  # ✓
```

**Prevention:**
- Document what each stage produces
- Verify COPY commands include all dependencies
- Test locally with `docker build` before deploying to Cloud Run

---

## Troubleshooting

### Build Failures

**1. Check build logs:**
```bash
# Get latest build ID
BUILD_ID=$(gcloud builds list --project=palet8-system --region=us-central1 --limit=1 --format="value(id)")

# View logs
gcloud beta builds log $BUILD_ID --project=palet8-system --region=us-central1
```

**2. Look for common errors:**
- `curl: (22)` → External repository failure (see Mistake #1)
- `command not found` → PATH issue (see Mistake #4)
- `returned a non-zero code` → Command failed, check logs above

---

### Container Startup Failures

**1. Check revision logs:**
```bash
# Get failed revision name from error message
REVISION="palet8-agents-00XXX-xxx"

# View startup logs
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.revision_name=$REVISION" \
  --limit=100 \
  --project=palet8-system \
  --format="value(textPayload)"
```

**2. Look for critical errors:**
- `BinaryNotFoundError` → Missing Prisma binaries (see Mistake #2)
- `failed to start and listen` → Startup too slow (see Mistake #3)
- `Database connection failed` → Check DATABASE_URL secret

---

### Runtime Issues After Successful Deployment

**1. Check recent logs:**
```bash
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=palet8-agents AND timestamp>=\"$(date -u -d '10 minutes ago' '+%Y-%m-%dT%H:%M:%SZ')\"" \
  --limit=200 \
  --project=palet8-system \
  --format="value(timestamp, severity, textPayload)"
```

**2. Search for specific issues:**
```bash
# Background task failures
gcloud logging read ... | grep "BACKGROUND_TASK"

# Database issues
gcloud logging read ... | grep -E "Database|prisma|query-engine"

# API errors
gcloud logging read ... | grep -E "ERROR|Exception|Failed"
```

---

## Rollback Procedure

### If a deployment fails or causes issues:

**1. List recent revisions:**
```bash
gcloud run revisions list \
  --service=palet8-agents \
  --region=us-central1 \
  --project=palet8-system \
  --limit=10
```

**2. Identify last known good revision:**
```
NAME                     STATUS  CREATION_TIMESTAMP
palet8-agents-00170-wdk  True    2025-11-19T01:57:45Z  ← Current (might be broken)
palet8-agents-00165-628  True    2025-11-18T09:30:06Z  ← Known good
```

**3. Rollback traffic:**
```bash
gcloud run services update-traffic palet8-agents \
  --to-revisions=palet8-agents-00165-628=100 \
  --region=us-central1 \
  --project=palet8-system
```

**4. Verify rollback:**
```bash
curl https://palet8-agents-702210710671.us-central1.run.app/health
```

---

## Monitoring & Verification

### Health Check

**Endpoint:** `https://palet8-agents-702210710671.us-central1.run.app/health`

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-19T...",
  "version": "0.1.0",
  "services": {
    "gemini_api": true,
    "openai_api": true,
    "flux_api": true,
    "product_generator": true,
    "database": true
  }
}
```

**All services must show `true`** for a healthy deployment.

---

### Startup Checklist

After every deployment, verify these log messages appear:

```
✅ Schema migrations completed successfully
✅ Database connection completed in lifespan
✅ Database connection verified - client reports connected
✅ Database connected successfully on attempt 1
✅ APPLICATION READY: All startup tasks completed
```

**If any ❌ or ⚠️ appear**, investigate before considering deployment successful.

---

### Background Task Monitoring

**Watch for these patterns in logs:**

```bash
# Success pattern (good)
✅ BACKGROUND_TASK_SUCCESS task_id=... shop=... status=completed

# Warning pattern (investigate)
⚠️ BACKGROUND_TASK_WARNING task_id=... status=no_task_created reason=aggregator_returned_none

# Failure pattern (urgent)
❌ BACKGROUND_TASK_FAILURE task_id=... error_type=... error="..."
```

**Set up log-based alerts** for `BACKGROUND_TASK_FAILURE` patterns.

---

### Performance Metrics

**Monitor these Cloud Run metrics:**

1. **Container Startup Latency**
   - Should be < 10 seconds
   - If > 20 seconds, investigate (see Mistake #3)

2. **Request Latency**
   - `/health` should respond in < 100ms
   - `/run` endpoint may take minutes (image generation)

3. **Error Rate**
   - Should be < 1% under normal operation
   - Spike indicates deployment issues

4. **Instance Count**
   - Auto-scales 0-10 instances
   - Sudden drops may indicate crashes

---

## Best Practices

### 1. Always Test Locally First
```bash
# Build Docker image locally
docker build -t palet8-agents-test .

# Run container
docker run -p 8000:8000 -e PORT=8000 palet8-agents-test

# Test health endpoint
curl http://localhost:8000/health
```

### 2. Use Incremental Changes
- Make one logical change per deployment
- Easier to identify which change caused issues
- Faster rollback if needed

### 3. Monitor Immediately After Deployment
- Watch logs for first 5 minutes
- Check health endpoint every 30 seconds
- Look for any ⚠️ or ❌ patterns

### 4. Document Changes
- Update CHANGELOG.md with each deployment
- Note revision numbers and what changed
- Include rollback commands if needed

### 5. Keep Stable Baseline
- Tag known-good revisions in comments
- Document configuration of stable versions
- Reference in rollback procedures

---

## Quick Reference

### Useful Commands

```bash
# Deploy
gcloud run deploy palet8-agents --source . --region us-central1 --project palet8-system --allow-unauthenticated

# Check status
gcloud run services describe palet8-agents --region=us-central1 --project=palet8-system

# View logs
gcloud run services logs read palet8-agents --region=us-central1 --project=palet8-system --limit=100

# List revisions
gcloud run revisions list --service=palet8-agents --region=us-central1 --project=palet8-system

# Rollback
gcloud run services update-traffic palet8-agents --to-revisions=palet8-agents-00XXX-xxx=100 --region=us-central1 --project=palet8-system

# Health check
curl https://palet8-agents-702210710671.us-central1.run.app/health
```

---

## Support & Resources

- **Cloud Console:** https://console.cloud.google.com/run?project=palet8-system
- **Service Logs:** Cloud Console → Cloud Run → palet8-agents → Logs
- **Build History:** Cloud Console → Cloud Build → History
- **Database:** Cloud Console → SQL → palet8-db

---

**Last Updated:** November 19, 2025
**Current Revision:** palet8-agents-00170-wdk
**Status:** ✅ Production Ready
