# Common Deployment Mistakes - Quick Reference

This document provides a quick checklist of the most frequent mistakes when deploying the palet8-agents service.

---

## Critical Mistakes (Will Break Deployment)

### ❌ #1: Using External Package Repositories
**Error:** `curl: (22) The requested URL returned error: 500`

**Cause:** Depending on third-party apt repositories (NodeSource, etc.)

**Fix:**
```dockerfile
# Use direct binary downloads instead
RUN curl -fsSL https://nodejs.org/dist/v20.11.0/node-v20.11.0-linux-x64.tar.xz -o /tmp/node.tar.xz \
    && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1
```

---

### ❌ #2: Missing Prisma Query Engine
**Error:** `BinaryNotFoundError: Expected /app/prisma-query-engine`

**Cause:** Only copying `/root/.local` but Prisma binaries are in `/root/.cache`

**Fix:**
```dockerfile
# Copy BOTH directories from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache /root/.cache
```

---

### ❌ #3: Runtime Prisma Generation
**Error:** `Container failed to start and listen on port`

**Cause:** Running `prisma generate` at startup instead of build time

**Fix:**
```dockerfile
# In builder stage:
RUN prisma generate

# In CMD (keep simple):
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

---

### ❌ #4: PATH Not Set in Builder
**Error:** `/bin/sh: 1: prisma: not found`

**Cause:** Prisma CLI installed to `/root/.local/bin` but PATH not set

**Fix:**
```dockerfile
FROM python:3.11 as builder
ENV PATH="/root/.local/bin:$PATH"  # ← Add this
```

---

## Verification Mistakes (False Positives)

### ⚠️ #7: Testing Old Revision Instead of New Deployment
**Symptom:** Health check passes but changes not visible

**Cause:** Deployment output showed old revision number, tested service before verifying which revision is actually serving traffic

**How it Happens:**
```bash
# Deploy completes with output:
Service [palet8-agents] revision [palet8-agents-00165-628] has been deployed...

# You test immediately:
curl https://palet8-agents-702210710671.us-central1.run.app/health
# ✅ Passes! (but you're testing the OLD revision)
```

**Fix:**
```bash
# ALWAYS verify revision number first
gcloud run services describe palet8-agents \
  --region=us-central1 \
  --project=palet8-system \
  --format="value(status.latestReadyRevisionName, status.traffic)"

# Check if it's a NEW revision
gcloud run revisions list --service=palet8-agents --region=us-central1 --project=palet8-system --limit=5

# If old revision is serving, manually route traffic
gcloud run services update-traffic palet8-agents \
  --to-revisions=palet8-agents-00XXX-xxx=100 \
  --region=us-central1 \
  --project=palet8-system
```

**Prevention:**
- Never trust revision number in deployment output alone
- Always run `gcloud run services describe` to verify
- Check revision creation timestamp matches deployment time
- Only test health endpoint AFTER confirming correct revision is serving

---

## Non-Critical Mistakes (Service Runs but Data Loss)

### ⚠️ #5: Background Task Race Conditions
**Symptom:** No Task records in database, but workflows complete

**Cause:** Querying database before transactions commit

**Fix:**
```python
async def aggregate_task_in_background(...):
    # Add delay
    await asyncio.sleep(2)

    # Add retry logic
    for attempt in range(3):
        agent_logs = await prisma.agentlog.find_many(...)
        if agent_logs:
            break
        await asyncio.sleep(retry_delays[attempt])
```

---

### ⚠️ #6: Silent Background Failures
**Symptom:** Errors not visible in logs

**Cause:** Catching exceptions without structured logging

**Fix:**
```python
# Use searchable patterns
print(f"✅ BACKGROUND_TASK_SUCCESS task_id={task_id}")
print(f"⚠️ BACKGROUND_TASK_WARNING task_id={task_id} reason=...")
print(f"❌ BACKGROUND_TASK_FAILURE task_id={task_id} error=...")
```

---

## Pre-Deployment Checklist

Before running `gcloud run deploy`, verify:

- [ ] Dockerfile has `ENV PATH="/root/.local/bin:$PATH"` in builder stage
- [ ] Dockerfile has `RUN prisma generate` in builder stage
- [ ] Dockerfile copies both `/root/.local` AND `/root/.cache`
- [ ] CMD uses direct `uvicorn` (not a shell script)
- [ ] Node.js installed via direct download (not apt repository)
- [ ] Background tasks have 2-second delay + retry logic
- [ ] Error logging uses structured patterns (✅, ⚠️, ❌)

## Post-Deployment Checklist

After deployment completes:

- [ ] **Verify revision number** is new (not cached/reused)
- [ ] **Check traffic routing** to ensure 100% to new revision
- [ ] **Confirm timestamp** matches deployment time
- [ ] **Test health endpoint** only after above checks pass
- [ ] **Monitor startup logs** for ✅ success indicators
- [ ] **Watch for errors** in first 5 minutes

---

## Post-Deployment Verification

After deployment succeeds, check logs for:

```bash
# These should all appear:
✅ Schema migrations completed successfully
✅ Database connection completed in lifespan
✅ Database connection verified - client reports connected
✅ Database connected successfully on attempt 1
✅ APPLICATION READY: All startup tasks completed

# These should NOT appear:
❌ BinaryNotFoundError
❌ Database connection attempt failed
⚠️ Migration warning: NotConnectedError
```

---

## Emergency Rollback

If deployment breaks production:

```bash
# 1. Find last known good revision
gcloud run revisions list --service=palet8-agents --region=us-central1 --project=palet8-system

# 2. Rollback immediately
gcloud run services update-traffic palet8-agents \
  --to-revisions=palet8-agents-00XXX-xxx=100 \
  --region=us-central1 \
  --project=palet8-system

# 3. Verify
curl https://palet8-agents-702210710671.us-central1.run.app/health
```

**Known Good Revisions:**
- `palet8-agents-00170-wdk` - All fixes applied (Nov 19, 2025)
- `palet8-agents-00165-628` - Baseline before fixes (Nov 18, 2025)

---

## Troubleshooting Decision Tree

```
Deployment failed?
├─ Build failed?
│  ├─ "curl: (22) error: 500" → Mistake #1 (External repo)
│  ├─ "prisma: not found" → Mistake #4 (PATH not set)
│  └─ "Step X failed" → Check build logs
│
└─ Container startup failed?
   ├─ "failed to start and listen" → Mistake #3 (Runtime Prisma gen)
   ├─ "BinaryNotFoundError" → Mistake #2 (Missing .cache)
   └─ Check startup logs

Deployment succeeded but:
├─ Changes not visible?
│  └─ Verify revision number first → Mistake #7
│     ├─ Old revision serving? → Route traffic manually
│     └─ New revision but same behavior? → Check code changes
│
├─ Database not working?
│  └─ Check for "BinaryNotFoundError" → Mistake #2
│
└─ No Task records created?
   └─ Check logs for "BACKGROUND_TASK" → Mistakes #5, #6

**Always start with:** Verify which revision is actually serving traffic!
```

---

## Quick Fixes Reference

| Error Message | Mistake # | Quick Fix |
|---------------|-----------|-----------|
| `curl: (22) error: 500` | #1 | Use direct Node.js download |
| `BinaryNotFoundError` | #2 | Add `COPY --from=builder /root/.cache` |
| `failed to start and listen` | #3 | Move `prisma generate` to builder |
| `prisma: not found` | #4 | Add `ENV PATH="/root/.local/bin:$PATH"` |
| `No agent logs found` | #5 | Add 2s delay + retry logic |
| No errors but no Task records | #6 | Add structured logging |
| Changes not visible after deploy | #7 | Verify revision number, check traffic routing |

---

**See:** [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed explanations.

**Last Updated:** November 19, 2025
