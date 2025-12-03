# Deployment Changelog

This document tracks all deployments to the palet8-agents Cloud Run service.

---

## [Revision 170] - 2025-11-19 01:57 UTC ✅ CURRENT PRODUCTION

**Status:** ✅ Stable - All systems operational

### Changes
- Fixed Prisma query engine binary missing issue
- Added `/root/.cache` copy from builder stage
- All functional updates from previous attempts now working

### Files Modified
- `Dockerfile:59-60` - Copy both `.local` and `.cache` from builder

### Verification
```
✅ Schema migrations completed successfully
✅ Database connection completed in lifespan
✅ Database connection verified - client reports connected
✅ Database connected successfully on attempt 1
✅ APPLICATION READY: All startup tasks completed
```

### Health Check
```json
{
  "status": "healthy",
  "services": {
    "gemini_api": true,
    "openai_api": true,
    "flux_api": true,
    "product_generator": true,
    "database": true
  }
}
```

---

## [Revision 169] - 2025-11-19 01:35 UTC ⚠️ PARTIAL

**Status:** ⚠️ Deployed but database not working

### Issues
- Container started successfully
- Database connection failed with `BinaryNotFoundError`
- Prisma query engine binaries missing
- Service appeared healthy but couldn't process requests

### Root Cause
Prisma client generated in builder but binaries in `/root/.cache` not copied to runtime stage

### Rollback
Auto-replaced by revision 170 within 20 minutes

---

## [Revision 168] - 2025-11-19 01:21 UTC ⚠️ PARTIAL

**Status:** ⚠️ Deployed but database not working (same issue as 169)

### Changes
- Added PATH environment variable to builder stage
- Fixed Prisma CLI not found error during build

### Issues
- Build succeeded
- Container started
- Database still not working (binaries missing)

### Lesson Learned
Successfully generating Prisma client doesn't guarantee binaries are available at runtime

---

## [Revision 167] - 2025-11-18 14:48 UTC ❌ FAILED

**Status:** ❌ Container failed to start

### Changes
- Removed `prisma generate` from runtime
- Changed CMD to use bash script

### Issues
- Container startup timeout
- Never listened on PORT=8000
- Health check failed

### Root Cause
Running expensive operations at container startup delays listening on port

### Rollback
Not deployed to production (deployment failed)

---

## [Revision 166] - 2025-11-18 14:37 UTC ❌ FAILED

**Status:** ❌ Container failed to start (runtime Prisma generation)

### Changes
- Moved `prisma generate` to runtime (deploy-with-migrations.sh)
- Added environment variable for Prisma checksum

### Issues
- Container took too long to start
- Prisma generation at runtime added 30+ seconds to startup
- Cloud Run killed container before it could listen on port

### Root Cause
Runtime operations should be fast; expensive operations belong in build stage

### Rollback
Not deployed to production (deployment failed)

---

## [Revision 165] - 2025-11-18 09:30 UTC ✅ BASELINE

**Status:** ✅ Working baseline (before character integration updates)

### Features
- All core functionality working
- Database connections stable
- No character integration yet
- Background task aggregation working (but could fail silently)

### Known Issues
- No retry logic for background tasks
- Limited error visibility
- Vulnerable to race conditions

### Notes
This revision served as the stable baseline for troubleshooting new deployments

---

## [Revision 164] - 2025-11-18 08:59 UTC ❌ FAILED

**Status:** ❌ Runtime database schema errors

### Changes
- Added `characterId` field to Task model in Prisma schema

### Issues
- Deployed without running database migrations
- Schema mismatch between Prisma client and actual database
- Runtime errors when querying Task records

### Root Cause
Database schema changes require migrations, can't just update Prisma schema

### Rollback
Immediately rolled back to revision 163

---

## Summary of Issues Resolved

### Problem 1: External Package Repository Failures ✅ FIXED (Rev 170)
**Affected Revisions:** 166, 167, multiple earlier attempts
**Solution:** Direct binary downloads from nodejs.org instead of NodeSource repository

### Problem 2: Prisma Query Engine Missing ✅ FIXED (Rev 170)
**Affected Revisions:** 168, 169
**Solution:** Copy `/root/.cache` directory from builder stage

### Problem 3: Container Startup Timeout ✅ FIXED (Rev 170)
**Affected Revisions:** 166, 167
**Solution:** Run `prisma generate` at build time, not runtime

### Problem 4: Background Task Race Conditions ✅ FIXED (Rev 170)
**Affected Revisions:** 165 and earlier
**Solution:** 2-second delay + retry logic with exponential backoff

### Problem 5: Silent Background Failures ✅ FIXED (Rev 170)
**Affected Revisions:** 165 and earlier
**Solution:** Structured error logging with searchable patterns

---

## Deployment Statistics

| Revision | Status | Uptime | Issue |
|----------|--------|--------|-------|
| 170 | ✅ Current | Active | None |
| 169 | ⚠️ Partial | 22 min | Database not working |
| 168 | ⚠️ Partial | 14 min | Database not working |
| 167 | ❌ Failed | 0 min | Container timeout |
| 166 | ❌ Failed | 0 min | Container timeout |
| 165 | ✅ Stable | 17 hours | Baseline |
| 164 | ❌ Failed | 30 min | Schema mismatch |

---

## Lessons Learned

### 1. Multi-Stage Docker Builds Require Careful Planning
- Document what each stage produces
- Verify all necessary files are copied between stages
- Test that generated artifacts (like Prisma binaries) are accessible

### 2. External Dependencies Are Risky
- Package repositories can have outages
- Direct downloads are more reliable
- Cache dependencies when possible

### 3. Build-Time vs Runtime Operations
- Expensive operations belong in build stage
- Containers must start listening quickly (< 10 seconds)
- Health checks have strict timeouts

### 4. Silent Failures Are Dangerous
- Always use structured logging
- Make errors searchable
- Add monitoring and alerting

### 5. Background Tasks Need Resilience
- Database transactions aren't instant
- Add delays and retry logic
- Monitor for failures

---

## Known Good Configurations

### Dockerfile Key Sections (Rev 170)

**Builder Stage:**
```dockerfile
FROM python:3.11 as builder
ENV PATH="/root/.local/bin:$PATH"
RUN pip install --no-cache-dir --user -r requirements.txt
RUN prisma generate
```

**Runtime Stage:**
```dockerfile
FROM python:3.11
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache /root/.cache
RUN curl -fsSL https://nodejs.org/dist/v20.11.0/node-v20.11.0-linux-x64.tar.xz...
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

**Background Task Fixes:**
```python
# routes/interactive.py
await asyncio.sleep(2)  # Wait for DB commits

# task_aggregator.py
for attempt in range(3):
    agent_logs = await prisma.agentlog.find_many(...)
    if agent_logs:
        break
    await asyncio.sleep(retry_delays[attempt])
```

---

**Last Updated:** November 19, 2025
**Current Production Revision:** palet8-agents-00170-wdk
**Service URL:** https://palet8-agents-702210710671.us-central1.run.app
