#!/bin/bash
# Deployment script for palet8-agents with migration handling

echo "🚀 Starting deployment with migrations..."

# Step 0: Generate Prisma client (moved from Dockerfile to avoid build failures)
echo "🔧 Generating Prisma client..."
export PRISMA_ENGINES_CHECKSUM_IGNORE_MISSING=1
npx prisma generate

# Step 1: Mark baseline migration as applied (database already has schema from manual setup)
echo "📊 Marking baseline migration as applied..."
npx prisma migrate resolve --applied "20251117000000_baseline" 2>&1 | grep -v "already" || true

# Step 2: Deploy any new migrations
echo "🔄 Deploying migrations..."
npx prisma migrate deploy

echo "✅ Migration process complete"

# Step 3: Start the application
echo "🎯 Starting application on port ${PORT:-8000}..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300
