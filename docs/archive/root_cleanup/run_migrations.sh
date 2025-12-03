#!/bin/bash
# Run Prisma migrations on production database
set -e

echo "Running Prisma database migrations..."

# Run pending migrations
python3 -c "
import asyncio
from prisma import Prisma

async def run_migrations():
    db = Prisma()
    await db.connect()
    print('✅ Connected to database')

    # Read and execute migration SQL
    with open('prisma/migrations/20251106_add_model_name/migration.sql', 'r') as f:
        migration_sql = f.read()

    print(f'Executing migration SQL...')
    await db.execute_raw(migration_sql)
    print('✅ Migration applied successfully')

    await db.disconnect()

asyncio.run(run_migrations())
"

echo "✅ Migrations complete"
