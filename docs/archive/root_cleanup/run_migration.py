#!/usr/bin/env python3
"""
Temporary script to run database migration for modelName column.
"""
import asyncio
import os
from google.cloud.sql.connector import Connector
import asyncpg


async def run_migration():
    """Run the modelName migration."""
    print("=" * 80)
    print("Starting database migration for modelName column...")
    print("=" * 80)

    # Cloud SQL connection details
    instance_connection_name = "palet8-system:us-central1:palet8-db"
    db_user = "palet8_user"
    db_name = "palet8_sessions"

    print(f"Connecting to: {instance_connection_name}")
    print(f"Database: {db_name}")
    print(f"User: {db_user}")

    connector = Connector()

    try:
        # Create connection using IAM authentication
        print("\nAttempting connection with IAM authentication...")
        conn: asyncpg.Connection = await connector.connect_async(
            instance_connection_name,
            "asyncpg",
            user=db_user,
            db=db_name,
            enable_iam_auth=True,
        )

        print("✅ Connected successfully!")

        # Run the migration
        print("\nExecuting migration SQL...")
        print('ALTER TABLE "AgentLog" ADD COLUMN IF NOT EXISTS "modelName" TEXT;')

        await conn.execute('ALTER TABLE "AgentLog" ADD COLUMN IF NOT EXISTS "modelName" TEXT;')

        print("✅ Migration executed successfully!")

        # Verify the column exists
        print("\nVerifying column exists...")
        result = await conn.fetch("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'AgentLog' AND column_name = 'modelName';
        """)

        if result:
            print(f"✅ Column verified: {result[0]['column_name']} ({result[0]['data_type']})")
        else:
            print("⚠️  Column not found in verification query")

        await conn.close()

    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        raise
    finally:
        await connector.close_async()

    print("\n" + "=" * 80)
    print("Migration complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_migration())
