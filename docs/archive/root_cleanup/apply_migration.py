#!/usr/bin/env python3
"""
Quick script to apply the modelName migration to production database.
"""
import os
import psycopg2
from urllib.parse import urlparse, unquote

# Get DATABASE_URL from environment or use the one from secrets
database_url = "postgresql://palet8_user:lcCQVLKZEhosKxavqnNirI%2FZYctc2vhTaq%2BJBaEZsKTaLCZttbnioh13FSjkm%2Bdu@localhost/palet8_sessions?host=/cloudsql/palet8-system:us-central1:palet8-db"

# Parse URL
parsed = urlparse(database_url)
username = parsed.username
password = unquote(parsed.password) if parsed.password else None
database = parsed.path.lstrip('/')
host = parsed.hostname

# Extract host from query params if present
if '?host=' in database_url:
    unix_socket = database_url.split('?host=')[1].split('&')[0]
    print(f"Using Unix socket: {unix_socket}")

    # For Cloud SQL, we need to use the socket
    conn = psycopg2.connect(
        database=database,
        user=username,
        password=password,
        host=unix_socket
    )
else:
    conn = psycopg2.connect(
        database=database,
        user=username,
        password=password,
        host=host,
        port=parsed.port or 5432
    )

print(f"Connected to database: {database}")

# Read migration SQL
with open('migrations/add_model_name.sql', 'r') as f:
    migration_sql = f.read()

print(f"Executing migration:\n{migration_sql}")

# Execute migration
cursor = conn.cursor()
cursor.execute(migration_sql)
conn.commit()

print("Migration applied successfully!")

# Verify column exists
cursor.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name='AgentLog' AND column_name='modelName'
""")

result = cursor.fetchone()
if result:
    print(f"✅ Verified: Column 'modelName' exists with type '{result[1]}'")
else:
    print("❌ ERROR: Column 'modelName' not found after migration!")

cursor.close()
conn.close()
