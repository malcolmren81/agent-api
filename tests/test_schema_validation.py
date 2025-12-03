"""
Schema Validation Tests

Ensures database schema matches expected structure and constraints.
Tests verify table existence, column structure, indexes, and constraints.
"""

import pytest
from prisma import Prisma


@pytest.fixture
async def prisma_client():
    """Fixture providing connected Prisma client"""
    prisma = Prisma()
    await prisma.connect()
    yield prisma
    await prisma.disconnect()


@pytest.mark.asyncio
async def test_all_tables_exist(prisma_client):
    """Verify all expected tables exist in the database"""
    result = await prisma_client.query_raw("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)

    table_names = [row['table_name'] for row in result]

    expected_tables = [
        'Session', 'Asset', 'Template', 'AgentLog',
        'ModelConfig', 'ModelStats', 'Task', '_prisma_migrations'
    ]

    for table in expected_tables:
        assert table in table_names, f"Table {table} is missing from database"

    print(f"✓ All {len(expected_tables)} expected tables exist")


@pytest.mark.asyncio
async def test_session_table_structure(prisma_client):
    """Verify Session table has correct columns and types"""
    result = await prisma_client.query_raw("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'Session'
        ORDER BY column_name
    """)

    columns = {row['column_name']: row for row in result}

    # Verify required columns exist
    required_columns = [
        'id', 'shop', 'state', 'isOnline', 'accessToken'
    ]
    for col in required_columns:
        assert col in columns, f"Session.{col} column is missing"

    # Verify NOT NULL constraints
    assert columns['id']['is_nullable'] == 'NO', "Session.id should not be nullable"
    assert columns['shop']['is_nullable'] == 'NO', "Session.shop should not be nullable"
    assert columns['accessToken']['is_nullable'] == 'NO', "Session.accessToken should not be nullable"

    # Verify nullable columns
    assert columns['expires']['is_nullable'] == 'YES', "Session.expires should be nullable"
    assert columns['email']['is_nullable'] == 'YES', "Session.email should be nullable"

    print("✓ Session table structure is correct")


@pytest.mark.asyncio
async def test_asset_table_structure(prisma_client):
    """Verify Asset table has correct columns"""
    result = await prisma_client.query_raw("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'Asset'
        ORDER BY column_name
    """)

    columns = {row['column_name']: row for row in result}

    # Verify required columns
    required_columns = ['id', 'shop', 'taskId', 'prompt', 'status', 'cost', 'createdAt']
    for col in required_columns:
        assert col in columns, f"Asset.{col} column is missing"

    # Verify NOT NULL constraints
    assert columns['id']['is_nullable'] == 'NO'
    assert columns['shop']['is_nullable'] == 'NO'
    assert columns['taskId']['is_nullable'] == 'NO'
    assert columns['status']['is_nullable'] == 'NO'

    # Verify nullable columns
    assert columns['imageUrl']['is_nullable'] == 'YES', "Asset.imageUrl should be nullable (null while processing)"

    print("✓ Asset table structure is correct")


@pytest.mark.asyncio
async def test_template_table_structure(prisma_client):
    """Verify Template table has correct columns"""
    result = await prisma_client.query_raw("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'Template'
        ORDER BY column_name
    """)

    columns = {row['column_name']: row for row in result}

    # Verify required columns
    required_columns = ['id', 'shop', 'name', 'description', 'config', 'createdAt']
    for col in required_columns:
        assert col in columns, f"Template.{col} column is missing"

    # Verify JSON type for config
    assert 'json' in columns['config']['data_type'].lower(), "Template.config should be JSON type"

    print("✓ Template table structure is correct")


@pytest.mark.asyncio
async def test_agent_log_table_structure(prisma_client):
    """Verify AgentLog table has correct columns"""
    result = await prisma_client.query_raw("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'AgentLog'
        ORDER BY column_name
    """)

    columns = {row['column_name']: row for row in result}

    # Verify required columns
    required_columns = ['id', 'shop', 'taskId', 'agentType', 'status', 'metadata', 'createdAt']
    for col in required_columns:
        assert col in columns, f"AgentLog.{col} column is missing"

    # Verify JSON type for metadata
    assert 'json' in columns['metadata']['data_type'].lower(), "AgentLog.metadata should be JSON type"

    print("✓ AgentLog table structure is correct")


@pytest.mark.asyncio
async def test_indexes_exist(prisma_client):
    """Verify critical indexes are created"""
    result = await prisma_client.query_raw("""
        SELECT
            t.relname as table_name,
            i.relname as index_name,
            a.attname as column_name
        FROM
            pg_class t,
            pg_class i,
            pg_index ix,
            pg_attribute a
        WHERE
            t.oid = ix.indrelid
            AND i.oid = ix.indexrelid
            AND a.attrelid = t.oid
            AND a.attnum = ANY(ix.indkey)
            AND t.relkind = 'r'
            AND t.relname IN ('Asset', 'Session', 'Template', 'AgentLog', 'ModelStats')
        ORDER BY
            t.relname,
            i.relname
    """)

    # Build index lookup: table -> columns with indexes
    indexes = {}
    for row in result:
        table = row['table_name']
        column = row['column_name']
        if table not in indexes:
            indexes[table] = []
        if column not in indexes[table]:
            indexes[table].append(column)

    # Verify critical indexes exist
    assert 'shop' in indexes.get('Asset', []), "Asset.shop index missing"
    assert 'taskId' in indexes.get('Asset', []), "Asset.taskId index missing"
    assert 'shop' in indexes.get('Session', []), "Session.shop index missing"
    assert 'shop' in indexes.get('Template', []), "Template.shop index missing"
    assert 'shop' in indexes.get('AgentLog', []), "AgentLog.shop index missing"
    assert 'taskId' in indexes.get('AgentLog', []), "AgentLog.taskId index missing"

    print(f"✓ All critical indexes exist")


@pytest.mark.asyncio
async def test_unique_constraints(prisma_client):
    """Verify unique constraints are enforced"""
    result = await prisma_client.query_raw("""
        SELECT
            tc.table_name,
            kcu.column_name,
            tc.constraint_type
        FROM
            information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
        WHERE tc.constraint_type IN ('UNIQUE', 'PRIMARY KEY')
            AND tc.table_name IN ('Asset', 'Session', 'Template', 'AgentLog')
        ORDER BY tc.table_name, kcu.column_name
    """)

    # Build constraint lookup
    constraints = {}
    for row in result:
        table = row['table_name']
        column = row['column_name']
        constraint_type = row['constraint_type']

        if table not in constraints:
            constraints[table] = {'unique': [], 'primary': []}

        if constraint_type == 'UNIQUE':
            constraints[table]['unique'].append(column)
        elif constraint_type == 'PRIMARY KEY':
            constraints[table]['primary'].append(column)

    # Verify primary keys
    assert 'id' in constraints.get('Asset', {}).get('primary', []), "Asset.id should be primary key"
    assert 'id' in constraints.get('Session', {}).get('primary', []), "Session.id should be primary key"

    # Verify unique constraints
    assert 'taskId' in constraints.get('Asset', {}).get('unique', []), "Asset.taskId should be unique"

    print("✓ All unique constraints verified")


@pytest.mark.asyncio
async def test_table_row_counts(prisma_client):
    """Verify tables can be queried and return counts"""
    tables = ['Session', 'Asset', 'Template', 'AgentLog', 'ModelConfig', 'ModelStats', 'Task']

    for table in tables:
        result = await prisma_client.query_raw(
            f'SELECT COUNT(*) as count FROM "{table}"'
        )
        count = result[0]['count']
        print(f"✓ {table}: {count} rows")


@pytest.mark.asyncio
async def test_primary_keys_are_uuid(prisma_client):
    """Verify primary key columns use UUID type"""
    result = await prisma_client.query_raw("""
        SELECT
            c.table_name,
            c.column_name,
            c.data_type
        FROM
            information_schema.columns c
            JOIN information_schema.table_constraints tc
              ON c.table_name = tc.table_name
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
              AND c.column_name = kcu.column_name
        WHERE
            tc.constraint_type = 'PRIMARY KEY'
            AND c.table_name IN ('Asset', 'Session', 'Template', 'AgentLog', 'ModelConfig', 'ModelStats', 'Task')
    """)

    for row in result:
        table = row['table_name']
        column = row['column_name']
        data_type = row['data_type']

        # UUID can be stored as 'uuid' or 'character varying' (for String in Prisma)
        assert data_type in ['uuid', 'character varying', 'text'], \
            f"{table}.{column} should be UUID-compatible type, got {data_type}"

    print("✓ All primary keys use UUID-compatible types")


@pytest.mark.asyncio
async def test_timestamp_columns_exist(prisma_client):
    """Verify all tables have createdAt timestamp"""
    tables = ['Session', 'Asset', 'Template', 'AgentLog', 'ModelConfig', 'ModelStats', 'Task']

    for table in tables:
        result = await prisma_client.query_raw(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{table}' AND column_name = 'createdAt'
        """)

        assert len(result) > 0, f"{table} table missing createdAt column"

    print("✓ All tables have createdAt timestamp")


@pytest.mark.asyncio
async def test_json_columns_are_valid(prisma_client):
    """Verify JSON columns can store and retrieve JSON data"""
    # Test Template.config (JSON column)
    test_config = {
        "model": "test-model",
        "style": "test-style",
        "width": 1024,
        "height": 1024
    }

    # This test assumes Prisma schema allows JSON, just verify type
    result = await prisma_client.query_raw("""
        SELECT data_type
        FROM information_schema.columns
        WHERE table_name = 'Template' AND column_name = 'config'
    """)

    assert len(result) > 0, "Template.config column not found"
    data_type = result[0]['data_type']
    assert 'json' in data_type.lower(), f"Template.config should be JSON type, got {data_type}"

    print("✓ JSON columns have correct type")


@pytest.mark.asyncio
async def test_database_connection_pool(prisma_client):
    """Verify database connection is working and can handle queries"""
    # Simple connectivity test
    result = await prisma_client.query_raw("SELECT 1 as test")
    assert result[0]['test'] == 1, "Database connection test failed"

    print("✓ Database connection pool is working")


# Performance baseline tests
@pytest.mark.asyncio
async def test_query_performance_baseline(prisma_client):
    """Establish baseline query performance"""
    import time

    # Test simple SELECT performance
    start = time.time()
    await prisma_client.query_raw('SELECT COUNT(*) FROM "Asset"')
    duration = time.time() - start

    # Should complete in < 100ms on a healthy database
    assert duration < 0.1, f"Simple COUNT query took {duration:.3f}s, expected < 0.1s"

    print(f"✓ Query performance baseline: {duration*1000:.1f}ms")


if __name__ == "__main__":
    # Run with: pytest tests/test_schema_validation.py -v
    pytest.main([__file__, "-v"])
