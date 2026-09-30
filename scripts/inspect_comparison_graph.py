"""Read-only graph inventory for selecting grounded comparison questions."""

import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.knowledge.config import load_settings
from services.knowledge.neo4j_store import Neo4jKnowledgeStore


async def main():
    settings = load_settings()
    graph = Neo4jKnowledgeStore(uri=settings.neo4j_uri, user=settings.neo4j_user,
                               password=settings.neo4j_password, database=settings.neo4j_database)
    queries = {
        "relation_types": """
            MATCH (source:Entity)-[relation:RELATED]->(target:Entity)
            WHERE source.knowledge_base_id = 'default'
            RETURN relation.relation_type AS relation_type, count(*) AS count
            ORDER BY count DESC
        """,
        "related_entities": """
            MATCH (entity:Entity)
            WHERE entity.knowledge_base_id = 'default'
              AND entity.name IN ['光学', '生理学', '认知心理学', '环境心理学', '建筑物理',
                                  '生物物理学', '视觉生理学', '神经生理学', '心理学', '建筑光学',
                                  '环境工程学', '照明工程', '医学统计学', '教育心理学', '发展心理学']
            OPTIONAL MATCH (entity)-[relation:RELATED]-(neighbor:Entity)
            RETURN entity.name AS name, entity.entity_type AS entity_type, count(relation) AS degree,
                   collect(DISTINCT neighbor.name)[0..12] AS neighbors
            ORDER BY degree DESC, name LIMIT 100
        """,
        "relevant_relations": """
            MATCH (source:Entity)-[relation:RELATED]->(target:Entity)
            WHERE source.knowledge_base_id = 'default'
              AND (source.name IN ['生理学', '认知心理学', '环境心理学', '建筑物理', '生物物理学',
                                   '视觉生理学', '神经生理学', '建筑光学', '医学统计学', '教育心理学',
                                   '发展心理学', '环境工程学']
                   OR target.name IN ['视觉生理学', '神经生理学', '环境心理学', '建筑光学', '医学统计学'])
            RETURN DISTINCT source.name AS source, relation.relation_type AS relation_type,
                   target.name AS target
            ORDER BY source, relation_type, target LIMIT 220
        """,
    }
    try:
        for name, query in queries.items():
            records, _, _ = await graph.driver.execute_query(query, database_=settings.neo4j_database, routing_="r")
            print(json.dumps({name: [record.data() for record in records]}, ensure_ascii=True), flush=True)
    finally:
        await graph.close()


if __name__ == "__main__":
    asyncio.run(main())
