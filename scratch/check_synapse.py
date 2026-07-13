from app.db.neo4j_driver import neo4j_db

res = neo4j_db.query("""
MATCH (s)-[r]->(o)
WHERE toLower(s.name) = 'synapse' OR toLower(o.name) = 'synapse'
RETURN s.name as subject, type(r) as relation, o.name as object
""")
print("SYNAPSE Relations:")
for r in res:
    print(f"  {r['subject']} --[{r['relation']}]--> {r['object']}")
