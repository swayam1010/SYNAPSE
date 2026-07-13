import os
import shutil
from app.db.neo4j_driver import neo4j_db
from app.core.config import settings

def clear_all():
    print("🧠 Initiating Lobotomy (Clearing all Synapse Memories)...")
    
    # 1. Clear Neo4j
    if neo4j_db.driver:
        print("Wiping Knowledge Graph (Neo4j)...")
        neo4j_db.query("MATCH (n) DETACH DELETE n;")
        print("Neo4j wiped.")
    
    # 2. Clear SQLite (Working Memory)
    if os.path.exists("synapse_sessions.db"):
        os.remove("synapse_sessions.db")
        print("synapse_sessions.db wiped.")
        
    # 3. Clear ChromaDB
    chroma_path = settings.CHROMA_DB_PATH
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        print(f"ChromaDB directory ({chroma_path}) wiped.")

    print("✅ All memories cleared. Synapse is a blank slate again.")

if __name__ == "__main__":
    clear_all()
