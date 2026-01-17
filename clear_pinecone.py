"""Clear all Pinecone namespaces."""
import os
import json

# Load settings
settings_path = os.path.join(os.path.dirname(__file__), "local.settings.json")
if os.path.exists(settings_path):
    with open(settings_path, 'r') as f:
        for k, v in json.load(f).get("Values", {}).items():
            os.environ[k] = str(v)

from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("donna-email")

stats = index.describe_index_stats()
namespaces = list(stats.get("namespaces", {}).keys())

print(f"Found {len(namespaces)} namespaces: {namespaces}")

for ns in namespaces:
    print(f"Deleting {ns}...")
    index.delete(delete_all=True, namespace=ns)

print("Done!")
