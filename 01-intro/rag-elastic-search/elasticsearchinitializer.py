from tqdm import tqdm

def initialize_client(client, documents, index_name):
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"}
            }
        }
    }

    if not client.indices.exists(index = index_name):
        client.indices.create(index=index_name, body=index_settings)

        for doc in tqdm(documents):
            client.index(index=index_name, document=doc)