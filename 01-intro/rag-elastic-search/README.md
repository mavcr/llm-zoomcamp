Same implementation as rag-text-search but using elastic search to run.

```
docker run -it \
    --rm \
    --name elasticsearch \
    -m 4GB \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:9.0.1
```

Make sure version installed through pip matches Elastic Search container.