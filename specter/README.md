## Specter (for ACM dataset)

```
# Preprocess for specter format
python preprocess_ACM.py

# Script to create embeddings for all the sentences and articles
chmod +x embed.sh
./embed.sh

# Calculate similarity and store the newly processed data
python similarity_ACM.py
```