batch_size=2

rm -rf "embeddings"
mkdir "embeddings"

dataset="ACM"

python specter_main.py --data_path="specter_format/articles_${dataset}_train.jsonl" --output "embeddings/articles_${dataset}_train.jsonl" --batch_size=$batch_size
python specter_main.py --data_path="specter_format/articles_${dataset}_dev_${dataset}.jsonl" --output "embeddings/articles_${dataset}_dev_${dataset}.jsonl" --batch_size=$batch_size
python specter_main.py --data_path="specter_format/articles_${dataset}_test_${dataset}.jsonl" --output "embeddings/articles_${dataset}_test_${dataset}.jsonl" --batch_size=$batch_size

python specter_main.py --data_path="specter_format/sentences_${dataset}_train.jsonl" --output "embeddings/sentences_${dataset}_train.jsonl" --batch_size=$batch_size