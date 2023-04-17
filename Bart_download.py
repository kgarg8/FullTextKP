from os import fspath
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
  
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

save_embedding_path = fspath(Path("embeddings/bart-base/"))
Path(save_embedding_path).mkdir(parents=True, exist_ok=True)

new_tokens = ["<sep>", "<digit>", "<cls>", "<eos>"]

print(tokenizer.encode("<sep>", add_special_tokens=False))

special_tokens_dict = {'additional_special_tokens': new_tokens}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
print(tokenizer.encode("<sep>", add_special_tokens=False))

model.save_pretrained(save_embedding_path)  # save
tokenizer.save_pretrained(save_embedding_path)  # save