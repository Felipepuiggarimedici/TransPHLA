import json
import numpy as np
def json_to_npy(json_path, npy_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"File content length: {len(content)}")
        print(f"File content preview: {content[:200]}")
        data = json.loads(content)
    np.save(npy_path, data, allow_pickle=True)
    print(f"Saved vocab dict as npy to {npy_path}")

if __name__ == "__main__":
    json_file = "tokenizer/vocab.json"
    npy_file = "tokenizer/Transformer_vocab_dict.npy"
    json_to_npy(json_file, npy_file)
