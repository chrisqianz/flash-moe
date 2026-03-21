# README_35B

Setup and run `flash-moe` with:

- `mlx-community/Qwen3.5-35B-A3B-4bit`

This guide covers the working flow from generating the expert index through launching the chat client.

---

## Prerequisites

- macOS with Metal support
- Xcode command line tools
- Python 3 in a virtual environment
- Local Hugging Face model snapshot for:

  `mlx-community/Qwen3.5-35B-A3B-4bit`

---

## Model path

In the commands below, this model snapshot is used:

```bash
MODEL=/Users/sbaruwal/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
```

Update that variable if your snapshot path differs.

---

## 1. Create and activate the virtual environment

From the repo root:

```bash
cd /Users/sbaruwal/Repo/flash-moe-1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install tokenizers
pip install numpy
```

---

## 2. Generate the 35B expert index

```bash
cd /Users/sbaruwal/Repo/flash-moe-1
source .venv/bin/activate

python3 build_expert_index_35b.py   --model-path "$MODEL"   --out expert_index_35b.json
```

Output:

- `expert_index_35b.json`

---

## 3. Repack the 35B routed experts

Validate first with a dry run:

```bash
python3 repack_experts_35b.py --index expert_index_35b.json --layers 11 --dry-run
```

Then repack all expert layers:

```bash
python3 repack_experts_35b.py --index expert_index_35b.json
```

---

## 4. Extract non-expert weights for the Metal runtime

```bash
python3 metal_infer/extract_weights_35b.py --output metal_infer/out_35b
```

Outputs:

- `metal_infer/out_35b/model_weights.bin`
- `metal_infer/out_35b/model_weights.json`

---

## 5. Export tokenizer and vocab files

Two files are required:

- `tokenizer.bin` for prompt/chat tokenization
- `vocab.bin` for token decoding

### Export tokenizer

```bash
python3 metal_infer/export_tokenizer_35b.py   "$MODEL/tokenizer.json"   metal_infer/tokenizer.bin
```

### Export vocab

```bash
python3 metal_infer/export_vocab_35b.py   "$MODEL/tokenizer.json"   metal_infer/vocab.bin
```

Verify both files exist:

```bash
ls -lah metal_infer/tokenizer.bin metal_infer/vocab.bin
```

---

## 6. Build the Metal inference binary

```bash
cd /Users/sbaruwal/Repo/flash-moe-1/metal_infer

clang -O2 -Wall -fobjc-arc   -framework Metal   -framework Foundation   -framework Accelerate   -lpthread   infer.m -o infer
```

---

## 7. Smoke test direct generation

From the repo root:

```bash
cd /Users/sbaruwal/Repo/flash-moe-1

./metal_infer/infer   --model "$MODEL"   --weights metal_infer/out_35b/model_weights.bin   --manifest metal_infer/out_35b/model_weights.json   --vocab metal_infer/vocab.bin   --prompt "Mount Everest"   --tokens 32
```

A healthy startup should show:

- 40 layers
- 256 experts
- hidden size 2048
- MoE intermediate 512
- shared intermediate 512

---

## 8. Run the HTTP server

Start the server from the repo root:

```bash
cd /Users/sbaruwal/Repo/flash-moe-1

./metal_infer/infer   --model "$MODEL"   --weights metal_infer/out_35b/model_weights.bin   --manifest metal_infer/out_35b/model_weights.json   --vocab metal_infer/vocab.bin   --serve 8000
```

The server listens on:

```text
http://0.0.0.0:8000
```

Quick API test:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{"messages":[{"role":"user","content":"Mount Everest"}],"max_tokens":32,"stream":true}'
```

---

## 9. Build the chat client

From `metal_infer`:

```bash
cd /Users/sbaruwal/Repo/flash-moe-1/metal_infer

clang -O2 -Wall -fobjc-arc   -framework Foundation   chat.m linenoise.c   -o chat
```

If your Makefile supports it, this is also fine:

```bash
make chat
```

---

## 10. Launch chat

With the server already running in another terminal:

```bash
cd /Users/sbaruwal/Repo/flash-moe-1/metal_infer
./chat
```

The chat client connects to:

```text
http://localhost:8000
```

Supported commands in the client:

- `/quit`
- `/exit`
- `/clear`
- `/sessions`

---

## 11. End-to-end setup sequence

```bash
cd /Users/sbaruwal/Repo/flash-moe-1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install tokenizers
pip install numpy

MODEL=/Users/sbaruwal/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec

python3 build_expert_index_35b.py --model-path "$MODEL" --out expert_index_35b.json
python3 repack_experts_35b.py --index expert_index_35b.json
python3 metal_infer/extract_weights_35b.py --output metal_infer/out_35b
python3 metal_infer/export_tokenizer_35b.py "$MODEL/tokenizer.json" metal_infer/tokenizer.bin
python3 metal_infer/export_vocab_35b.py "$MODEL/tokenizer.json" metal_infer/vocab.bin

cd metal_infer
clang -O2 -Wall -fobjc-arc -framework Metal -framework Foundation -framework Accelerate -lpthread infer.m -o infer
clang -O2 -Wall -fobjc-arc -framework Foundation chat.m linenoise.c -o chat
```

Then start the server:

```bash
cd /Users/sbaruwal/Repo/flash-moe-1
./metal_infer/infer   --model "$MODEL"   --weights metal_infer/out_35b/model_weights.bin   --manifest metal_infer/out_35b/model_weights.json   --vocab metal_infer/vocab.bin   --serve 8000
```

And in a second terminal:

```bash
cd /Users/sbaruwal/Repo/flash-moe-1/metal_infer
./chat
```

---

## 12. Files that should exist at the end

```text
expert_index_35b.json
metal_infer/out_35b/model_weights.bin
metal_infer/out_35b/model_weights.json
metal_infer/tokenizer.bin
metal_infer/vocab.bin
metal_infer/infer
metal_infer/chat
```

---

## 13. Architecture notes

This 35B adaptation uses:

- 40 layers
- 256 experts
- hidden size 2048
- MoE intermediate size 512
- shared expert intermediate size 512

Also note:

- `tokenizer.bin` and `vocab.bin` are different files
- `tokenizer.bin` is used for prompt/chat tokenization
- `vocab.bin` is used for token decoding
