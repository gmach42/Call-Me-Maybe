# explore_llm.py  (put this at the root of your project, same level as src/)
import json
import torch
import llm_sdk

model = llm_sdk.Small_LLM_Model()

# ─────────────────────────────────────────────
# 1. VOCABULARY EXPLORATION
# ─────────────────────────────────────────────

vocab_path = model.get_path_to_vocab_file()
with open(vocab_path) as f:
    raw_vocab = json.load(f)

# Support both shapes:
# 1) {"0": "<unk>", "1": "<s>"}   -> id(str) -> token
# 2) {"<unk>": 0, "<s>": 1}       -> token -> id
sample_key = next(iter(raw_vocab))
sample_val = raw_vocab[sample_key]

if isinstance(sample_key, str) and sample_key.isdigit():
    # id(str) -> token
    vocab = {int(k): v for k, v in raw_vocab.items()}
    reverse_vocab = {v: k for k, v in vocab.items()}
elif isinstance(sample_val, int):
    # token -> id
    reverse_vocab = dict(raw_vocab)
    vocab = {token_id: token_str for token_str, token_id in reverse_vocab.items()}
else:
    raise ValueError("Unsupported vocabulary JSON format")

# vocab is  { "0": "<unk>", "1": "<s>", ... }
# keys are strings, so let's normalize to int keys
vocab = {int(k): v for k, v in vocab.items()}
reverse_vocab = {v: k for k, v in vocab.items()}

print(f"Vocabulary size: {len(vocab)}")
print(f"First 10 tokens: {[(k, vocab[k]) for k in range(10)]}")

# Look for function-related tokens
print("\n--- Tokens containing 'fn_' ---")
for token_str, token_id in reverse_vocab.items():
    if "fn_" in token_str:
        print(f"  id={token_id:6d}  repr={repr(token_str)}")

# Look for JSON structural tokens
print("\n--- JSON structural tokens ---")
for target in ["{", "}", '"', ":", ",", "true", "false", "null"]:
    if target in reverse_vocab:
        print(f"  {repr(target):10s} → id={reverse_vocab[target]}")
    else:
        # might exist with a leading space, e.g. " true"
        for token_str, token_id in reverse_vocab.items():
            if token_str.strip() == target:
                print(f"  {repr(token_str):10s} → id={token_id}  (has leading space)")

# ─────────────────────────────────────────────
# 2. ENCODING / DECODING ROUNDTRIP
# ─────────────────────────────────────────────

print("\n--- Encode / decode roundtrip ---")
test_strings = [
    "What is the sum of 2 and 3?",
    '{"name": "fn_add_numbers"',
    "fn_add_numbers",
    "true",
    "42",
    "3.14",
]

for s in test_strings:
    ids = model.encode(s)
    decoded = model.decode(ids)
    tokens = [repr(vocab.get(i, "???")) for i in ids]
    print(f"\n  input:   {repr(s)}")
    print(f"  ids:     {ids}")
    print(f"  tokens:  {tokens}")
    print(f"  decoded: {repr(decoded)}")

# ─────────────────────────────────────────────
# 3. RAW LOGITS — WHAT DOES THE MODEL WANT NEXT?
# ─────────────────────────────────────────────

print("\n--- Raw next-token prediction ---")

def get_top_k(prompt: str, k: int = 10) -> None:
    ids = model.encode(prompt)
    tensor = torch.tensor([ids])
    logits = model.get_logits_from_input_ids(tensor)  # shape: [vocab_size]

    topk = torch.topk(logits, k)
    print(f"\n  prompt: {repr(prompt)}")
    print(f"  top {k} predicted next tokens:")
    for score, token_id in zip(topk.values.tolist(), topk.indices.tolist()):
        token_str = vocab.get(token_id, "???")
        print(f"    id={token_id:6d}  score={score:8.3f}  repr={repr(token_str)}")

get_top_k("What is the sum of 2 and 3?")
get_top_k('{"name": "')
get_top_k('{"name": "fn_add_numbers", "parameters": {"a": ')

# ─────────────────────────────────────────────
# 4. NAIVE GENERATION (no constraints) — see how it fails
# ─────────────────────────────────────────────

print("\n--- Naive greedy generation (unconstrained) ---")

def generate_naive(prompt: str, max_new_tokens: int = 60) -> str:
    ids = model.encode(prompt)
    for _ in range(max_new_tokens):
        tensor = torch.tensor([ids])
        logits = model.get_logits_from_input_ids(tensor)
        next_token = torch.argmax(logits).item()
        ids.append(next_token)

        # stop on EOS token (usually id=2 or model-specific)
        if next_token in (2,):
            break

    generated_ids = ids[len(model.encode(prompt)):]
    return model.decode(generated_ids)

prompts_to_test = [
    "What is the sum of 2 and 3? Answer in JSON: ",
    "Greet John. Answer in JSON: ",
]

for prompt in prompts_to_test:
    result = generate_naive(prompt)
    print(f"\n  prompt:   {repr(prompt)}")
    print(f"  output:   {repr(result)}")
    try:
        parsed = json.loads(result)
        print(f"  valid JSON: YES → {parsed}")
    except json.JSONDecodeError as e:
        print(f"  valid JSON: NO  → {e}")
