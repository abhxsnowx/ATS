# test_hf.py
from huggingface_hub import InferenceClient

client = InferenceClient(token="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")

# After enabling providers, test these
models = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "HuggingFaceH4/zephyr-7b-beta",
]

for model in models:
    try:
        r = client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one word"}],
            max_tokens=10,
        )
        print(f"✅ {model} → {r.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"❌ {model} → {str(e)[:80]}")