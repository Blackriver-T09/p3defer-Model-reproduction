import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# 模型路径（假设已经下载）
MODEL_PATH = "./models/gemma-7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading server-side LLM model with float16 & device_map...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"  # 自动分配层到不同显卡或CPU
)
model.eval()
print("Model loaded successfully.")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    inputs = tokenizer(query, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            temperature=0.7
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return jsonify({"output": response})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
