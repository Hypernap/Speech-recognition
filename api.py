from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM,AutoTokenizer , AutoModelForSeq2SeqLM
import torch
import tokenizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={r"/summarize": {"origins": "http://localhost:3000"}})
@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.get_json()
        text = data['text']
        
        text1 = text[0:int(len(text)/2)]
        text2 = text[int(len(text)/2):]
        # Load the pre-trained GPT-2 model and tokenizer
        model_name = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_path = "C:\codes\Speech recognition\sum"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        # Tokenize the input text
        inputs1 = tokenizer.encode("summarize: " + text1, return_tensors="pt", max_length=1024, truncation=True)
        inputs2 = tokenizer.encode("summarize: " + text2, return_tensors="pt", max_length=1024, truncation=True)
        # Generate a summary
        summary_ids1 = model.generate(inputs1, max_length=200, num_beams=4, early_stopping=True)
        summary_ids2 = model.generate(inputs2, max_length=200, num_beams=4, early_stopping=True)
        # Decode the summary
        summary1 = tokenizer.decode(summary_ids1[0], skip_special_tokens=True)
        summary2 = tokenizer.decode(summary_ids2[0], skip_special_tokens=True)
        
        return jsonify({"summary": summary1+summary2})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
