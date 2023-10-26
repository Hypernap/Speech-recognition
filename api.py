from flask import Flask, request, jsonify
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import spacy

app = Flask(__name__)

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.get_json()
        text = data['text']
        
        stopwords = list(STOP_WORDS)
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        
        word_frequencies = {}
        for word in doc:
            if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
        
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / max_frequency
        
        sentence_tokens = [sent for sent in doc.sents]
        
        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]
        
        select_length = int(len(sentence_tokens) * 0.3)
        summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
        
        summarized_text = ' '.join([str(s) for s in summary])
        
        return jsonify({"summary": summarized_text})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
