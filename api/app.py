from flask import Flask, request, jsonify
import json
import pickle
import random
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://abrehamyetwale.netlify.app"}})

def detect_language(text):
    return 'amharic' if re.search(r'[\u1200-\u137F]', text) else 'english'

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({'error': 'Empty message'}), 400
        
        # Load intents directly (ensure A.json exists)
        with open('A.json', 'r', encoding='utf-8') as f:
            intents = json.load(f)['intents']
        
        # Load model directly (ensure chatbot_model.pkl exists)
        with open('chatbot_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        language = detect_language(user_input)
        user_input = user_input.lower()

        # Simplified matching logic
        best_match = None
        highest_score = 0
        
        for intent in intents:
            for pattern in intent['patterns'].get(language, []):
                score = sum(1 for word in user_input.split() if word in pattern.lower())
                if score > highest_score:
                    best_match, highest_score = intent, score
        
        if best_match and highest_score >= 1:
            response = random.choice(best_match['responses'].get(language, []))
        else:
            response = random.choice([
                "ይቅርታ፣ አልገባኝም። እባክዎን እንደገና ይሞክሩ።",
                "Sorry, I didn't understand. Please try again."
            ])
        
        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'error': f'System error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))