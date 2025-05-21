from flask import Flask, request, jsonify, session
import json
import pickle
import random
import re
import os
import logging
import numpy as np
from functools import lru_cache
from fuzzywuzzy import fuzz
from flask_cors import CORS
from flask_session import Session

# Configuration
class Config:
    MODEL_PATH = os.getenv('MODEL_PATH', 'chatbot_model.pkl')
    INTENTS_PATH = os.getenv('INTENTS_PATH', 'A.json')
    MAX_INPUT_LENGTH = int(os.getenv('MAX_INPUT_LENGTH', 500))
    SESSION_TYPE = os.getenv('SESSION_TYPE', 'filesystem')
    SESSION_FILE_DIR = os.getenv('SESSION_FILE_DIR', './flask_session')
    SECRET_KEY = os.getenv('SECRET_KEY', 'yudcslkknuhiurhqwpzvb')
    SESSION_COOKIE_NAME = 'ai_chatbot_session'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = True  # Enable in production
    SESSION_COOKIE_SAMESITE = 'Lax'
    ML_THRESHOLD = float(os.getenv('ML_THRESHOLD', 0.6))  # Confidence threshold for ML model
    FUZZY_THRESHOLD = int(os.getenv('FUZZY_THRESHOLD', 70))  # Similarity threshold for fuzzy matching

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize session before CORS
server_session = Session()
server_session.init_app(app) 

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://abrehamyetwale.netlify.app"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure server-side sessions
if app.config['SESSION_TYPE'] == 'filesystem':
    try:
        os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
        logging.info(f"Session directory created at {app.config['SESSION_FILE_DIR']}")
    except Exception as e:
        logging.error(f"Failed to create session directory: {str(e)}")
        raise e  # Halt app if session dir can't be created
Session(app)

# Set up logging
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

@app.route('/')
def home():
    return jsonify({'status': 'OK'}), 200

def detect_language(text):
    """Detect Amharic or English with improved regex"""
    return 'amharic' if re.search(r'[\u1200-\u137F]', text) else 'english'

@lru_cache(maxsize=1)
def load_model_cached(filename):
    """Cached model loader with version check"""
    try:
        with open(filename, 'rb') as file:
            model_data = pickle.load(file)
            # Handle both old and new model formats
            if len(model_data) == 3:
                model, vectorizer, label_encoder = model_data
                return {
                    'model': model,
                    'vectorizer': vectorizer,
                    'label_encoder': label_encoder,
                    'version': 1
                }
            else:
                return model_data  # Assume it's a dict with version info
    except Exception as e:
        logging.error(f"Model loading error: {str(e)}")
        return None

def load_intents_safe(filename):
    """Safe JSON loading with validation"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # Create intent index for faster lookup
            intents = data['intents']
            intent_index = {intent['tag']: intent for intent in intents}
            return {
                'intents': intents,
                'intent_index': intent_index
            }
    except Exception as e:
        logging.error(f"Intents loading error: {str(e)}")
        return {'intents': [], 'intent_index': {}}

def sanitize_input(text):
    """Basic input sanitization"""
    return re.sub(r'[<>{};]', '', text).strip()

def handle_personalization(user_input, context, language):
    """Improved personalization handler"""
    user_input = user_input.lower()
    
    if "ስሜ" in user_input or "my name is" in user_input:
        name = sanitize_input(user_input.split("ስሜ")[-1].split("my name is")[-1])
        if name:
            session['user_name'] = name
            return {
                'response': f"ስላዎኩዎት ደስ ብሎኛል, {name}!" if language == 'amharic' else f"Nice to meet you, {name}!",
                'status': 200
            }
    
    if session.get('user_name') and ("እንደምን አሉ?" in user_input or "how are you" in user_input):
        return {
            'response': f"ደና ነኝ, {session['user_name']}! እርስዎስ?" if language == 'amharic' else f"I'm fine, {session['user_name']}! How about you?",
            'status': 200
        }
    
    return None

def predict_with_ml(model_data, user_input, language):
    """Predict intent using ML model with confidence threshold"""
    try:
        # Vectorize the input
        X = model_data['vectorizer'].transform([user_input])
        
        # Get prediction probabilities
        if hasattr(model_data['model'], 'predict_proba'):
            probs = model_data['model'].predict_proba(X)[0]
            max_prob = np.max(probs)
            if max_prob < app.config['ML_THRESHOLD']:
                return None, max_prob
            
            pred_id = np.argmax(probs)
        else:
            # For models without predict_proba
            pred_id = model_data['model'].predict(X)[0]
            max_prob = 1.0  # Assume full confidence
            
        intent_tag = model_data['label_encoder'].inverse_transform([pred_id])[0]
        return intent_tag, max_prob
        
    except Exception as e:
        logging.error(f"ML prediction error: {str(e)}")
        return None, 0

def get_fuzzy_match(intents, user_input, language):
    """Get best fuzzy match for user input"""
    best_match, highest_similarity = None, 0
    for intent in intents:
        for pattern in intent['patterns'].get(language, []):
            similarity = fuzz.token_set_ratio(user_input, pattern.lower())
            if similarity > highest_similarity:
                highest_similarity, best_match = similarity, intent
                if highest_similarity == 100:  # Early exit if perfect match
                    return best_match, highest_similarity
    return best_match, highest_similarity

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = sanitize_input(data.get('message', ''))
        
        if not user_input:
            return jsonify({'error': 'Empty message'}), 400
            
        intents_data = load_intents_safe(app.config['INTENTS_PATH'])
        model_data = load_model_cached(app.config['MODEL_PATH'])
        
        if not intents_data['intents']:
            return jsonify({'error': 'Service unavailable - intents missing'}), 503
        
        language = detect_language(user_input)
        personal_response = handle_personalization(user_input, session, language)
        
        if personal_response:
            return jsonify({'response': personal_response['response']}), personal_response['status']
        
        # Try ML model first if available
        ml_intent = None
        ml_confidence = 0
        if model_data and model_data.get('model'):
            ml_intent, ml_confidence = predict_with_ml(model_data, user_input, language)
        
        response = None
        if ml_intent and ml_confidence >= app.config['ML_THRESHOLD']:
            # Use ML prediction if confidence is high enough
            intent = intents_data['intent_index'].get(ml_intent)
            if intent:
                response = random.choice(intent['responses'].get(language, []))
                logging.info(f"Used ML model (confidence: {ml_confidence:.2f}) for intent: {ml_intent}")
        
        # Fall back to fuzzy matching if ML didn't find a good match
        if not response:
            best_match, similarity = get_fuzzy_match(intents_data['intents'], user_input, language)
            if best_match and similarity >= app.config['FUZZY_THRESHOLD']:
                response = random.choice(best_match['responses'].get(language, []))
                logging.info(f"Used fuzzy matching (similarity: {similarity}) for intent: {best_match['tag']}")
        
        # Final fallback
        if not response:
            response = random.choice([
                "ይቅርታ፣ አልገባኝም። እባክዎን አስተካክለው እንደገና ይጠይቁ.",
                "Sorry, I didn't understand. Please try again."
            ])
            logging.info("No matching intent found")
        
        return jsonify({
            'response': response,
            'metadata': {
                'detected_language': language,
                'matched_method': 'ml' if ml_intent and ml_confidence >= app.config['ML_THRESHOLD'] else 'fuzzy',
                'confidence': ml_confidence if ml_intent else similarity/100 if 'similarity' in locals() else 0
            }
        })
    
    except Exception as e:
        error_msg = f'ይቅርታ፣ የሲስተም ስህተት ተፈጥሯል። {str(e)}' if detect_language(str(e)) == 'amharic' else f'Sorry, there was a system error. {str(e)}'
        logging.error(f"Chat error: {str(e)}")
        return jsonify({'error': error_msg}), 500

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))