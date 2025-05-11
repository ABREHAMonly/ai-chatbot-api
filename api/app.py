from flask import Flask, request, jsonify, session
import json
import pickle
import random
import re
import os
import logging
from functools import lru_cache
from fuzzywuzzy import fuzz
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_session import Session  # For server-side sessions

# Configuration
class Config:
    MODEL_PATH = os.getenv('MODEL_PATH', 'chatbot_model.pkl')
    INTENTS_PATH = os.getenv('INTENTS_PATH', 'A.json')
    MAX_INPUT_LENGTH = int(os.getenv('MAX_INPUT_LENGTH', 500))
    SESSION_TYPE = os.getenv('SESSION_TYPE', 'filesystem')  # Changed default
    SESSION_FILE_DIR = os.getenv('SESSION_FILE_DIR', './flask_session')  # New
    SECRET_KEY = os.getenv('SECRET_KEY', 'yudcslkknuhiurhqwpzvb')
    RATE_LIMIT = os.getenv('RATE_LIMIT', '5 per minute')

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Configure CORS for your Netlify domain
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://abrehamyetwale.netlify.app"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=app.config['RATELIMIT_STORAGE_URI'],
    default_limits=[app.config['RATE_LIMIT']]
)

# Configure server-side sessions
if app.config['SESSION_TYPE'] == 'filesystem':
    os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
Session(app)

# Set up logging
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def detect_language(text):
    """Detect Amharic or English with improved regex"""
    return 'amharic' if re.search(r'[\u1200-\u137F]', text) else 'english'

@lru_cache(maxsize=1)
def load_model_cached(filename):
    """Cached model loader with version check"""
    try:
        with open(filename, 'rb') as file:
            model, vectorizer, label_encoder = pickle.load(file)
            
            # Add version compatibility check
            if not hasattr(model, 'model_version'):
                logging.warning('No version found in model')
            elif model.model_version != '1.0':
                raise ValueError("Model version mismatch")
                
            return model, vectorizer, label_encoder
    except Exception as e:
        logging.error(f"Model loading error: {str(e)}")
        return None, None, None

def load_intents_safe(filename):
    """Safe JSON loading with validation"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # Basic validation
            if 'intents' not in data:
                raise ValueError("Invalid intents file structure")
                
            return data['intents']
    except Exception as e:
        logging.error(f"Intents loading error: {str(e)}")
        return []

def sanitize_input(text):
    """Basic input sanitization"""
    return re.sub(r'[<>{};]', '', text).strip()

def handle_personalization(user_input, context, language):
    """Improved personalization handler"""
    user_input = user_input.lower()
    
    # Check for name storage
    if "ስሜ" in user_input or "my name is" in user_input:
        name = sanitize_input(user_input.split("ስሜ")[-1].split("my name is")[-1])
        if name:
            session['user_name'] = name
            return {
                'response': (
                    f"ስላዎኩዎት ደስ ብሎኛል, {name}!" if language == 'amharic' 
                    else f"Nice to meet you, {name}!"
                ),
                'status': 200
            }
    
    # Check for personal inquiry
    if session.get('user_name') and ("እንደምን አሉ?" in user_input or "how are you" in user_input):
        return {
            'response': (
                f"ደና ነኝ, {session['user_name']}! እርስዎስ?" if language == 'amharic'
                else f"I'm fine, {session['user_name']}! How about you?"
            ),
            'status': 200
        }
    
    return None

@app.route('/')
def health_check():
    return jsonify({"status": "healthy", "version": "1.0.0"})

@app.route('/api/chat', methods=['POST'])
@limiter.limit(app.config['RATE_LIMIT'])
def chat():
    """Enhanced chat endpoint with security and validation"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Invalid content type'}), 415
            
        data = request.get_json()
        user_input = sanitize_input(data.get('message', ''))
        
        # Input validation
        if not user_input:
            return jsonify({'error': 'Empty message'}), 400
            
        if len(user_input) > app.config['MAX_INPUT_LENGTH']:
            return jsonify({'error': 'Message too long'}), 413
        
        # Load resources
        intents = load_intents_safe(app.config['INTENTS_PATH'])
        model, vectorizer, label_encoder = load_model_cached(app.config['MODEL_PATH'])
        
        if not intents or not model:
            logging.error("Critical resources failed to load")
            return jsonify({'error': 'Service unavailable'}), 503
        
        # Language detection
        language = detect_language(user_input)
        
        # Check for personalization
        personal_response = handle_personalization(user_input, session, language)
        if personal_response:
            return jsonify({'response': personal_response['response']}), personal_response['status']
        
        # Intent matching
        best_match = None
        highest_similarity = 0
        
        for intent in intents:
            for pattern in intent['patterns'].get(language, []):
                similarity = fuzz.token_set_ratio(user_input, pattern.lower())
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = intent
        
        # Generate response
        if best_match and highest_similarity >= 70:
            response = random.choice(best_match['responses'].get(language, []))
        else:
            default_responses = [
                "ይቅርታ፣ አልገባኝም። እባክዎን አስተካክለው እንደገና ይጠይቁ.",
                "Sorry, I didn't understand. Please try again."
            ]
            response = random.choice(default_responses)
        
        # Log successful interaction
        logging.info(f"Processed message: {user_input} | Response: {response}")
        
        return jsonify({'response': response})
    
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        language = detect_language(user_input) if 'user_input' in locals() else 'english'
        error_msg = (
            f'ይቅርታ፣ የሲስተም ስህተት ተፈጥሯል። {str(e)}' if language == 'amharic'
            else f'Sorry, there was a system error. {str(e)}'
        )
        return jsonify({'error': error_msg}), 500

@app.after_request
def add_security_headers(response):
    """Add security headers for production"""
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

if __name__ == '__main__':
    # Production configuration
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(
        debug=debug_mode,
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        use_reloader=debug_mode
    )