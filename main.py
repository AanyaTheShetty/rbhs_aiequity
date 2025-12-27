from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import json
import requests
import google.generativeai as genai
from PIL import Image
from gradio_client import Client, handle_file
import tempfile
from werkzeug.utils import secure_filename
import time

# Common food items that work well with Calorie Ninja API
CALORIE_NINJA_FOODS = [
    "apple", "banana", "orange", "strawberry", "blueberry", "raspberry", "blackberry",
    "grape", "watermelon", "mango", "pineapple", "peach", "pear", "plum", "cherry",
    "avocado", "tomato", "cucumber", "carrot", "broccoli", "spinach", "lettuce",
    "kale", "cabbage", "cauliflower", "bell pepper", "onion", "garlic", "potato",
    "sweet potato", "corn", "peas", "green beans", "asparagus", "celery", "mushroom",
    "chicken breast", "chicken thigh", "turkey", "beef", "pork", "lamb", "bacon",
    "sausage", "ham", "salmon", "tuna", "cod", "shrimp", "crab", "lobster",
    "egg", "milk", "cheese", "yogurt", "butter", "cream", "ice cream",
    "rice", "pasta", "bread", "oatmeal", "quinoa", "couscous", "noodles",
    "pizza", "burger", "hot dog", "sandwich", "taco", "burrito", "sushi",
    "salad", "soup", "steak", "chicken wings", "french fries", "onion rings",
    "cookie", "cake", "brownie", "donut", "muffin", "pancake", "waffle",
    "almond", "cashew", "peanut", "walnut", "pistachio", "peanut butter",
    "honey", "maple syrup", "sugar", "chocolate", "coffee", "tea", "juice",
    "beans", "lentils", "chickpeas", "tofu", "tempeh", "hummus",
    "olive oil", "coconut oil", "vinegar", "soy sauce", "ketchup", "mayonnaise",
    "apple pie", "cheesecake", "tiramisu", "pudding", "jello",
    "bagel", "croissant", "biscuit", "tortilla", "pita bread",
    "beef stew", "chicken soup", "chili", "curry", "stir fry",
    "smoothie", "protein shake", "energy bar", "granola", "cereal"
]

app = Flask(__name__)
app.secret_key = "password" 

# Set up API keys
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
CALORIENINJA_API_KEY = os.environ.get("CALORIENINJA_API_KEY")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gradio client
GRADIO_CLIENT = None

def get_gradio_client():
    """Lazy initialization of Gradio client"""
    global GRADIO_CLIENT
    if GRADIO_CLIENT is None:
        try:
            print("Initializing Gradio client for fredsok/ingredientsmodel...")
            GRADIO_CLIENT = Client("calcuplate/ingredientClassificationModel")
            print("✓ Gradio client initialized successfully")
        except Exception as e:
            print(f"Error initializing Gradio client: {e}")
            raise
    return GRADIO_CLIENT

# ============== HELPER FUNCTIONS ==============

def call_gemini_with_retry(model, prompt, max_retries=3, initial_delay=1):
    """
    Call Gemini API with exponential backoff retry logic
    
    Args:
        model: Gemini model instance
        prompt: The prompt to send
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
    
    Returns:
        Response text from Gemini
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            
            # Check if it's a retryable error
            is_retryable = (
                'rate limit' in error_msg or
                'quota' in error_msg or
                'timeout' in error_msg or
                'temporarily unavailable' in error_msg or
                '429' in error_msg or
                '503' in error_msg or
                '500' in error_msg
            )
            
            if not is_retryable or attempt == max_retries - 1:
                # Don't retry for non-retryable errors or on last attempt
                raise
            
            # Calculate delay with exponential backoff
            delay = initial_delay * (2 ** attempt)
            print(f"Gemini API attempt {attempt + 1} failed: {e}")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
    
    # This shouldn't be reached, but just in case
    raise last_error

def clean_ingredient_name(name):
    """Clean up ingredient name for display"""
    # Remove parentheses content and underscores
    name = name.split('_(')[0]  # Remove (meat), (fish), etc.
    name = name.replace('_', ' ')
    return name.title()

def predict_ingredients_gradio(file_path):
    """Predict ingredients from image file using Gradio model"""
    try:
        print("=" * 50)
        print("CALLING GRADIO MODEL API")
        print(f"Image file path: {file_path}")
        
        # Verify file exists and has content
        if not os.path.exists(file_path):
            raise Exception("Image file does not exist")
        
        file_size = os.path.getsize(file_path)
        print(f"Image file size: {file_size} bytes")
        
        if file_size == 0:
            raise Exception("Image file is empty")
        
        # Get Gradio client
        client = get_gradio_client()
        
        # Call the prediction API with the file path directly
        print("Calling Gradio predict API...")
        result = client.predict(
            image=handle_file(file_path),
            api_name="/predict"
        )
        
        print(f"Gradio API Result: {result}")
        
        # Process results
        predictions = []
        
        # The result format depends on the model output
        # Common formats: dict with 'label' and 'confidences', or list of tuples
        if isinstance(result, dict):
            # Format: {'label': 'apple', 'confidences': [{'label': 'apple', 'confidence': 0.95}, ...]}
            if 'confidences' in result:
                for item in result['confidences'][:5]:  # Top 5
                    clean_name = clean_ingredient_name(item.get('label', ''))
                    predictions.append({
                        'name': clean_name,
                        'value': item.get('confidence', 0),
                        'raw_name': item.get('label', '')
                    })
            elif 'label' in result:
                # Single prediction format
                clean_name = clean_ingredient_name(result['label'])
                predictions.append({
                    'name': clean_name,
                    'value': result.get('confidence', 0.9),  # Default confidence if not provided
                    'raw_name': result['label']
                })
        elif isinstance(result, list):
            # Format: [('apple', 0.95), ('banana', 0.03), ...]
            for item in result[:5]:  # Top 5
                if isinstance(item, tuple) and len(item) >= 2:
                    label, confidence = item[0], item[1]
                    clean_name = clean_ingredient_name(label)
                    predictions.append({
                        'name': clean_name,
                        'value': float(confidence),
                        'raw_name': label
                    })
                elif isinstance(item, dict):
                    clean_name = clean_ingredient_name(item.get('label', ''))
                    predictions.append({
                        'name': clean_name,
                        'value': item.get('score', 0) or item.get('confidence', 0),
                        'raw_name': item.get('label', '')
                    })
        elif isinstance(result, str):
            # Single string result
            clean_name = clean_ingredient_name(result)
            predictions.append({
                'name': clean_name,
                'value': 0.9,  # Default confidence
                'raw_name': result
            })
        
        if not predictions:
            raise Exception(f"No predictions returned from Gradio API. Raw result: {result}")
        
        print("=" * 50)
        print("GRADIO MODEL PREDICTIONS:")
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['name']}: {pred['value']*100:.2f}%")
        print("=" * 50)
        
        return predictions
        
    except Exception as e:
        print(f"Error in predict_ingredients_gradio: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/home")
def home():
    return render_template("landing.html")

@app.route("/reminders")
def reminder():
    return render_template("reminders.html")

@app.route("/treatment_info")
def treatment_info():
    return render_template("treatment_info.html")    
    
@app.route('/upload', methods=['GET', 'POST'])
def identify_food():
    if request.method == "POST":
        temp_file_path = None
        try:
            # ================== ACCEPT multipart/form-data ==================
            if "image" not in request.files:
                return jsonify({'error': 'No image file uploaded'}), 400

            file = request.files["image"]
            
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            # Optional: nutritional needs
            raw_needs = request.form.get("nutritional_needs", "[]")
            try:
                nutritional_needs = json.loads(raw_needs)
            except:
                nutritional_needs = []

            # Save uploaded file to temporary location
            filename = secure_filename(file.filename)
            # Get file extension
            _, ext = os.path.splitext(filename)
            if not ext:
                ext = '.jpg'
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                file.save(temp_file.name)
                temp_file_path = temp_file.name
            
            print(f"Image saved to temporary file: {temp_file_path}")
            print(f"File size: {os.path.getsize(temp_file_path)} bytes")

            # ================== RUN GRADIO PREDICTOR ==================
            predictions = predict_ingredients_gradio(temp_file_path)

            if not predictions:
                return jsonify({'error': 'No ingredients detected'}), 400

            # Top prediction
            top_prediction = predictions[0]
            top_food = top_prediction['name']
            max_confidence = top_prediction['value']

            # ================== CALORIE NINJA NUTRITION ==================
            nutrition_data = None
            try:
                if CALORIENINJA_API_KEY:
                    resp = requests.get(
                        f'https://api.calorieninjas.com/v1/nutrition?query={top_food}',
                        headers={'X-Api-Key': CALORIENINJA_API_KEY},
                        timeout=6
                    )
                    if resp.status_code == 200:
                        items = resp.json().get('items', [])
                        if items:
                            item = items[0]
                            nutrition_data = {
                                'calories': item.get('calories', 0),
                                'protein_g': item.get('protein_g', 0),
                                'carbohydrates_total_g': item.get('carbohydrates_total_g', 0),
                                'fat_total_g': item.get('fat_total_g', 0),
                                'fiber_g': item.get('fiber_g', 0),
                                'sugar_g': item.get('sugar_g', 0),
                                'sodium_mg': item.get('sodium_mg', 0),
                                'serving_size_g': item.get('serving_size_g', 100)
                            }
            except Exception as e:
                print("CalorieNinja error:", e)

            # ================== GEMINI ADVICE (simple prompt) ==================
            gemini_advice = None
            try:
                if GOOGLE_API_KEY:
                    if nutrition_data:
                        calories = nutrition_data['calories']
                        protein = nutrition_data['protein_g']
                        carbs = nutrition_data['carbohydrates_total_g']
                        fat = nutrition_data['fat_total_g']
                        fiber = nutrition_data['fiber_g']
                        sugar = nutrition_data['sugar_g']
                        sodium = nutrition_data['sodium_mg']

                        if nutritional_needs:
                            needs_str = ", ".join(nutritional_needs)
                            prompt = (
                                f"You are a nutrition expert. The food identified is {top_food}. "
                                f"Nutritional information: {calories} calories, {protein}g protein, "
                                f"{carbs}g carbohydrates, {fat}g fat, {fiber}g fiber, {sugar}g sugar, {sodium}mg sodium. "
                                f"The person has the following nutritional needs/preferences: {needs_str}. "
                                f"In 2-3 sentences, provide practical, actionable advice about whether this food is a good choice for their needs. "
                                f"Be specific about how the nutritional content aligns (or doesn't align) with their requirements. "
                                f"Keep it conversational and supportive."
                            )
                        else:
                            prompt = (
                                f"You are a nutrition expert. The food identified is {top_food}. "
                                f"Nutritional information: {calories} calories, {protein}g protein, "
                                f"{carbs}g carbohydrates, {fat}g fat, {fiber}g fiber, {sugar}g sugar, {sodium}mg sodium. "
                                f"In 2-3 sentences, provide practical, actionable health advice about this food. "
                                f"Is this generally a good nutritional choice? What are the key benefits or concerns? "
                                f"Keep it conversational and supportive."
                            )
                    else:
                        prompt = (
                            f"You are a nutrition expert. The food identified is {top_food}. "
                            f"In 2-3 sentences, provide practical health advice about this food. "
                            f"Is this generally a good nutritional choice? What are the key benefits or concerns? "
                            f"Keep it conversational and supportive."
                        )
                
                print(f"Prompt (first 100 chars): {prompt[:100]}...")
                print("Calling Gemini API NOW...")
                
                model = genai.GenerativeModel(
                    "gemini-2.5-flash",
                    generation_config={
                        "response_mime_type": "text/plain"
                    }
                )

                gemini_advice = call_gemini_with_retry(model, prompt, max_retries=3, initial_delay=1)
                
                print(f"SUCCESS! Gemini advice received ({len(gemini_advice)} characters)")
                print(f"Advice preview: {gemini_advice[:100]}...")
                print("=" * 50)
                
            except Exception as gemini_error:
                print("=" * 50)
                print(f"GEMINI ERROR: {str(gemini_error)}")
                print("Error type:", type(gemini_error).__name__)
                import traceback
                traceback.print_exc()
                print("=" * 50)
            
            # Build response in Clarifai format for compatibility with frontend
            response_data = {
                'outputs': [
                    {
                        'data': {
                            'concepts': []
                        }
                    }
                ]
            }
            
            # Add all predictions as concepts
            for pred in predictions:
                response_data['outputs'][0]['data']['concepts'].append({
                    'name': pred['name'],
                    'value': pred['value'],
                    'nutrition': nutrition_data if pred == predictions[0] else None,
                    'gemini_advice': gemini_advice if pred == predictions[0] else None
                })

            return jsonify(response_data), 200

        except Exception as e:
            print("UPLOAD ERROR:", e)
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    print(f"Temporary file deleted: {temp_file_path}")
                except Exception as e:
                    print(f"Error deleting temporary file: {e}")

    return render_template("upload.html")



@app.route("/symptomTracker")
def symptom():
    return render_template("symptomReport.html")

@app.route("/glucose")
def glucose():
    return render_template("glucose.html")

@app.route("/settings")
def settings():
    return render_template("settings.html")

@app.route("/test-gemini")
def test_gemini():
    """Test endpoint to verify Gemini is working"""
    try:
        if not GOOGLE_API_KEY:
            return jsonify({'error': 'GOOGLE_API_KEY not set'}), 500
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content("Say hello in one sentence")
        
        return jsonify({
            'success': True,
            'message': 'Gemini is working!',
            'response': response.text
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/test-gradio")
def test_gradio():
    """Test endpoint to verify Gradio API is working"""
    try:
        client = get_gradio_client()
        
        return jsonify({
            'success': True,
            'message': 'Gradio client initialized successfully',
            'model': 'fredsok/ingredientsmodel'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/api/food-list")
def get_food_list():
    """Return list of supported food items"""
    return jsonify({
        'success': True,
        'foods': sorted(CALORIE_NINJA_FOODS)
    })

@app.route("/api/override-food", methods=['POST'])
def override_food():
    """Override detected food with user-selected food"""
    try:
        data = request.get_json()
        
        if not data or 'food_name' not in data:
            return jsonify({'error': 'Food name is required'}), 400
        
        food_name = data['food_name'].strip().lower()
        nutritional_needs = data.get('nutritional_needs', [])
        
        print(f"Override request for food: {food_name}")
        
        # Get nutrition data from Calorie Ninja
        nutrition_data = None
        try:
            if CALORIENINJA_API_KEY:
                resp = requests.get(
                    f'https://api.calorieninjas.com/v1/nutrition?query={food_name}',
                    headers={'X-Api-Key': CALORIENINJA_API_KEY},
                    timeout=6
                )
                if resp.status_code == 200:
                    items = resp.json().get('items', [])
                    if items:
                        item = items[0]
                        nutrition_data = {
                            'calories': item.get('calories', 0),
                            'protein_g': item.get('protein_g', 0),
                            'carbohydrates_total_g': item.get('carbohydrates_total_g', 0),
                            'fat_total_g': item.get('fat_total_g', 0),
                            'fiber_g': item.get('fiber_g', 0),
                            'sugar_g': item.get('sugar_g', 0),
                            'sodium_mg': item.get('sodium_mg', 0),
                            'serving_size_g': item.get('serving_size_g', 100)
                        }
        except Exception as e:
            print(f"CalorieNinja error: {e}")
        
        if not nutrition_data:
            return jsonify({'error': 'Could not fetch nutrition data for this food'}), 400
        
        # Generate Gemini advice
        gemini_advice = None
        try:
            if GOOGLE_API_KEY:
                calories = nutrition_data['calories']
                protein = nutrition_data['protein_g']
                carbs = nutrition_data['carbohydrates_total_g']
                fat = nutrition_data['fat_total_g']
                fiber = nutrition_data['fiber_g']
                sugar = nutrition_data['sugar_g']
                sodium = nutrition_data['sodium_mg']

                if nutritional_needs:
                    needs_str = ", ".join(nutritional_needs)
                    prompt = (
                        f"You are a nutrition expert. The food identified is {food_name}. "
                        f"Nutritional information: {calories} calories, {protein}g protein, "
                        f"{carbs}g carbohydrates, {fat}g fat, {fiber}g fiber, {sugar}g sugar, {sodium}mg sodium. "
                        f"The person has the following nutritional needs/preferences: {needs_str}. "
                        f"In 2-3 sentences, provide practical, actionable advice about whether this food is a good choice for their needs. "
                        f"Be specific about how the nutritional content aligns (or doesn't align) with their requirements. "
                        f"Keep it conversational and supportive."
                    )
                else:
                    prompt = (
                        f"You are a nutrition expert. The food identified is {food_name}. "
                        f"Nutritional information: {calories} calories, {protein}g protein, "
                        f"{carbs}g carbohydrates, {fat}g fat, {fiber}g fiber, {sugar}g sugar, {sodium}mg sodium. "
                        f"In 2-3 sentences, provide practical, actionable health advice about this food. "
                        f"Is this generally a good nutritional choice? What are the key benefits or concerns? "
                        f"Keep it conversational and supportive."
                    )
            
                print(f"Calling Gemini for override advice...")
                
                model = genai.GenerativeModel(
                    "gemini-2.5-flash",
                    generation_config={
                        "response_mime_type": "text/plain"
                    }
                )

                gemini_advice = call_gemini_with_retry(model, prompt, max_retries=3, initial_delay=1)
                
                print(f"Gemini advice received: {gemini_advice[:100]}...")
                
        except Exception as gemini_error:
            print(f"Gemini error: {str(gemini_error)}")
        
        # Build response
        response_data = {
            'success': True,
            'food': {
                'name': food_name.title(),
                'nutrition': nutrition_data,
                'gemini_advice': gemini_advice
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"Override error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route("/api/chatbot", methods=['POST'])
def chatbot():
    """Diabetes assistant chatbot endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        user_message = data['message'].strip()
        conversation_history = data.get('history', [])
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        print(f"Chatbot request: {user_message}")
        
        if not GOOGLE_API_KEY:
            return jsonify({
                'success': False,
                'error': 'Chatbot service is currently unavailable. Please try again later.'
            }), 503
        
        # Enhanced system prompt with topic filtering built-in
        system_prompt = """You are a helpful diabetes assistant specialized in helping diabetic patients. 
You can answer questions about:
- Glucose monitoring and blood sugar management
- Nutrition, diet planning, and food choices for diabetics
- Diabetes medications and treatments
- Symptoms, complications, and general diabetes care
- Lifestyle modifications for diabetes management
- Exercise and physical activity for diabetics
- Stress management and mental health related to diabetes

CRITICAL INSTRUCTION - STAY ON TOPIC:
If the user asks about topics UNRELATED to diabetes (like politics, sports, general knowledge, other medical conditions, etc.), you MUST respond with:
"I'm specialized in diabetes care and management. I can help with questions about glucose monitoring, nutrition for diabetics, medications, symptoms, and lifestyle management. Is there anything diabetes-related I can help you with?"

RESPONSE GUIDELINES:
1. Provide accurate, helpful, and supportive information
2. Be concise but thorough - aim for 2-4 paragraphs
3. Always recommend consulting healthcare professionals for medical decisions
4. Use a friendly, encouraging, and empathetic tone
5. Break down complex topics into easy-to-understand language
6. Provide practical, actionable advice when appropriate
"""
        
        # Build conversation context
        messages = []
        
        # Add recent history (last 10 messages to keep context manageable)
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        for msg in recent_history:
            messages.append({
                'role': msg['role'],
                'parts': [msg['content']]
            })
        
        # Generate response using Gemini with retry logic
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={
                "response_mime_type": "text/plain",
                "temperature": 0.7,
                "max_output_tokens": 1024,  # Limit response length
            },
            system_instruction=system_prompt
        )
        
        # Create chat session with history
        chat = model.start_chat(history=messages)
        
        # Send the user's message with retry logic
        try:
            response_text = call_gemini_with_retry(chat, user_message, max_retries=3, initial_delay=1)
        except Exception as gemini_error:
            error_msg = str(gemini_error).lower()
            
            # Provide specific error messages based on error type
            if 'rate limit' in error_msg or 'quota' in error_msg or '429' in error_msg:
                return jsonify({
                    'success': False,
                    'error': 'The chatbot is experiencing high demand. Please wait a moment and try again.'
                }), 429
            elif 'timeout' in error_msg:
                return jsonify({
                    'success': False,
                    'error': 'The request timed out. Please try again.'
                }), 504
            elif 'api key' in error_msg or 'authentication' in error_msg:
                return jsonify({
                    'success': False,
                    'error': 'Chatbot service configuration error. Please contact support.'
                }), 503
            else:
                # Generic error
                print(f"Gemini API error: {gemini_error}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': 'I apologize, but I encountered an error. Please try again.'
                }), 500
        
        print(f"Chatbot response: {response_text[:100]}...")
        
        return jsonify({
            'success': True,
            'response': response_text
        }), 200
        
    except Exception as e:
        print(f"Chatbot error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred. Please try again.'
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
