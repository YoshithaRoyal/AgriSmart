import os
import random
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.config['SECRET_KEY'] = 'agrismart_super_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agri.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    histories = db.relationship('History', backref='user', lazy=True, cascade="all, delete-orphan")

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    n = db.Column(db.Float)
    p = db.Column(db.Float)
    k = db.Column(db.Float)
    temp = db.Column(db.Float)
    humidity = db.Column(db.Float)
    ph = db.Column(db.Float)
    rainfall = db.Column(db.Float)
    crop = db.Column(db.String(100))
    fertilizer = db.Column(db.String(100))
    profit = db.Column(db.String(100))

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

DATASET_PATH = 'Crop_recommendation.csv'
MODEL = None
IDEAL_METRICS = {}

# Crop Growing Tips Mapping
CROP_TIPS = {
    'rice': {'season': 'Kharif (Monsoon)', 'harvest': 'Winter', 'tip': 'Requires high water availability; ensure proper field flooding.'},
    'maize': {'season': 'Kharif', 'harvest': 'Autumn', 'tip': 'Needs well-drained soil; avoid waterlogging.'},
    'chickpea': {'season': 'Rabi (Winter)', 'harvest': 'Spring', 'tip': 'Highly sensitive to excess water; grows well in cool climates.'},
    'kidneybeans': {'season': 'Kharif', 'harvest': 'Autumn', 'tip': 'Prefers loamy soil; sensitive to frost.'},
    'pigeonpeas': {'season': 'Kharif', 'harvest': 'Winter/Spring', 'tip': 'Deep root system makes it drought-tolerant.'},
    'mothbeans': {'season': 'Kharif', 'harvest': 'Autumn', 'tip': 'Extremely drought resistant; requires very little water.'},
    'mungbean': {'season': 'Kharif / Zaid', 'harvest': 'Varies', 'tip': 'Short duration crop; improves soil fertility.'},
    'blackgram': {'season': 'Kharif', 'harvest': 'Autumn', 'tip': 'Requires warm weather; intolerant to frost.'},
    'lentil': {'season': 'Rabi', 'harvest': 'Spring', 'tip': 'Prefers cool climate and moderately fertile soil.'},
    'pomegranate': {'season': 'Perennial', 'harvest': 'Late Summer', 'tip': 'Requires hot and dry climate during fruit development.'},
    'banana': {'season': 'Perennial', 'harvest': 'Year-round', 'tip': 'Requires heavy rainfall and high humidity.'},
    'mango': {'season': 'Perennial', 'harvest': 'Summer', 'tip': 'Needs distinct dry season for flowering and fruiting.'},
    'grapes': {'season': 'Perennial', 'harvest': 'Summer/Autumn', 'tip': 'Requires pruning and support structures; sensitive to high humidity.'},
    'watermelon': {'season': 'Summer / Zaid', 'harvest': 'Late Summer', 'tip': 'Requires hot weather and sandy loam soil.'},
    'muskmelon': {'season': 'Summer / Zaid', 'harvest': 'Late Summer', 'tip': 'Susceptible to frost; needs plenty of sunshine.'},
    'apple': {'season': 'Perennial', 'harvest': 'Autumn', 'tip': 'Requires chilling hours (cold winter) for proper blooming.'},
    'orange': {'season': 'Perennial', 'harvest': 'Winter', 'tip': 'Prefers sub-tropical climate and well-drained soil.'},
    'papaya': {'season': 'Perennial', 'harvest': 'Year-round', 'tip': 'Highly sensitive to frost and waterlogging.'},
    'coconut': {'season': 'Perennial', 'harvest': 'Year-round', 'tip': 'Thrives in coastal climates with high humidity.'},
    'cotton': {'season': 'Kharif', 'harvest': 'Winter', 'tip': 'Needs plenty of sunshine and dry weather during harvest.'},
    'jute': {'season': 'Kharif', 'harvest': 'Autumn', 'tip': 'Requires hot and humid climate and plenty of water.'},
    'coffee': {'season': 'Perennial', 'harvest': 'Winter', 'tip': 'Thrives in shade at higher altitudes with distributed rainfall.'},
    'wheat': {'season': 'Rabi (Winter)', 'harvest': 'Spring', 'tip': 'Requires cool, moist weather during early growth and dry weather during ripening.'}
}

# Estimated profit mapping (mock data per acre in USD)
PROFIT_MAPPING = {
    'rice': 300,
    'maize': 250,
    'chickpea': 400,
    'kidneybeans': 450,
    'pigeonpeas': 350,
    'mothbeans': 300,
    'mungbean': 320,
    'blackgram': 310,
    'lentil': 380,
    'pomegranate': 800,
    'banana': 600,
    'mango': 700,
    'grapes': 1000,
    'watermelon': 550,
    'muskmelon': 500,
    'apple': 900,
    'orange': 650,
    'papaya': 400,
    'coconut': 750,
    'cotton': 450,
    'jute': 200,
    'coffee': 850,
    'wheat': 340
}

def generate_synthetic_data():
    """Generates a synthetic Crop_recommendation.csv dataset if it doesn't exist."""
    print("Generating synthetic dataset...")
    data = []
    crops = list(PROFIT_MAPPING.keys())
    
    for _ in range(2200): # 100 samples per 22 crops approx
        crop = random.choice(crops)
        # Random plausible values
        N = random.randint(0, 140)
        P = random.randint(5, 145)
        K = random.randint(5, 205)
        temperature = round(random.uniform(10.0, 45.0), 2)
        humidity = round(random.uniform(15.0, 99.0), 2)
        ph = round(random.uniform(3.5, 9.5), 2)
        rainfall = round(random.uniform(20.0, 298.0), 2)
        data.append([N, P, K, temperature, humidity, ph, rainfall, crop])
        
    df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])
    df.to_csv(DATASET_PATH, index=False)
    print(f"Dataset generated and saved to {DATASET_PATH}")

def train_model():
    """Trains the Random Forest Classifier on startup."""
    global MODEL, IDEAL_METRICS
    if not os.path.exists(DATASET_PATH):
        generate_synthetic_data()
        
    print("Loading dataset and training model...")
    df = pd.read_csv(DATASET_PATH)
    
    # Calculate ideal metrics (averages) for each crop
    numeric_df = df.drop(columns=['label'])
    IDEAL_METRICS = df.groupby('label')[numeric_df.columns].mean().to_dict('index')
    
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    MODEL = RandomForestClassifier(n_estimators=100, random_state=42)
    MODEL.fit(X, y)
    print("Model training complete.")

def recommend_fertilizer(user_n, user_p, user_k, ideal_n, ideal_p, ideal_k, soil_type='Loamy'):
    """Recommends fertilizer based on nutrient gaps and soil type."""
    n_gap = ideal_n - user_n
    p_gap = ideal_p - user_p
    k_gap = ideal_k - user_k
    
    deficiencies = []
    if n_gap > 10: deficiencies.append('N')
    if p_gap > 10: deficiencies.append('P')
    if k_gap > 10: deficiencies.append('K')
    
    recommendation = {'name': 'Organic Compost', 'reason': 'Soil nutrients are well-balanced.'}
    
    if len(deficiencies) == 3:
        recommendation = {'name': 'NPK 19:19:19', 'reason': 'Soil is deficient in all major nutrients (N, P, K).'}
    elif len(deficiencies) == 2:
        if 'N' in deficiencies and 'P' in deficiencies:
            recommendation = {'name': 'DAP (Diammonium Phosphate)', 'reason': 'High deficiency in Nitrogen and Phosphorus.'}
        elif 'N' in deficiencies and 'K' in deficiencies:
            recommendation = {'name': 'Potassium Nitrate', 'reason': 'Deficient in Nitrogen and Potassium.'}
        elif 'P' in deficiencies and 'K' in deficiencies:
            recommendation = {'name': 'Monopotassium Phosphate', 'reason': 'Deficient in Phosphorus and Potassium.'}
    elif len(deficiencies) == 1:
        if 'N' in deficiencies:
            recommendation = {'name': 'Urea', 'reason': 'Soil is primarily deficient in Nitrogen.'}
        elif 'P' in deficiencies:
            recommendation = {'name': 'Single Super Phosphate', 'reason': 'Soil is primarily deficient in Phosphorus.'}
        elif 'K' in deficiencies:
            recommendation = {'name': 'Muriate of Potash (MOP)', 'reason': 'Soil is primarily deficient in Potassium.'}

    # Add Soil Type specific advice
    soil_advice = (
        " For Sandy soil, ensure frequent, light watering as nutrients leach quickly." if soil_type == "Sandy" else
        " For Clay soil, avoid overwatering as it retains moisture and can cause root rot." if soil_type == "Clay" else
        " For Loamy soil, maintain standard practices as it holds nutrients well."
    )
    recommendation['reason'] += soil_advice
            
    return recommendation

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL:
        return jsonify({'error': 'Model is not trained yet.'}), 500
        
    try:
        data = request.json
        N = float(data.get('N'))
        P = float(data.get('P'))
        K = float(data.get('K'))
        temperature = float(data.get('temperature'))
        humidity = float(data.get('humidity'))
        ph = float(data.get('ph'))
        rainfall = float(data.get('rainfall'))
        soil_type = data.get('soil_type', 'Loamy')
        
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        probabilities = MODEL.predict_proba(features)[0]
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        
        top_crops = []
        for idx in top3_indices:
            crop = MODEL.classes_[idx]
            prob = probabilities[idx]
            top_crops.append({'crop': crop.capitalize(), 'probability': round(prob * 100, 1)})
            
        prediction = top_crops[0]['crop'].lower()
        
        base_profit_usd = PROFIT_MAPPING.get(prediction, 0)
        profit_inr = base_profit_usd * 80 # Convert to Rupees
        tips = CROP_TIPS.get(prediction, {'season': 'Unknown', 'harvest': 'Unknown', 'tip': 'No tips available.'})
        ideal = IDEAL_METRICS.get(prediction, {'N': 0, 'P': 0, 'K': 0})
        
        fertilizer = recommend_fertilizer(N, P, K, ideal.get('N', 0), ideal.get('P', 0), ideal.get('K', 0), soil_type)
        crop_name = prediction.capitalize()
        profit_str = f"₹{profit_inr:,} per acre"

        # Mock Market Trends for 12 months in Rupees
        market_trends = [profit_inr + random.randint(-4000, 4000) for _ in range(12)]

        # Save to database if user is logged in
        if current_user.is_authenticated:
            new_history = History(
                user_id=current_user.id,
                n=N, p=P, k=K, temp=temperature, humidity=humidity, ph=ph, rainfall=rainfall,
                crop=crop_name, fertilizer=fertilizer['name'], profit=profit_str
            )
            db.session.add(new_history)
            db.session.commit()
        
        return jsonify({
            'crop': crop_name,
            'top_crops': top_crops,
            'market_trends': market_trends,
            'profit': profit_str,
            'tips': tips,
            'ideal_metrics': ideal,
            'fertilizer': fertilizer
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- Authentication & Dashboard Routes ---

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.', 'error')
            return redirect(url_for('register'))
            
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login unsuccessful. Please check username and password.', 'error')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    histories = History.query.filter_by(user_id=current_user.id).order_by(History.date.desc()).all()
    return render_template('dashboard.html', histories=histories)

# --- API Endpoints for Streamlit Frontend ---

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
        
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 400
        
    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password_hash=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Account created successfully'})

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    user = User.query.filter_by(username=username).first()
    
    if user and check_password_hash(user.password_hash, password):
        login_user(user)
        return jsonify({'success': True, 'message': 'Logged in successfully', 'username': user.username})
    else:
        return jsonify({'error': 'Invalid username or password'}), 401

@app.route('/api/logout', methods=['POST'])
@login_required
def api_logout():
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/dashboard', methods=['GET'])
@login_required
def api_dashboard():
    histories = History.query.filter_by(user_id=current_user.id).order_by(History.date.desc()).all()
    history_data = []
    for h in histories:
        history_data.append({
            'date': h.date.strftime('%Y-%m-%d %H:%M:%S'),
            'N': h.n, 'P': h.p, 'K': h.k,
            'temperature': h.temp, 'humidity': h.humidity,
            'ph': h.ph, 'rainfall': h.rainfall,
            'crop': h.crop, 'fertilizer': h.fertilizer, 'profit': h.profit
        })
    return jsonify({'success': True, 'histories': history_data})


import urllib.request
import json

@app.route('/weather', methods=['POST'])
def get_weather():
    try:
        data = request.json
        city = data.get('city')
        if not city:
            return jsonify({'error': 'City is required'}), 400
            
        # 1. Geocode City using Open-Meteo Geocoding API
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1&language=en&format=json"
        req = urllib.request.Request(geocode_url, headers={'User-Agent': 'AgriSmart/1.0'})
        with urllib.request.urlopen(req) as response:
            geo_data = json.loads(response.read().decode())
            
        if 'results' not in geo_data or not geo_data['results']:
            return jsonify({'error': 'City not found'}), 404
            
        lat = geo_data['results'][0]['latitude']
        lon = geo_data['results'][0]['longitude']
        
        # 2. Get Weather using Open-Meteo Forecast API
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=auto"
        req = urllib.request.Request(weather_url, headers={'User-Agent': 'AgriSmart/1.0'})
        with urllib.request.urlopen(req) as response:
            weather_data = json.loads(response.read().decode())
            
        current = weather_data.get('current', {})
        # Note: Open-Meteo gives current precipitation, we might want to scale it to simulate annual rainfall or just provide current rain
        # For realistic testing, if rain is 0, we'll provide a standard proxy rainfall for the city based on its latitude or just a random reasonable value since weather API only gives current rain
        temp = current.get('temperature_2m', 25)
        humidity = current.get('relative_humidity_2m', 60)
        rain = current.get('precipitation', 0)
        
        # Simulate an annual rainfall proxy based on humidity so the ML model doesn't just get 0mm
        simulated_rainfall = rain if rain > 10 else humidity * 2.5 
        
        return jsonify({
            'temperature': temp,
            'humidity': humidity,
            'rainfall': round(simulated_rainfall, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/weather_coords', methods=['POST'])
def get_weather_coords():
    try:
        data = request.json
        lat = data.get('lat')
        lon = data.get('lon')
        if not lat or not lon:
            return jsonify({'error': 'Latitude and longitude required'}), 400
            
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=auto"
        req = urllib.request.Request(weather_url, headers={'User-Agent': 'AgriSmart/1.0'})
        with urllib.request.urlopen(req) as response:
            weather_data = json.loads(response.read().decode())
            
        current = weather_data.get('current', {})
        temp = current.get('temperature_2m', 25)
        humidity = current.get('relative_humidity_2m', 60)
        rain = current.get('precipitation', 0)
        
        simulated_rainfall = rain if rain > 10 else humidity * 2.5 
        
        return jsonify({
            'temperature': temp,
            'humidity': humidity,
            'rainfall': round(simulated_rainfall, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

import urllib.parse

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    train_model()
    # Adding debug mode might restart app twice
    app.run(debug=True, port=5000)
