import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os
import random
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.ensemble import RandomForestClassifier
import urllib.request
import json

# --- DATABASE SETUP ---
DB_PATH = "agri_streamlit.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  date TEXT NOT NULL,
                  n REAL, p REAL, k REAL, temp REAL, humidity REAL, ph REAL, rainfall REAL,
                  crop TEXT, fertilizer TEXT, profit TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

init_db()

def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username, password_hash FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def register_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                  (username, generate_password_hash(password)))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    finally:
        conn.close()
    return success

def add_history(user_id, n, p, k, temp, humidity, ph, rainfall, crop, fertilizer, profit):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("""INSERT INTO history 
                 (user_id, date, n, p, k, temp, humidity, ph, rainfall, crop, fertilizer, profit) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (user_id, date_str, n, p, k, temp, humidity, ph, rainfall, crop, fertilizer, profit))
    conn.commit()
    conn.close()

def get_history(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""SELECT date, crop, profit, fertilizer, n, p, k, temp, humidity, ph, rainfall 
                 FROM history WHERE user_id=? ORDER BY id DESC""", (user_id,))
    rows = c.fetchall()
    columns = [desc[0] for desc in c.description]
    conn.close()
    return [dict(zip(columns, row)) for row in rows]

# --- ML MODEL & CONSTANTS ---
DATASET_PATH = 'Crop_recommendation.csv'

PROFIT_MAPPING = {
    'rice': 300, 'maize': 250, 'chickpea': 400, 'kidneybeans': 450,
    'pigeonpeas': 350, 'mothbeans': 300, 'mungbean': 320, 'blackgram': 310,
    'lentil': 380, 'pomegranate': 800, 'banana': 600, 'mango': 700,
    'grapes': 1000, 'watermelon': 550, 'muskmelon': 500, 'apple': 900,
    'orange': 650, 'papaya': 400, 'coconut': 750, 'cotton': 450,
    'jute': 200, 'coffee': 850, 'wheat': 340
}

CROP_TIPS = {
    'rice': {'season': 'Kharif', 'harvest': 'Winter', 'tip': 'Requires high water availability.'},
    'maize': {'season': 'Kharif', 'harvest': 'Autumn', 'tip': 'Needs well-drained soil.'},
    'chickpea': {'season': 'Rabi', 'harvest': 'Spring', 'tip': 'Sensitive to excess water.'},
    'kidneybeans': {'season': 'Kharif', 'harvest': 'Autumn', 'tip': 'Prefers loamy soil.'},
    'pigeonpeas': {'season': 'Kharif', 'harvest': 'Winter/Spring', 'tip': 'Drought-tolerant.'},
    'mothbeans': {'season': 'Kharif', 'harvest': 'Autumn', 'tip': 'Extremely drought resistant.'},
    'mungbean': {'season': 'Kharif / Zaid', 'harvest': 'Varies', 'tip': 'Short duration crop.'},
    'blackgram': {'season': 'Kharif', 'harvest': 'Autumn', 'tip': 'Requires warm weather.'},
    'lentil': {'season': 'Rabi', 'harvest': 'Spring', 'tip': 'Prefers cool climate.'},
    'pomegranate': {'season': 'Perennial', 'harvest': 'Late Summer', 'tip': 'Requires hot dry climate.'},
    'banana': {'season': 'Perennial', 'harvest': 'Year-round', 'tip': 'Requires heavy rainfall.'},
    'mango': {'season': 'Perennial', 'harvest': 'Summer', 'tip': 'Needs distinct dry season.'},
    'grapes': {'season': 'Perennial', 'harvest': 'Summer/Autumn', 'tip': 'Requires pruning.'},
    'watermelon': {'season': 'Summer', 'harvest': 'Late Summer', 'tip': 'Hot weather, sandy loam.'},
    'muskmelon': {'season': 'Summer', 'harvest': 'Late Summer', 'tip': 'Plenty of sunshine.'},
    'apple': {'season': 'Perennial', 'harvest': 'Autumn', 'tip': 'Requires chilling hours.'},
    'orange': {'season': 'Perennial', 'harvest': 'Winter', 'tip': 'Sub-tropical climate.'},
    'papaya': {'season': 'Perennial', 'harvest': 'Year-round', 'tip': 'Sensitive to frost.'},
    'coconut': {'season': 'Perennial', 'harvest': 'Year-round', 'tip': 'Coastal climates.'},
    'cotton': {'season': 'Kharif', 'harvest': 'Winter', 'tip': 'Plenty of sunshine.'},
    'jute': {'season': 'Kharif', 'harvest': 'Autumn', 'tip': 'Hot and humid climate.'},
    'coffee': {'season': 'Perennial', 'harvest': 'Winter', 'tip': 'Thrives in shade.'},
    'wheat': {'season': 'Rabi', 'harvest': 'Spring', 'tip': 'Cool, moist early growth.'}
}

@st.cache_resource
def load_and_train_model():
    if not os.path.exists(DATASET_PATH):
        # Generate synthetic data if missing
        data = []
        crops = list(PROFIT_MAPPING.keys())
        for _ in range(2200):
            crop = random.choice(crops)
            N = random.randint(0, 140)
            P = random.randint(5, 145)
            K = random.randint(5, 205)
            temp = round(random.uniform(10.0, 45.0), 2)
            hum = round(random.uniform(15.0, 99.0), 2)
            ph = round(random.uniform(3.5, 9.5), 2)
            rain = round(random.uniform(20.0, 298.0), 2)
            data.append([N, P, K, temp, hum, ph, rain, crop])
        df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])
        df.to_csv(DATASET_PATH, index=False)
    else:
        df = pd.read_csv(DATASET_PATH)

    numeric_df = df.drop(columns=['label'])
    ideal_metrics = df.groupby('label')[numeric_df.columns].mean().to_dict('index')

    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, ideal_metrics

MODEL, IDEAL_METRICS = load_and_train_model()

def recommend_fertilizer(user_n, user_p, user_k, ideal_n, ideal_p, ideal_k, soil_type='Loamy'):
    n_gap = ideal_n - user_n
    p_gap = ideal_p - user_p
    k_gap = ideal_k - user_k
    
    deficiencies = []
    if n_gap > 10: deficiencies.append('N')
    if p_gap > 10: deficiencies.append('P')
    if k_gap > 10: deficiencies.append('K')
    
    rec = {'name': 'Organic Compost', 'reason': 'Soil nutrients are well-balanced.'}
    if len(deficiencies) == 3:
        rec = {'name': 'NPK 19:19:19', 'reason': 'Soil is deficient in all major nutrients.'}
    elif len(deficiencies) == 2:
        if 'N' in deficiencies and 'P' in deficiencies:
            rec = {'name': 'DAP', 'reason': 'High deficiency in Nitrogen and Phosphorus.'}
        elif 'N' in deficiencies and 'K' in deficiencies:
            rec = {'name': 'Potassium Nitrate', 'reason': 'Deficient in Nitrogen and Potassium.'}
        elif 'P' in deficiencies and 'K' in deficiencies:
            rec = {'name': 'Monopotassium Phosphate', 'reason': 'Deficient in Phosphorus and Potassium.'}
    elif len(deficiencies) == 1:
        if 'N' in deficiencies:
            rec = {'name': 'Urea', 'reason': 'Soil is primarily deficient in Nitrogen.'}
        elif 'P' in deficiencies:
            rec = {'name': 'Single Super Phosphate', 'reason': 'Soil is primarily deficient in Phosphorus.'}
        elif 'K' in deficiencies:
            rec = {'name': 'Muriate of Potash', 'reason': 'Soil is primarily deficient in Potassium.'}

    soil_advice = (
        " For Sandy soil, ensure frequent, light watering." if soil_type == "Sandy" else
        " For Clay soil, avoid overwatering." if soil_type == "Clay" else
        " For Loamy soil, maintain standard practices."
    )
    rec['reason'] += soil_advice
    return rec

def fetch_weather(city):
    try:
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1&language=en&format=json"
        req = urllib.request.Request(geocode_url, headers={'User-Agent': 'AgriSmart/1.0'})
        with urllib.request.urlopen(req) as response:
            geo_data = json.loads(response.read().decode())
        
        if 'results' not in geo_data or not geo_data['results']:
            return None, "City not found"
            
        lat = geo_data['results'][0]['latitude']
        lon = geo_data['results'][0]['longitude']
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=auto"
        req = urllib.request.Request(weather_url, headers={'User-Agent': 'AgriSmart/1.0'})
        with urllib.request.urlopen(req) as response:
            weather_data = json.loads(response.read().decode())
            
        current = weather_data.get('current', {})
        temp = current.get('temperature_2m', 25)
        humidity = current.get('relative_humidity_2m', 60)
        rain = current.get('precipitation', 0)
        
        simulated_rainfall = rain if rain > 10 else humidity * 2.5 
        return {'temperature': temp, 'humidity': humidity, 'rainfall': round(simulated_rainfall, 2)}, None
    except Exception as e:
        return None, str(e)

# --- INITIALIZE SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# --- STYLING ---
st.set_page_config(page_title="AgriSmart - Crop Recommendation", page_icon="🌱", layout="wide")

bg_color = "#f4f9f4"
text_color = "#333"
primary_color = "#2c5f2d"
btn_color = "#97bc62"
success_box = "linear-gradient(135deg, #e0f2e9 0%, #c4e0c4 100%)"

st.markdown(f"""
<style>
    .stApp {{ background-color: {bg_color}; }}
    .main {{ background-color: {bg_color}; }}
    h1, h2, h3, h4, p, label, .stMarkdown {{ color: {text_color} !important; font-family: 'Inter', sans-serif; }}
    h1, h2, h3 {{ color: {primary_color} !important; }}
    .stButton>button {{ background-color: {btn_color}; color: white !important; border-radius: 10px; border: none; padding: 10px 24px; transition: all 0.3s ease; }}
    .stButton>button:hover {{ background-color: {primary_color}; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }}
    .success-box {{ padding: 20px; border-radius: 10px; background: {success_box}; border-left: 5px solid {primary_color}; margin: 20px 0; }}
    .success-box h2, .success-box h4 {{ color: #111 !important; }}
</style>
""", unsafe_allow_html=True)

# --- NAVIGATION ---
st.sidebar.markdown("---")
st.sidebar.title("🌱 AgriSmart")
if st.session_state.logged_in:
    st.sidebar.write(f"Welcome, **{st.session_state.username}**!")
    page = st.sidebar.radio("Menu", ["Home & Predict", "Yield & ROI Calculator", "Agri-News & Tips", "Dashboard"])
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_id = None
        st.rerun()
else:
    page = st.sidebar.radio("Menu", ["Home & Predict", "Yield & ROI Calculator", "Agri-News & Tips", "Login", "Register"])

# --- PAGES ---
if page == "Home & Predict":
    st.title("Intelligent Crop & Fertilizer Recommendation")
    st.write("Enter your soil metrics and weather data below to get AI-powered recommendations.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Soil Nutrients")
        soil_type = st.selectbox("Soil Type", ["Loamy", "Sandy", "Clay"])
        n = st.number_input("Nitrogen (N)", min_value=0, max_value=150, value=50)
        p = st.number_input("Phosphorus (P)", min_value=0, max_value=150, value=50)
        k = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

    with col2:
        st.subheader("Weather Details")
        city = st.text_input("City Name (Auto-fill weather)", placeholder="e.g., London")
        if st.button("Get Weather Data"):
            if city:
                with st.spinner("Fetching weather data..."):
                    data, err = fetch_weather(city)
                    if data:
                        st.session_state.temp = data['temperature']
                        st.session_state.humidity = data['humidity']
                        st.session_state.rainfall = data['rainfall']
                        st.success(f"Weather fetched for {city}!")
                    else:
                        st.error(f"Could not fetch weather: {err}")
            else:
                st.warning("Please enter a city name.")

        temp_val = st.session_state.get('temp', 25.0)
        hum_val = st.session_state.get('humidity', 60.0)
        rain_val = st.session_state.get('rainfall', 100.0)

        temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=float(temp_val), step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=float(hum_val), step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=float(rain_val), step=1.0)

    st.markdown("---")
    if st.button("Predict Optimal Crop", use_container_width=True):
        with st.spinner("Analyzing soil and weather data..."):
            features = np.array([[n, p, k, temp, humidity, ph, rainfall]])
            probabilities = MODEL.predict_proba(features)[0]
            top3_indices = np.argsort(probabilities)[-3:][::-1]
            
            top_crops = []
            for idx in top3_indices:
                crop = MODEL.classes_[idx]
                prob = probabilities[idx]
                top_crops.append({'crop': crop.capitalize(), 'probability': round(prob * 100, 1)})
                
            prediction = top_crops[0]['crop'].lower()
            base_profit_usd = PROFIT_MAPPING.get(prediction, 0)
            profit_inr = base_profit_usd * 80
            profit_str = f"₹{profit_inr:,} per acre"
            
            tips = CROP_TIPS.get(prediction, {'season': 'Unknown', 'harvest': 'Unknown', 'tip': 'No tips.'})
            ideal = IDEAL_METRICS.get(prediction, {'N': 0, 'P': 0, 'K': 0})
            fertilizer = recommend_fertilizer(n, p, k, ideal.get('N', 0), ideal.get('P', 0), ideal.get('K', 0), soil_type)
            market_trends = [profit_inr + random.randint(-4000, 4000) for _ in range(12)]
            
            # Save to database if logged in
            if st.session_state.logged_in:
                add_history(st.session_state.user_id, n, p, k, temp, humidity, ph, rainfall, prediction.capitalize(), fertilizer['name'], profit_str)

            st.markdown(f"""
            <div class="success-box">
                <h2>🌾 Top Recommendation: {prediction.capitalize()}</h2>
                <h4>💰 Estimated Profit: {profit_str}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("📈 Alternative Crops")
            alt_cols = st.columns(len(top_crops) - 1)
            for i, c in enumerate(top_crops[1:]):
                with alt_cols[i]:
                    st.write(f"**{c['crop']}**")
                    st.progress(c['probability'] / 100.0)
                    st.caption(f"{c['probability']}% Match")

            st.subheader("💡 Growing Tips")
            st.write(f"**Season:** {tips['season']}")
            st.write(f"**Harvest:** {tips['harvest']}")
            st.info(tips['tip'])

            st.subheader("🧪 Fertilizer Recommendation")
            st.write(f"**Recommended:** {fertilizer['name']}")
            st.write(f"**Reason:** {fertilizer['reason']}")
            
            if not st.session_state.logged_in:
                st.warning("Log in to save your prediction history!")
            
            st.markdown("---")
            st.subheader("📊 Nutrient Analysis: Your Soil vs. Ideal")
            fig, ax = plt.subplots(figsize=(8, 4))
            categories = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
            user_values = [n, p, k]
            ideal_values = [ideal.get('N', 0), ideal.get('P', 0), ideal.get('K', 0)]
            x = range(len(categories))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], user_values, width, label='Your Soil', color='#97bc62')
            ax.bar([i + width/2 for i in x], ideal_values, width, label='Ideal for ' + prediction.capitalize(), color='#2c5f2d')
            ax.set_ylabel('Nutrient Value')
            ax.set_title('Soil Nutrients Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            st.pyplot(fig)
            
            st.markdown("---")
            st.subheader(f"💲 12-Month Price Trend ({prediction.capitalize()})")
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            ax2.plot(months, market_trends, marker='o', color='#2c5f2d', linewidth=2)
            ax2.set_ylabel("Price (₹/acre)")
            ax2.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig2)
            
            # PDF Generation
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            fig.savefig(tmp_file.name, bbox_inches='tight')
            tmp_file.close()
            tmp_file2 = tempfile.NamedTemporaryFile(delete=False, suffix='_trend.png')
            fig2.savefig(tmp_file2.name, bbox_inches='tight')
            tmp_file2.close()

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            pdf.set_font("Helvetica", style="B", size=16)
            pdf.cell(200, 10, txt="AgriSmart Crop Recommendation Report", ln=True, align="C")
            pdf.set_font("Helvetica", size=10)
            pdf.cell(200, 10, txt=f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
            pdf.ln(5)
            
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.cell(200, 10, txt="[INPUT METRICS]", ln=True)
            pdf.set_font("Helvetica", size=12)
            pdf.cell(200, 8, txt=f"Nitrogen (N): {n}  |  Phosphorus (P): {p}  |  Potassium (K): {k}", ln=True)
            pdf.cell(200, 8, txt=f"pH Level: {ph}  |  Temperature: {temp} C  |  Humidity: {humidity}%  |  Rainfall: {rainfall}mm", ln=True)
            pdf.ln(5)
            
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.cell(200, 10, txt="[RECOMMENDATION]", ln=True)
            pdf.set_font("Helvetica", size=12)
            pdf.cell(200, 8, txt=f"Crop: {prediction.capitalize()}", ln=True)
            safe_profit = str(profit_str).replace("₹", "Rs. ")
            pdf.cell(200, 8, txt=f"Estimated Profit: {safe_profit}", ln=True)
            pdf.ln(5)
            
            pdf.image(tmp_file.name, x=10, w=190)
            pdf.ln(5)
            
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.cell(200, 10, txt="[GROWING TIPS]", ln=True)
            pdf.set_font("Helvetica", size=12)
            pdf.cell(200, 8, txt=f"Season: {tips['season']}  |  Harvest: {tips['harvest']}", ln=True)
            pdf.multi_cell(0, 8, txt=f"Tip: {tips['tip']}")
            pdf.ln(5)
            
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.cell(200, 10, txt="[FERTILIZER RECOMMENDATION]", ln=True)
            pdf.set_font("Helvetica", size=12)
            pdf.cell(200, 8, txt=f"Recommended: {fertilizer['name']}", ln=True)
            pdf.multi_cell(0, 8, txt=f"Reason: {fertilizer['reason']}")
            
            pdf.add_page()
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.cell(200, 10, txt="[12-MONTH PRICE TRENDS]", ln=True)
            pdf.image(tmp_file2.name, x=10, w=190)
            
            pdf_bytes = pdf.output()
            os.unlink(tmp_file.name)
            os.unlink(tmp_file2.name)
            
            st.markdown("---")
            st.subheader("📄 Download Report")
            st.download_button(label="⬇️ Download Recommendation Report (.pdf)", data=bytes(pdf_bytes), file_name=f"AgriSmart_Report_{prediction.capitalize()}.pdf", mime="application/pdf", use_container_width=True)

elif page == "Login":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center; color: #2c5f2d;'>👋 Welcome Back to AgriSmart</h2>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("👤 Username")
            password = st.text_input("🔒 Password", type="password")
            submitted = st.form_submit_button("Log In", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    user = get_user(username)
                    if user and check_password_hash(user[2], password):
                        st.session_state.logged_in = True
                        st.session_state.username = user[1]
                        st.session_state.user_id = user[0]
                        st.success("Logged in successfully! Redirecting...")
                        st.rerun()
                    else:
                        st.error('Invalid username or password.')

elif page == "Register":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center; color: #2c5f2d;'>🌱 Create an Account</h2>", unsafe_allow_html=True)
        with st.form("register_form"):
            username = st.text_input("👤 Choose a Username")
            password = st.text_input("🔒 Choose a Password", type="password")
            submitted = st.form_submit_button("Register Account", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    if register_user(username, password):
                        st.success("✨ Account created successfully! Please go to the Login page.")
                    else:
                        st.error("Username already exists.")

elif page == "Dashboard" and st.session_state.logged_in:
    st.title("Your Prediction History")
    with st.spinner("Loading history..."):
        histories = get_history(st.session_state.user_id)
        if histories:
            df = pd.DataFrame(histories)
            df = df[['date', 'crop', 'profit', 'fertilizer', 'n', 'p', 'k', 'temp', 'humidity', 'ph', 'rainfall']]
            st.dataframe(df, use_container_width=True)
            st.markdown("---")
            st.download_button(label="⬇️ Export History to CSV", data=df.to_csv(index=False), file_name="AgriSmart_Prediction_History.csv", mime="text/csv")
        else:
            st.info("No prediction history found. Try making a prediction on the Home page!")

elif page == "Yield & ROI Calculator":
    st.title("Interactive Yield & ROI Calculator")
    st.write("Calculate estimated costs and profits for your farm.")
    crops = ["Rice", "Maize", "Chickpea", "Kidneybeans", "Pigeonpeas", "Mothbeans", "Mungbean", "Blackgram", "Lentil", "Pomegranate", "Banana", "Mango", "Grapes", "Watermelon", "Muskmelon", "Apple", "Orange", "Papaya", "Coconut", "Cotton", "Jute", "Coffee", "Wheat"]
    col1, col2 = st.columns(2)
    with col1: selected_crop = st.selectbox("Select Crop", sorted(crops))
    with col2: land_size = st.number_input("Land Size (Acres)", min_value=0.1, value=1.0, step=0.1)
        
    if st.button("Calculate ROI", use_container_width=True):
        base_profit_map = {
            'Rice': 24000, 'Maize': 20000, 'Chickpea': 32000, 'Kidneybeans': 36000,
            'Pigeonpeas': 28000, 'Mothbeans': 24000, 'Mungbean': 25600, 'Blackgram': 24800,
            'Lentil': 30400, 'Pomegranate': 64000, 'Banana': 48000, 'Mango': 56000,
            'Grapes': 80000, 'Watermelon': 44000, 'Muskmelon': 40000, 'Apple': 72000,
            'Orange': 52000, 'Papaya': 32000, 'Coconut': 60000, 'Cotton': 36000,
            'Jute': 16000, 'Coffee': 68000, 'Wheat': 27200
        }
        net_profit_per_acre = base_profit_map.get(selected_crop, 25000)
        total_revenue = net_profit_per_acre * land_size / 0.4
        net_profit = net_profit_per_acre * land_size
        expenses = total_revenue - net_profit
        seed_cost = expenses * 0.15
        fertilizer_cost = expenses * 0.30
        labor_cost = expenses * 0.40
        misc_cost = expenses * 0.15
        
        st.markdown("---")
        st.subheader("💰 Financial Breakdown")
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Revenue", f"₹{total_revenue:,.2f}")
        c2.metric("Total Expenses", f"₹{expenses:,.2f}")
        c3.metric("Net Profit", f"₹{net_profit:,.2f}")
        
        st.subheader("📊 Expense vs Profit Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = ['Seed Cost', 'Fertilizer Cost', 'Labor Cost', 'Misc. Expenses', 'Net Profit']
        sizes = [seed_cost, fertilizer_cost, labor_cost, misc_cost, net_profit]
        colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#2c5f2d']
        explode = (0, 0, 0, 0, 0.1)
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 10})
        ax.axis('equal')
        fig.patch.set_alpha(0.0)
        ax.set_facecolor((0, 0, 0, 0))
        st.pyplot(fig)

elif page == "Agri-News & Tips":
    st.title("📰 Agricultural News & Daily Tips")
    st.write("Stay updated with the latest trends and best practices in farming.")
    st.markdown("---")
    st.subheader("💡 Tip of the Day")
    st.info("**Water Conservation:** Consider using drip irrigation for row crops. It can reduce water usage by up to 50% compared to traditional surface irrigation, while also delivering nutrients directly to the root zone.")
    st.markdown("---")
    st.subheader("Latest News")
    news_items = [
        {"title": "Government announces new subsidies for organic farming", "date": "May 1, 2026", "content": "In a bid to promote sustainable agriculture, the agriculture ministry has rolled out a new subsidy scheme covering up to 40% of the cost for organic fertilizers and biopesticides."},
        {"title": "Monsoon predicted to be above average this year", "date": "April 28, 2026", "content": "The meteorological department forecasts an above-average monsoon, bringing relief to farmers in rain-fed agricultural zones. Preparation for the Kharif season is in full swing."},
        {"title": "New drought-resistant wheat variety launched", "date": "April 25, 2026", "content": "Researchers have successfully developed a new wheat strain that yields 15% more in arid conditions. Seeds will be available to farmers starting next Rabi season."},
        {"title": "Market Trends: Rising demand for exotic fruits", "date": "April 20, 2026", "content": "Market analysis shows a 30% increase in urban demand for crops like Dragon Fruit and Avocado. Farmers are encouraged to diversify their orchards to capture this premium market."}
    ]
    for item in news_items:
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; border-left: 5px solid #2c5f2d; transition: transform 0.2s;">
            <h4 style="margin-top: 0; color: #2c5f2d;">{item['title']}</h4>
            <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">📅 {item['date']}</p>
            <p style="color: #333; margin-bottom: 0;">{item['content']}</p>
        </div>
        """, unsafe_allow_html=True)
