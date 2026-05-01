import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os

# Constants
API_URL = "http://127.0.0.1:5000"

# Initialize Session State
if 'api_session' not in st.session_state:
    st.session_state.api_session = requests.Session()
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# Styling (Rich Aesthetic)
st.set_page_config(page_title="AgriSmart - Crop Recommendation", page_icon="🌱", layout="wide")

bg_color = "#f4f9f4"
text_color = "#333"
primary_color = "#2c5f2d"
btn_color = "#97bc62"
success_box = "linear-gradient(135deg, #e0f2e9 0%, #c4e0c4 100%)"

st.markdown(f"""
<style>
    /* Custom Styling for Premium Look */
    .stApp {{
        background-color: {bg_color};
    }}
    .main {{
        background-color: {bg_color};
    }}
    h1, h2, h3, h4, p, label, .stMarkdown {{
        color: {text_color} !important;
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3 {{
        color: {primary_color} !important;
    }}
    .stButton>button {{
        background-color: {btn_color};
        color: white !important;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {primary_color};
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    .success-box {{
        padding: 20px;
        border-radius: 10px;
        background: {success_box};
        border-left: 5px solid {primary_color};
        margin: 20px 0;
    }}
    .success-box h2, .success-box h4 {{
        color: #111 !important; /* Always keep box text dark for readability against bright gradients */
    }}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.title("🌱 AgriSmart")
if st.session_state.logged_in:
    st.sidebar.write(f"Welcome, **{st.session_state.username}**!")
    page = st.sidebar.radio("Menu", ["Home & Predict", "Yield & ROI Calculator", "Agri-News & Tips", "Dashboard"])
    if st.sidebar.button("Logout"):
        resp = st.session_state.api_session.post(f"{API_URL}/api/logout")
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()
else:
    page = st.sidebar.radio("Menu", ["Home & Predict", "Yield & ROI Calculator", "Agri-News & Tips", "Login", "Register"])

# --- Home & Predict Page ---
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
                    resp = requests.post(f"{API_URL}/weather", json={"city": city})
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.temp = data['temperature']
                        st.session_state.humidity = data['humidity']
                        st.session_state.rainfall = data['rainfall']
                        st.success(f"Weather fetched for {city}!")
                    else:
                        st.error("Could not fetch weather data. Check city name or API server.")
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
        payload = {
            "N": n, "P": p, "K": k, "temperature": temp, 
            "humidity": humidity, "ph": ph, "rainfall": rainfall,
            "soil_type": soil_type
        }
        with st.spinner("Analyzing soil and weather data..."):
            try:
                resp = st.session_state.api_session.post(f"{API_URL}/predict", json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    st.markdown(f"""
                    <div class="success-box">
                        <h2>🌾 Top Recommendation: {result['crop']}</h2>
                        <h4>💰 Estimated Profit: {result['profit']}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader("📈 Alternative Crops")
                    alt_cols = st.columns(len(result['top_crops']) - 1)
                    for i, c in enumerate(result['top_crops'][1:]):
                        with alt_cols[i]:
                            st.write(f"**{c['crop']}**")
                            st.progress(c['probability'] / 100.0)
                            st.caption(f"{c['probability']}% Match")

                    st.subheader("💡 Growing Tips")
                    st.write(f"**Season:** {result['tips']['season']}")
                    st.write(f"**Harvest:** {result['tips']['harvest']}")
                    st.info(result['tips']['tip'])

                    st.subheader("🧪 Fertilizer Recommendation")
                    st.write(f"**Recommended:** {result['fertilizer']['name']}")
                    st.write(f"**Reason:** {result['fertilizer']['reason']}")
                    
                    if not st.session_state.logged_in:
                        st.warning("Log in to save your prediction history!")
                    
                    # --- VISUALIZATION: Input vs Ideal Metrics ---
                    st.markdown("---")
                    st.subheader("📊 Nutrient Analysis: Your Soil vs. Ideal")
                    ideal = result['ideal_metrics']
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    categories = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
                    user_values = [n, p, k]
                    ideal_values = [ideal.get('N', 0), ideal.get('P', 0), ideal.get('K', 0)]
                    
                    x = range(len(categories))
                    width = 0.35
                    
                    ax.bar([i - width/2 for i in x], user_values, width, label='Your Soil', color='#97bc62')
                    ax.bar([i + width/2 for i in x], ideal_values, width, label='Ideal for ' + result['crop'], color='#2c5f2d')
                    
                    ax.set_ylabel('Nutrient Value')
                    ax.set_title('Soil Nutrients Comparison')
                    ax.set_xticks(x)
                    ax.set_xticklabels(categories)
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # --- VISUALIZATION: Market Trends ---
                    st.markdown("---")
                    st.subheader(f"💲 12-Month Price Trend ({result['crop']})")
                    fig2, ax2 = plt.subplots(figsize=(8, 3))
                    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    ax2.plot(months, result['market_trends'], marker='o', color='#2c5f2d', linewidth=2)
                    ax2.set_ylabel("Price (₹/acre)")
                    ax2.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig2)
                    
                    # Save plots to temp files for PDF
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    fig.savefig(tmp_file.name, bbox_inches='tight')
                    tmp_file.close()
                    
                    tmp_file2 = tempfile.NamedTemporaryFile(delete=False, suffix='_trend.png')
                    fig2.savefig(tmp_file2.name, bbox_inches='tight')
                    tmp_file2.close()

                    # --- GENERATE PDF REPORT ---
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Helvetica", size=12)
                    
                    # Header
                    pdf.set_font("Helvetica", style="B", size=16)
                    pdf.cell(200, 10, txt="AgriSmart Crop Recommendation Report", ln=True, align="C")
                    pdf.set_font("Helvetica", size=10)
                    pdf.cell(200, 10, txt=f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
                    pdf.ln(5)
                    
                    # Metrics
                    pdf.set_font("Helvetica", style="B", size=12)
                    pdf.cell(200, 10, txt="[INPUT METRICS]", ln=True)
                    pdf.set_font("Helvetica", size=12)
                    pdf.cell(200, 8, txt=f"Nitrogen (N): {n}  |  Phosphorus (P): {p}  |  Potassium (K): {k}", ln=True)
                    pdf.cell(200, 8, txt=f"pH Level: {ph}  |  Temperature: {temp} C  |  Humidity: {humidity}%  |  Rainfall: {rainfall}mm", ln=True)
                    pdf.ln(5)
                    
                    # Recommendation
                    pdf.set_font("Helvetica", style="B", size=12)
                    pdf.cell(200, 10, txt="[RECOMMENDATION]", ln=True)
                    pdf.set_font("Helvetica", size=12)
                    pdf.cell(200, 8, txt=f"Crop: {result['crop']}", ln=True)
                    
                    # Replace ₹ with Rs. for PDF compatibility
                    safe_profit = str(result['profit']).replace("₹", "Rs. ")
                    pdf.cell(200, 8, txt=f"Estimated Profit: {safe_profit}", ln=True)
                    pdf.ln(5)
                    
                    # Chart
                    pdf.image(tmp_file.name, x=10, w=190)
                    pdf.ln(5)
                    
                    # Tips & Fertilizer
                    pdf.set_font("Helvetica", style="B", size=12)
                    pdf.cell(200, 10, txt="[GROWING TIPS]", ln=True)
                    pdf.set_font("Helvetica", size=12)
                    pdf.cell(200, 8, txt=f"Season: {result['tips']['season']}  |  Harvest: {result['tips']['harvest']}", ln=True)
                    pdf.multi_cell(0, 8, txt=f"Tip: {result['tips']['tip']}")
                    pdf.ln(5)
                    
                    pdf.set_font("Helvetica", style="B", size=12)
                    pdf.cell(200, 10, txt="[FERTILIZER RECOMMENDATION]", ln=True)
                    pdf.set_font("Helvetica", size=12)
                    pdf.cell(200, 8, txt=f"Recommended: {result['fertilizer']['name']}", ln=True)
                    pdf.multi_cell(0, 8, txt=f"Reason: {result['fertilizer']['reason']}")
                    
                    pdf.add_page()
                    # Chart 2: Trends
                    pdf.set_font("Helvetica", style="B", size=12)
                    pdf.cell(200, 10, txt="[12-MONTH PRICE TRENDS]", ln=True)
                    pdf.image(tmp_file2.name, x=10, w=190)
                    
                    pdf_bytes = pdf.output()
                    
                    # Clean up temp file
                    os.unlink(tmp_file.name)
                    os.unlink(tmp_file2.name)
                    
                    st.markdown("---")
                    st.subheader("📄 Download Report")
                    st.download_button(
                        label="⬇️ Download Recommendation Report (.pdf)",
                        data=bytes(pdf_bytes),
                        file_name=f"AgriSmart_Report_{result['crop']}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                else:
                    st.error(f"Error: {resp.json().get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"Could not connect to backend API: {e}")

# --- Login Page ---
elif page == "Login":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center; color: #2c5f2d;'>👋 Welcome Back to AgriSmart</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #555;'>Sign in to access your dashboard and prediction history.</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("👤 Username")
            password = st.text_input("🔒 Password", type="password")
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Log In", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    resp = st.session_state.api_session.post(f"{API_URL}/api/login", json={"username": username, "password": password})
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.logged_in = True
                        st.session_state.username = data['username']
                        st.success("Logged in successfully! Redirecting...")
                        st.rerun()
                    else:
                        st.error(resp.json().get('error', 'Login failed.'))

# --- Register Page ---
elif page == "Register":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center; color: #2c5f2d;'>🌱 Create an Account</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #555;'>Join AgriSmart to save your intelligent crop recommendations.</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.form("register_form"):
            username = st.text_input("👤 Choose a Username")
            password = st.text_input("🔒 Choose a Password", type="password")
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Register Account", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    resp = requests.post(f"{API_URL}/api/register", json={"username": username, "password": password})
                    if resp.status_code == 200:
                        st.success("✨ Account created successfully! Please go to the Login page.")
                    else:
                        st.error(resp.json().get('error', 'Registration failed.'))

# --- Dashboard Page ---
elif page == "Dashboard" and st.session_state.logged_in:
    st.title("Your Prediction History")
    
    with st.spinner("Loading history..."):
        resp = st.session_state.api_session.get(f"{API_URL}/api/dashboard")
        if resp.status_code == 200:
            histories = resp.json().get('histories', [])
            if histories:
                df = pd.DataFrame(histories)
                # Reorder columns for better presentation
                df = df[['date', 'crop', 'profit', 'fertilizer', 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
                st.dataframe(df, use_container_width=True)
                
                st.markdown("---")
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Export History to CSV",
                    data=csv,
                    file_name="AgriSmart_Prediction_History.csv",
                    mime="text/csv"
                )
            else:
                st.info("No prediction history found. Try making a prediction on the Home page!")
        else:
            st.error("Failed to load history. Ensure backend is running.")

# --- Yield & ROI Calculator ---
elif page == "Yield & ROI Calculator":
    st.title("Interactive Yield & ROI Calculator")
    st.write("Calculate estimated costs and profits for your farm.")
    
    crops = ["Rice", "Maize", "Chickpea", "Kidneybeans", "Pigeonpeas", "Mothbeans", "Mungbean", "Blackgram", "Lentil", "Pomegranate", "Banana", "Mango", "Grapes", "Watermelon", "Muskmelon", "Apple", "Orange", "Papaya", "Coconut", "Cotton", "Jute", "Coffee", "Wheat"]
    
    col1, col2 = st.columns(2)
    with col1:
        selected_crop = st.selectbox("Select Crop", sorted(crops))
    with col2:
        land_size = st.number_input("Land Size (Acres)", min_value=0.1, value=1.0, step=0.1)
        
    if st.button("Calculate ROI", use_container_width=True):
        # Base profit mapping in rupees per acre
        base_profit_map = {
            'Rice': 24000, 'Maize': 20000, 'Chickpea': 32000, 'Kidneybeans': 36000,
            'Pigeonpeas': 28000, 'Mothbeans': 24000, 'Mungbean': 25600, 'Blackgram': 24800,
            'Lentil': 30400, 'Pomegranate': 64000, 'Banana': 48000, 'Mango': 56000,
            'Grapes': 80000, 'Watermelon': 44000, 'Muskmelon': 40000, 'Apple': 72000,
            'Orange': 52000, 'Papaya': 32000, 'Coconut': 60000, 'Cotton': 36000,
            'Jute': 16000, 'Coffee': 68000, 'Wheat': 27200
        }
        
        net_profit_per_acre = base_profit_map.get(selected_crop, 25000)
        
        # Financial breakdown (Revenue - Expenses = Net Profit)
        # Assuming Net Profit is 40% of Total Revenue
        total_revenue = net_profit_per_acre * land_size / 0.4
        net_profit = net_profit_per_acre * land_size
        expenses = total_revenue - net_profit
        
        # Breakdown of expenses
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
        
        # Visualization
        st.subheader("📊 Expense vs Profit Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = ['Seed Cost', 'Fertilizer Cost', 'Labor Cost', 'Misc. Expenses', 'Net Profit']
        sizes = [seed_cost, fertilizer_cost, labor_cost, misc_cost, net_profit]
        colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#2c5f2d']
        explode = (0, 0, 0, 0, 0.1)  # explode Net Profit
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 10})
        ax.axis('equal')
        
        # Make the background transparent for a cleaner look
        fig.patch.set_alpha(0.0)
        ax.set_facecolor((0, 0, 0, 0))
        
        st.pyplot(fig)

# --- Agri-News & Tips Page ---
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
