# 🌱 AgriSmart: Intelligent Crop & Fertilizer Recommendation System

AgriSmart is a modern, full-stack web application designed to empower farmers and agricultural enthusiasts with AI-driven insights. By analyzing soil metrics and local weather data, AgriSmart recommends the most profitable crops to plant, suggests appropriate fertilizers, and provides interactive financial calculators to estimate yields and returns.

## ✨ Features

- **🌾 AI Crop Prediction**: Utilizes a Random Forest Classifier trained on soil nutrients (Nitrogen, Phosphorus, Potassium), pH levels, and weather data (temperature, humidity, rainfall) to recommend the optimal crop.
- **🌤️ Live Weather Integration**: Automatically fetches real-time weather data based on the user's city via the Open-Meteo API.
- **🧪 Fertilizer Recommendations**: Analyzes soil nutrient gaps against ideal crop metrics to recommend specific fertilizers (e.g., Urea, DAP, NPK 19:19:19) tailored to the soil type.
- **💰 Interactive Yield & ROI Calculator**: Calculate estimated total revenue, seed costs, fertilizer costs, labor, and net profit based on land size, visualized with dynamic pie charts.
- **📰 Agricultural News & Tips**: Keep up to date with the latest farming trends, subsidies, and daily agricultural tips.
- **📄 Downloadable PDF Reports**: Generate comprehensive PDF reports summarizing input metrics, crop recommendations, growing tips, and market price trends for offline reference.
- **🔒 User Authentication & Dashboard**: Secure login/registration system that allows users to save and track their past predictions in a personalized dashboard.

## 🛠️ Technology Stack

**Frontend**
- [Streamlit](https://streamlit.io/): Powers the rich, interactive user interface.
- [Matplotlib](https://matplotlib.org/): For dynamic data visualization and charting.
- [FPDF](https://pyfpdf.github.io/fpdf2/): For seamless PDF report generation.

**Backend**
- [Flask](https://flask.palletsprojects.com/): Lightweight Python framework handling API routing and model inference.
- [Flask-SQLAlchemy](https://flask-sqlalchemy.palletsprojects.com/): ORM for database management.
- [SQLite](https://www.sqlite.org/): Local database for storing user credentials and prediction history.
- [Flask-Login](https://flask-login.readthedocs.io/): For secure session management.

**Machine Learning**
- [Scikit-Learn](https://scikit-learn.org/): Powers the core prediction engine using `RandomForestClassifier`.
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/): For data processing and manipulation.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed on your system.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YoshithaRoyal/AgriSmart.git
   cd AgriSmart
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

To run AgriSmart locally, you need to start both the Flask backend server and the Streamlit frontend interface.

1. **Start the Flask Backend**
   In your terminal, run:
   ```bash
   python app.py
   ```
   The Flask server will start on `http://127.0.0.1:5000`. It will automatically generate a synthetic dataset and train the ML model on the first run.

2. **Start the Streamlit Frontend**
   Open a *new* terminal window (make sure your virtual environment is activated), and run:
   ```bash
   streamlit run streamlit_app.py
   ```
   The application will automatically open in your default web browser at `http://localhost:8501`.

## 🤝 Contributing
Contributions are always welcome! Feel free to open an issue or submit a Pull Request if you have ideas for new features or improvements.

## 📜 License
This project is open-source and available under the MIT License.
