# Restaurant-Explorer
# 🌟 Smart Restaurant Explorer

An interactive Streamlit web application that combines three powerful Machine Learning utilities into a single platform for analyzing and exploring restaurant data.

## 🚀 Features

1. **📊 Restaurant Rating Predictor**  
   - Uses an XGBoost regression model to predict a restaurant’s rating based on its features.  
   - Just enter the restaurant name and get its predicted rating.

2. **🧠 Restaurant Recommendation System**  
   - Recommends restaurants based on user preferences for **city**, **cuisine**, and **price range**.  
   - Uses **cosine similarity** on restaurant features for accurate recommendations.

3. **🌍 Location-Based Analysis**  
   - Interactive map showing restaurant locations with ratings.  
   - Bar charts showing:  
     - Cities with most restaurants  
     - Top cities by average rating  
     - Top cities by average price range

---

## 🛠️ Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io)  
- **ML Models**: `XGBoost`, `cosine_similarity` from `scikit-learn`  
- **Data Manipulation**: `pandas`, `numpy`  
- **Visualization**: `matplotlib`, `seaborn`, `folium`  
- **Map Integration**: `folium`, `streamlit-folium`

---

## 📦 Setup Instructions (Local)

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
3. **Run the app**
   ```bash
   streamlit run app.py

---

## Author
Anuj Thakur
Connect on LinkedIn: linkedin.com/in/anuj-singh-thakur-077b82245/

## 📃 License
This project is licensed under the Apache License 2.0.
