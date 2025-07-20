import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Smart Restaurant Explorer", layout="wide")
st.title("🌟 Smart Restaurant Explorer")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset .csv")  
    df.fillna({'Cuisines': df['Cuisines'].mode()[0]}, inplace=True)
    return df

df_original = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🌟 Rating Predictor",
    "🌎 Recommendation System",
    "🏙️ Location-Based Analysis"
])

# Shared preprocessing
cat_columns = ['Restaurant Name', 'City', 'Cuisines', 'Currency',
               'Has Table booking', 'Has Online delivery',
               'Is delivering now', 'Switch to order menu']

if page == "🌟 Rating Predictor":
    st.header("Predict Restaurant Rating")

    df = df_original.drop(['Restaurant ID', 'Address', 'Locality',
                           'Locality Verbose', 'Rating color', 'Rating text'], axis=1)

    X = df.drop(['Aggregate rating'], axis=1)
    y = df['Aggregate rating']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_columns)
        ],
        remainder='passthrough'
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42, n_estimators=100))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    rest_name = st.text_input("Enter the restaurant name to predict rating")
    if st.button("Predict"):
        row = df_original[df_original['Restaurant Name'] == rest_name]
        if row.empty:
            st.warning("Restaurant not found.")
        else:
            input_data = row.drop(['Restaurant ID', 'Address', 'Locality', 'Locality Verbose', 'Aggregate rating'], axis=1)
            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Rating for '{rest_name}' is: {prediction:.2f}")

elif page == "🌎 Recommendation System":
    st.header("Restaurant Recommendation")

    df = df_original.drop(['Restaurant ID', 'Address', 'Locality', 'Locality Verbose'], axis=1)
    X = df[['City', 'Cuisines', 'Currency', 'Has Table booking',
            'Has Online delivery', 'Is delivering now',
            'Switch to order menu', 'Price range', 'Average Cost for two']]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['City', 'Cuisines', 'Currency', 'Has Table booking',
                                                            'Has Online delivery', 'Is delivering now',
                                                            'Switch to order menu'])
        ],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X)
    similarity_matrix = cosine_similarity(X_processed)

    city = st.selectbox("Select City", sorted(df['City'].unique()))
    cuisine = st.text_input("Preferred Cuisine", "North Indian")
    price_range = st.selectbox("Select Price Range (1-4)", [1, 2, 3, 4])

    if st.button("Recommend"):
        user_pref = df[(df['City'] == city) &
                       (df['Cuisines'].str.contains(cuisine, case=False)) &
                       (df['Price range'] == price_range)]

        if user_pref.empty:
            st.warning("No matching restaurants found.")
        else:
            ref_idx = user_pref.index[0]
            sim_scores = list(enumerate(similarity_matrix[ref_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            recommended = []
            for idx, score in sim_scores[1:]:
                row = df.iloc[idx]
                if (row['Price range'] == price_range) and (row['City'] == city):
                    if cuisine.lower() in row['Cuisines'].lower():
                        recommended.append(row)
                    if len(recommended) >= 5:
                        break

            if recommended:
                st.subheader("Top 5 Recommendations:")
                for r in recommended:
                    st.markdown(f"**{r['Restaurant Name']}** | {r['Cuisines']} | Rating: {r['Aggregate rating']}")
            else:
                st.info("No similar restaurants found.")

elif page == "🏙️ Location-Based Analysis":
    st.header("Location-Based Restaurant Insights")

    df = df_original.dropna(subset=['Latitude', 'Longitude'])

    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    marker_cluster = MarkerCluster().add_to(m)

    for idx, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['Restaurant Name']} ({row['City']})\nRating: {row['Aggregate rating']}",
        ).add_to(marker_cluster)

    st_data = st_folium(m, width=700, height=500)

    city_counts = df['City'].value_counts().sort_values(ascending=False)
    st.subheader("Top Cities by Restaurant Count")
    fig1, ax1 = plt.subplots(figsize=(12,5))
    sns.barplot(x=city_counts.index[:15], y=city_counts.values[:15], ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    st.pyplot(fig1)

    avg_rating_city = df.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False)
    st.subheader("Top Cities by Avg Rating")
    fig2, ax2 = plt.subplots(figsize=(12,5))
    sns.barplot(x=avg_rating_city.index[:15], y=avg_rating_city.values[:15], ax=ax2, palette='magma')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    st.pyplot(fig2)

    avg_price_city = df.groupby('City')['Price range'].mean().sort_values(ascending=False)
    st.subheader("Top Cities by Price Range")
    fig3, ax3 = plt.subplots(figsize=(12,5))
    sns.barplot(x=avg_price_city.index[:15], y=avg_price_city.values[:15], ax=ax3, palette='coolwarm')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    st.pyplot(fig3)

    st.success("Map and charts loaded successfully.")
