import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

cleaned_data =[]
encoded_data =[]

if __name__ == '__main__': 
    cleaned_data = pd.read_csv('cleaned_data.csv')
    encoded_data = pd.read_csv('encoded_data.csv')
    cleaned_data['Maincity'] = cleaned_data['Maincity'].fillna("")


st.title('Swiggy Recommendation System')
# st.write(cleaned_data.head())

unique_places = cleaned_data['Place'].unique()
unique_maincities = cleaned_data['Maincity'].unique()
unique_cuisines = cleaned_data['cuisine'].unique()

search_place = st.selectbox('Place', unique_places)
search_maincity = st.selectbox('Main City', unique_maincities)
search_cuisine = st.selectbox('Cuisine', unique_cuisines)

min_cost = int(cleaned_data['costvalue'].min())
max_cost = int(cleaned_data['costvalue'].max())
min_rating = float(cleaned_data['rating'].min())
max_rating = float(cleaned_data['rating'].max())

search_costvalue = st.slider('Price', min_value=min_cost, max_value=max_cost, value=(min_cost, max_cost))
search_rating = st.slider('Rating', min_value=min_rating, max_value=max_rating, value=(min_rating, max_rating), step=0.1)

search_button = st.button('Search')



def get_kmeans_recommendations(query, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(encoded_data)
    cluster = kmeans.predict([query])
    similar_indices = [i for i, x in enumerate(kmeans.labels_) if x == cluster]
    return cleaned_data.iloc[similar_indices]

def get_cosine_recommendations(query):
    similarities = cosine_similarity([query], encoded_data)
    similar_indices = similarities.argsort()[0][-10:][::-1]
    return cleaned_data.iloc[similar_indices]


if search_button:
    query_encoded = encoded_data[(cleaned_data['Place'] == search_place) & 
                                 (cleaned_data['Maincity'] == search_maincity) & 
                                 (cleaned_data['cuisine'] == search_cuisine) & 
                                 (cleaned_data['costvalue'] >= search_costvalue[0]) & 
                                 (cleaned_data['costvalue'] <= search_costvalue[1]) & 
                                 (cleaned_data['rating'] >= search_rating[0]) & 
                                 (cleaned_data['rating'] <= search_rating[1])]
    
    if not query_encoded.empty:
        query_encoded = query_encoded.iloc[0].values
        
        # st.subheader('K-Means Recommendations')
        # kmeans_recommendations = get_kmeans_recommendations(query_encoded)
        # st.write(kmeans_recommendations)
        

        st.subheader('Cosine Similarity Recommendations')
        cosine_recommendations = get_cosine_recommendations(query_encoded)
        # st.write(cosine_recommendations)

        for i, v in cosine_recommendations.iterrows():
            container = st.container(border=True)  
            container.markdown(f"""
            <h3 style='font-size: 24px;'><b>{v['name']}</b></h3>
            """, unsafe_allow_html=True)
            container.write(f"City: {v["city"]}")
            container.write(f"Cuisine: {v['cuisine']}")
            container.write(f"Cost: {v["cost"]} | Rating: {"⭐"*int(v["rating"])}")
            container.write(f"Rating Count: {v["rating_count"]}")
            container.link_button("More Details", f"{v["link"]}")
            
    else:
        st.write('No matching restaurant found in the encoded data.')
