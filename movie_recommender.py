from audioop import reverse
from matplotlib.pyplot import axis
import pandas as pd
import numpy  as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title_from_id(movieId):
    return df[df.index == movieId]["title"].values[0]

def get_id_from_title(title):
    return df[df.title == title]["movieId"].values[0]


# Read Csv File
df = pd.read_csv("movies.csv")
#print(df.columns)

# Selection of features
features = ["movieId", "title", "genres"]

# Creation of a column in DF which combines all selected features

def combine_features(row):
    return str(row["movieId"])+" "+str(row["title"])+" "+str(row["genres"])

df["combined_features"] = df.apply(combine_features, axis=1)
#print(df["combined_features"].head())

# Creation of a count matrix from the new column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

# Cosine similarity computation based on count matrix

cos_sim = cosine_similarity(count_matrix)
movie_user_likes = "Forrest Gump (1994)"
#Get the id of movie from its title

movie_id  = get_id_from_title(movie_user_likes)

similar_movies = list(enumerate(cos_sim[movie_id]))

# List of similar movies in descending order
sorted_similar_movies = sorted(similar_movies, key= lambda x:x[1], reverse= True)

# Printing titles of first 20 movies
i=0
for movie in similar_movies:
    print (get_title_from_id(movie[0]))
    i = i+1
    if i>20:
        break