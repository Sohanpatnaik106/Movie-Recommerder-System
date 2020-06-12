# Gather the data and importing the required modules

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')
df = df[['Title','Genre','Director','Actors','Plot']]

sw = stopwords.words("english")

# Extracting the keywords from the plot of the movies

keywords = []

for index, row in df.iterrows():
	plot = row['Plot']
	plot = plot.lower()
	tokenize = word_tokenize(plot)
	tokens_without_sw = " ".join([word for word in tokenize if not word in sw])
	keywords.append(tokens_without_sw)

# Creating the model using cosine-similarity

vectorizer = CountVectorizer()
cv = vectorizer.fit_transform(keywords)
cosine_sim = cosine_similarity(cv, cv)

# Function the returns the indices of the top 15 similar movies with the movie entered by the user

def recommendations(title, cosine_sim = cosine_sim):
	index = int()
	for i in range(len(df)):
		if(df['Title'][i] == title):
			index = i
			break

	recommended_movies_indices = sorted(range(len(cosine_sim[index])), key = lambda sub: cosine_sim[index][sub])[-16:] 
	return recommended_movies_indices

# Taking input from the user and print the recommended movies

movie_user_likes = input("Enter the movie you like : ")
print("The recommended movies for you are : ")

recommended_movies_indices = recommendations(movie_user_likes, cosine_sim)

for i in range(len(recommended_movies_indices)):
	if df['Title'][recommended_movies_indices[i]] != movie_user_likes:
		print(df['Title'][recommended_movies_indices[i]])