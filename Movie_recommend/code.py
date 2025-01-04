# movie recommendation system using cosine similarity

# importing the dependencies
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# reading the dataset
# loading the dataset
movies=pd.read_csv(r'C:\Users\csvis\.vscode\ML projects\Movie_recommend\movies.csv')
# displaying the first few rows of the dataset
movies.head()

#displaying the shape of the dataset
movies.shape

#displaying the column of the dtaaset
movies.columns

#displaying the information of the dataset
movies.info()

movies.describe()

# checking for the missing values
movies.isnull().sum()

# checking for the missing values
movies.isnull().sum()

# creating a Tfidf Vectorizer to convert the text data into a matrix of TF-IDF features
tfdif=TfidfVectorizer(stop_words='english')

#replace the NaN values with an empty string
movies['genres']=movies['genres'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfdif_matrix=tfdif.fit_transform(movies['genres'])

#output the shape of tfdif_matrix
tfdif_matrix.shape

#computing the cosine similarity on tfdif_matrix
cosine_sim=cosine_similarity(tfdif_matrix, tfdif_matrix)

#displaying the cosine similarity matrix
cosine_sim



