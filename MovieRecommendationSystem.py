
import sklearn as sk
import pandas as pd
import numpy as np

text=["London Paris London","Paris Paris London"]
 
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
a=cv.fit_transform(text)
print(cv.get_feature_names())
print(a.toarray())

from sklearn.metrics.pairwise import cosine_similarity
similarity_score=cosine_similarity(a)

data=pd.read_csv("movie_dataset.csv")
data.isnull().sum()

features=['keywords','cast','genres','director']
for feature in features:
    data[feature]=data[feature].fillna("")

data['combined']=data['keywords']+" "+data['director']+" "+data['genres']+" "+data['cast']


    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
a=cv.fit_transform(data['combined'])

a.toarray()
from sklearn.metrics.pairwise import cosine_similarity
similarity_score=cosine_similarity(a)

scores=similarity_score[0]

j=0
ab=[(i,scores[i]) for i in range(0,4803)]
ab.sort(key = lambda x:x[1])

movie_user_likes='The Green Mile'
index=data[data['original_title']==movie_user_likes].index

similar_movies=list(enumerate(similarity_score[index[0]]))
sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)

names=[]
for movie in sorted_similar_movies[1:10]:
    names.append(data['original_title'][movie[0]])
    
    
