# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.
    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.
    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    pass
    l = len(movies)
    movies['tokens'] = pd.Series(np.random.randn(l), index=movies.index)

    genres = []
    genres = movies['genres'].tolist()
    tokens = []
    '''
    #print(genres)
    for i in movies.index:
        #print(i)
        movies['tokens'][i] = tokenize_string(movies['genres'][i])
    print(movies)
    '''
    for i in genres:
        tokens.append(tokenize_string(i))
    tokens_col = pd.Series(tokens)
    movies['tokens'] = tokens_col.values
    #print(movies)
    return movies

    


#movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi'] ,[789, 'Romance']], columns=['movieId', 'genres'])
#movies = tokenize(movies)
def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i
    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    pass
    l = len(movies)
    movies['features'] = pd.Series(np.random.randn(l), index=movies.index)
    vocab = {}
    tokens = []
    for i in movies['tokens']:
        for j in i:
            if j not in tokens:
                tokens.append(j)
    tokens = sorted(tokens)
    #print('tokens')
    #print(tokens)
    for i in range(len(tokens)):
        vocab[tokens[i]] = i
        
    #print(sorted(vocab.items()))
    df_vocab = defaultdict(list)
    for k, v in vocab.items():
        for index in range(l):
            if k in movies['tokens'][index]:
                df_vocab[k].append(movies['movieId'][index])
        
    
    ndList = []
    for index in range(l):
        ndList.append(movies['tokens'][index])
    #print(ndList)
    new_movies = pd.DataFrame(data = movies, index = movies.index)
    #print(new_movies)
    new_movies.drop_duplicates(['movieId'], inplace = True)
    #print(new_movies)
    #print(tokens)
    #print(vocab)
    #max_k = max(vocab.values())
    N = len(movies)
    #print(';;;;;;;;;;;;;;')
    #print(len(vocab))
    #print(max_k, N)
    '''
    no_dup_list = []
    for index in range(len(new_movies)):
                       no_dup_list.append(new_movies['tokens'][index])
    '''
    csr = []
    #print(no_dup_list)
    for r in movies.itertuples():
        row = []
        col = []
        data = []
        maxi = {}
        #print(movies['tokens'][index])
        #print(len(movies['tokens'][index]))
        row = [0] * len(r.tokens)
        #print(row)
        maxi = Counter(r.tokens)
        max_k = max(maxi.values())        
        #print(row)
        for i in r.tokens:
            
            #maxi = {}
            print(i)
            col.append(vocab[i])
            tf = r.tokens.count(i)
            '''
            maxi = Counter(movies['tokens'][index])
            max_k = max(maxi.values())
            '''
            #print(tf)
            '''
            for j in no_dup_list:
                if i in j:
                    count += 1
            '''
            dfi = len(df_vocab[i])
            tfidf = (tf / max_k) * math.log10(N/dfi)
            data.append(tfidf)
            #print(tfidf)
    
        csr.append(csr_matrix((data,(row, col)), shape = (1, len(vocab))))
    movies['features'] = csr

    #print(movies)
    #print(movies['features'][0].toarray())
    return movies, vocab
      
#featurize(movies)


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    pass
    #print("Cos_Sim")
    #print(float(a.dot(b.transpose())/((np.sqrt(a*(a.transpose())))*(np.sqrt(b*(b.transpose()))))))
    return(float(a.dot(b.transpose())/((np.sqrt(a*(a.transpose())))*(np.sqrt(b*(b.transpose()))))))


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.
    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.
    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.
    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    pass
    cosine_sim(movies['features'][0],movies['features'][1])
    #print(ratings_train)
    #print(ratings_test)
    rating = []
    
    
    for test in ratings_test.itertuples():
        cosine_scores = []
        weighted_ratings = []
        r = []
        new_movies_test = movies[movies['movieId'] == test.movieId]
        #print(new_movies_test)
        csr_mat_test = new_movies_test.iloc[0]['features']
        #print(csr_mat_test.toarray())
        for train in ratings_train.itertuples():
            if train.userId == test.userId:
                new_movies_train = movies[movies['movieId'] == train.movieId]
                #print(new_movies_train)
                csr_mat_train = new_movies_train.iloc[0]['features']
                c = cosine_sim(csr_mat_test, csr_mat_train)
                r.append(train.rating)
                
                if(math.isnan(c) == False):
                    w = train.rating*c
                    weighted_ratings.append(w)            
                    cosine_scores.append(c)
        cosine_score_sum = sum(cosine_scores)
        weighted_rating_sum = sum(weighted_ratings)
        if(cosine_score_sum == 0):
            #print('Mean')
            #print(sum(r)/float(len(r)))
            rating.append(sum(r)/len(r))
        else :   
            #print('Division')
            #print(weighted_rating_sum/cosine_score_sum)
            rating.append(weighted_rating_sum/cosine_score_sum)
    rating_array = np.array(rating)
    #print(rating_array)
    return rating_array
     
                

def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
