# import packages
import csv
import sklearn
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn import preprocessing
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error


def train_test_split(ratings):
    '''
    function to split data into test and train
    :param ratings: Numpy array having complete data
    :return: Numpy array of test and train data

    '''
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(
            ratings[user, :].nonzero()[0], size=10, replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
    assert(np.all((train * test) == 0))
    return train, test


def pairwise_distances_scratch(matrix):
    '''
    Function to calculate cosine similarity
    :param matrix: Numpy array whose distances are to be
        calculated
    :return distances: Numpy array of calculated distances
        using cosine similarity
    '''
    size = matrix.shape[0]
    distances = np.empty(shape=(size, size), dtype=float)
    for i in range(0, size):
        norm_i = np.linalg.norm(matrix[i])
        for j in range(0, size):
            dot_product = np.dot(matrix[i], matrix[j])
            norm_j = np.linalg.norm(matrix[j])
            if norm_i == 0 or norm_j == 0:
                distances[i][j] = 1
            else:
                distances[i][j] = 1 - (dot_product / (norm_i * norm_j))
    return distances


def predict_topk_nobias(ratings, similarity, kind='user', k=40):
    '''
    predict movies using top-k values and no bias

    In case of user-based rating, certain users may tend to always give
    high or low ratings to all movies. For example, one user might give 5
    stars to all movies he likes and 1 to all he dislikes. Another user might
    give 4 and 2 for the movies he likes and dislikes. To remove this bias,
    mean rating is subtracted each because relative difference in the
    ratings that these users give is more important than the absolute rating
    values. In case of item-based filtering, this issue does not arise because
    we consider all ratings for items from a single user.

    :param ratings: Numpy array of train data (ratings)
    :param similarity: Numpy array of similarity matrix calculated using pairwise
        distances
    :param kind: user/item based similarity
    :param k: value of k for top k ratings
    '''
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        user_bias = ratings.mean(axis=1)
        ratings = (ratings - user_bias[:, np.newaxis]).copy()
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:, i])[:-k - 1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(
                    ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
        pred += user_bias[:, np.newaxis]
    if kind == 'item':
        for j in range(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:, j])[:-k - 1:-1]]
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(
                    ratings[i, :][top_k_items].T)
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))

    return pred


# Matrix factorization with Regularization Approach for Model Based CF from scratch
def matrix_factorization(R, P, Q, K=19, steps=10, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T


def get_rmse(pred, actual):
    '''
    Function to calculate RMSE between predicted and actual values
    :param pred: Predicted ratings
    :param actual: Actual ratings
    :return rmse: RMSE of predicted and actual values
    '''
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()

    return np.sqrt(mean_squared_error(pred, actual))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        help='Path to u.data file',
        required=True)
    parser.add_argument(
        '--item',
        type=str,
        help='Path to u.item file',
        required=True)
    args = parser.parse_args()

    # reading CSV file
    rating_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    df_data = pd.read_csv(
        args.data,
        sep='\t',
        names=rating_cols,
        encoding='latin-1')

    # unique users and items
    n_users = df_data.user_id.unique().shape[0]
    n_items = df_data.movie_id.unique().shape[0]

    # mapping movie id to movie name
    df_item = pd.read_csv(
        args.item,
        sep='|',
        header=None,
        usecols=[
            0,
            1],
        encoding='latin-1')
    df_movie = df_item.values

    # converting data into user-movie matrix
    ratings = np.zeros((n_users, n_items))
    for row in df_data.itertuples():
        ratings[row[1] - 1, row[2] - 1] = row[3]

    ratings = ratings

    # Cosine Similarity

    # user-user
    user_similarity = pairwise_distances_scratch(ratings)
    # item-item
    item_similarity = pairwise_distances_scratch(ratings.T,)

    # prediction- both item and user using function defined
    item_prediction = predict_topk_nobias(
        ratings, item_similarity, kind='item', k=15)
    user_prediction = predict_topk_nobias(
        ratings, user_similarity, kind='user', k=50)

    # Scaling the ratings of movies in range 1 to 5
    # which were normalized while removing bias

    # item
    scaler = preprocessing.MinMaxScaler(feature_range=(1, 5))
    item_prediction = scaler.fit_transform(item_prediction)
    # user
    scaler = preprocessing.MinMaxScaler(feature_range=(1, 5))
    user_prediction = scaler.fit_transform(user_prediction)

    # storing top 5 predictions for each user

    # item
    top_predictions_cosine = np.empty([n_users, 6], dtype=object)
    for i in range(0, item_prediction.shape[0]):
        index = item_prediction[i].argsort()[-5:][::-1]
        top_predictions_cosine[i][0] = i + 1
        k = 1
        for j in index:
            top_predictions_cosine[i][k] = df_movie[j][1]
            k = k + 1

    # user
    top_predictions_cosine_user = np.empty([n_users, 6], dtype=object)
    for i in range(0, user_prediction.shape[0]):
        index = user_prediction[i].argsort()[-5:][::-1]
        top_predictions_cosine_user[i][0] = i + 1
        k = 1
        for j in index:
            top_predictions_cosine_user[i][k] = df_movie[j][1]
            k = k + 1

    # Singular Value Decomposition

    # get SVD components from train matrix. Choose k.
    U, sigma, vT = svds(ratings, k=19)
    s_diag_matrix = np.diag(sigma)
    # prediction using SVD
    svd_pred = np.dot(np.dot(U, s_diag_matrix), vT)

    # scaling the ratings in range 1 to 5
    scaler = preprocessing.MinMaxScaler(feature_range=(1, 5))
    svd_pred = scaler.fit_transform(svd_pred)

    # storing top 5 movies for each user
    top_predictions_svd = np.empty([n_users, 6], dtype=object)
    for i in range(0, svd_pred.shape[0]):
        index = svd_pred[i].argsort()[-5:][::-1]
        top_predictions_svd[i][0] = i + 1
        k = 1
        for j in index:
            top_predictions_svd[i][k] = df_movie[j][1]
            k = k + 1

    # Regularization - matrix factorization

    # get Matrix components from train matrix. Choose k.
    R = np.array(ratings)
    N = len(R)
    M = len(R[0])
    K = 19
    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)

    nP, nQ = matrix_factorization(R, P, Q)
    R_pred = np.dot(nP, nQ.T)
    top_predictions_R = np.empty([n_users, 6], dtype=object)
    for i in range(0, R_pred.shape[0]):
        index = R_pred[i].argsort()[-5:][::-1]
        top_predictions_R[i][0] = i + 1
        k = 1
        for j in index:
            top_predictions_R[i][k] = df_movie[j][1]
            k = k + 1

    # writing to csv file

    # Create output directory
    Path('./output_files').mkdir(parents=True, exist_ok=True)

    heading_list = []
    heading_list.append('User ID')
    heading_list.append('Movie 1')
    heading_list.append('Movie 2')
    heading_list.append('Movie 3')
    heading_list.append('Movie 4')
    heading_list.append('Movie 5')

    print('Item-based RMSE: ' + str(get_rmse(item_prediction, ratings)))
    # item-item predictions
    with open('./output_files/cosine_predictions_item.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(heading_list)
        wr.writerows(top_predictions_cosine)

    print('User-based RMSE: ' + str(get_rmse(user_prediction, ratings)))
    # user-user predictions
    with open('./output_files/cosine_predictions_user.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(heading_list)
        wr.writerows(top_predictions_cosine_user)

    print('SVD RMSE: ' + str(get_rmse(svd_pred, ratings)))
    # writing to csv file (SVD)
    with open('./output_files/svd_predictions.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(heading_list)
        wr.writerows(top_predictions_svd)

    print('NMF RMSE: ' + str(get_rmse(R_pred, ratings)))
    #writing to csv file (REG)
    with open('./output_files/regularization_predictions.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(heading_list)
        wr.writerows(top_predictions_R)


if __name__ == '__main__':
    main()
