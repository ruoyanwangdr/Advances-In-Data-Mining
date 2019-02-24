import numpy as np
import pandas as pd
import time
from sklearn import linear_model
#RW gets 1.01, 1.01
def mean_user(x):
    user_count=np.bincount(x[0])
    zeros_train = np.array(np.where(user_count[1:len(user_count)] == 0))
    non_zero_train = np.array([np.where(user_count[1:len(user_count)] != 0)])
    times_user_train_correct = np.delete(user_count[1:len(user_count)], zeros_train)
    mean_user = np.array(x.groupby([0])[2].mean())
    full = np.repeat(mean_user,times_user_train_correct)
    return np.array(full)

def mean_movie(x):
    movie_count = np.bincount(x[1])
    zeros_train = np.array(np.where(movie_count[1:len(movie_count)] == 0))
    non_zero_train = np.array([np.where(movie_count[1:len(movie_count)] != 0)])
    times_movie_train_correct = np.delete(movie_count[1:len(movie_count)], zeros_train)
    mean_movie = np.array(x.groupby([1])[2].mean())
    full = np.repeat(mean_movie, times_movie_train_correct)
    return np.array(full)

def combo(fn):
    df = pd.DataFrame(fn)
    ratings_user=pd.DataFrame(fn)
    ratings_user=ratings_user.append(ratings_user)
    user_average = df.groupby(by=0, as_index=False)[2].mean()
    user_average = user_average.append(user_average)
    ratings_movie=pd.DataFrame(fn)
    ratings_movie=ratings_movie.append(ratings_movie)
    movie_average = df.groupby(by=1, as_index=False)[2].mean()
    movie_average = movie_average.append(movie_average)
    global_average = np.mean(fn[:,2])

    nfolds = 5

    err_train=np.zeros(nfolds)
    err_test=np.zeros(nfolds)
    mae_train=np.zeros(nfolds)
    mae_test=np.zeros(nfolds)
    alpha=np.zeros(nfolds)
    beta=np.zeros(nfolds)
    gamma=np.zeros(nfolds)

    np.random.seed(1)

    seqs=[x%nfolds for x in range(len(fn))]
    np.random.shuffle(seqs)

    start_time = time.time()
    print ('Recommendations from a combination of user and movie averages:')
    for fold in range(nfolds):

        train_set=np.array([x!=fold for x in seqs])
        test_set=np.array([x==fold for x in seqs])
        train = pd.DataFrame(ratings_movie.iloc[test_set], columns=[0, 1, 2], dtype=int)
        test = pd.DataFrame(ratings_movie.iloc[test_set], columns=[0, 1, 2], dtype=int)
        X = np.vstack([np.array(mean_user(train)), np.array(mean_movie(train))]).T
        reg = linear_model.LinearRegression()

        reg.fit(X[:,:],np.array(train[2]))

        alpha[fold] = reg.coef_[0]  # coeff of alpha
        beta[fold] = reg.coef_[1]  # coeff of beta
        gamma[fold] = reg.intercept_  # coeff of the intercept (gamma)
        #print alpha[fold], beta, gamma
        # applying the values above to the formula in the book

        pred_train = alpha[fold] * mean_user((train)) + beta[fold] * mean_movie((train)) + gamma[fold]
        pred_test= alpha[fold] * mean_user((test)) + beta[fold] * mean_movie((test)) + gamma[fold]
        pred_train[pred_train > 5] = 5
        pred_train[pred_train < 1] = 1
        pred_test[pred_test>5]=5
        pred_test[pred_test<1]=1

        err_train[fold] = np.sqrt(np.mean((np.array(train[2]) - pred_train)**2))
        mae_train[fold] = np.mean(np.abs(np.array(train[2]) - pred_train))
        err_test[fold] = np.sqrt(np.mean((np.array(test[2]) - pred_test)**2))
        mae_test[fold] = np.mean(np.abs(test[2] - pred_test))
        print("Fold " + str(fold+1) + ": RMSE_train = " + str(err_train[fold]) + "; RMSE_test = " + str(err_test[fold]))



    print("\n")
    print('Mean error on TRAIN: '+ str(np.mean(err_train)))
    print('Mean error on  TEST: ' + str(np.mean(err_test)))
    print ('MAE on TRAIN: ' + str(np.mean(mae_train)))
    print ('MAE on  TEST: ' + str(np.mean(mae_test)))
    print ("alpha =", np.mean(alpha), "; beta =",np.mean(beta) , "; gamma =", np.mean(gamma))

    print("Linear regression runtime:  %s seconds ---" % (time.time() - start_time))
