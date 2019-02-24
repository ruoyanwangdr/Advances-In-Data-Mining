import numpy as np
import pandas as pd
import time

start_time = time.time()
print('Matrix factorization. Params: num_factors=10, num_iter=75, reg=0.05, learn_rate=0.005, np.random.seed(1)')

def gravityAlgorithm(train,test,num_factors=10,num_iter=75,reg=0.05,learn_rate=0.005):
    """
    Gravity matrix implementation from Takacs et al. (2007). We compute:
    X = UM where U is an nxm matrix and M is an mxk matrix.
    U is our user column, M is our movie column.
    """
    train=np.array(train)
    test=np.array(test)
    U = np.random.rand(max(np.max(train[:,0]), np.max(test[:,0]) + 1), num_factors)
    M = np.random.rand(num_factors,max(np.max(train[:,1]),np.max(test[:,1])) + 1)
    #print(len(U),len(M))

    for i in range(num_iter):
        for j in range(len(train)):
            e_grad = 2 * (train[j,2] - np.dot(U[train[j,0],:], M[:,train[j,1]]))
            # compute the gradient of e**2 to M before changing U (negative of the gradient)
            m_grad = e_grad * U[train[j,0],:]
            u_grad = e_grad * M[:,train[j,1]]
            U[train[j, 0], :] += learn_rate * (u_grad - reg * U[train[j, 0], :])
            M[:, train[j, 1]] += learn_rate * (m_grad - reg * M[:,train[j, 1]])

        # calculate estimated ratings

        ER = np.dot(U,M)

        # make prediction for train
        predictionTrain = np.zeros(len(train))
        for i in range(len(train)):
            predictionTrain[i] = ER[train[i, 0], train[i, 1]]
            if predictionTrain[i] > 5:
                predictionTrain[i] = 5
            if predictionTrain[i] < 1:
                predictionTrain[i] = 1

        # make prediction for test
        prediction = np.zeros(len(test))
        for i in range(len(test)):
            prediction[i] = ER[test[i,0], test[i, 1]]
            if prediction[i] > 5:
                prediction[i] = 5
            if prediction[i] < 1:
                prediction[i] = 1

        return (predictionTrain, prediction)

def matrix_fact(fn):
    ratings_user=pd.DataFrame(fn)
    ratings_user=ratings_user.append(ratings_user)

    nfolds = 5

    err_train=np.zeros(nfolds)
    err_test=np.zeros(nfolds)
    mae_train=np.zeros(nfolds)
    mae_test=np.zeros(nfolds)

    np.random.seed(1)

    seqs=[x%nfolds for x in range(len(fn))]
    np.random.shuffle(seqs)

    for fold in range(nfolds):
        train_set = np.array([x!=fold for x in seqs])
        test_set = np.array([x==fold for x in seqs])
        train = pd.DataFrame(ratings_user.iloc[train_set], columns=[0, 1, 2], dtype=int)
        test = pd.DataFrame(ratings_user.iloc[test_set], columns=[0, 1, 2], dtype=int)
        # pred = gravityAlgorithm(train, test, num_factors=10, num_iter=75, reg=0.05, learn_rate=0.005)
        err_train[fold] = np.sqrt(np.mean((np.array(train[2]) - gravityAlgorithm(train, test, num_factors=10, num_iter=75, reg=0.05, learn_rate=0.005)[0])**2))
        mae_train[fold] = np.mean(np.abs(np.array(train[2]) - gravityAlgorithm(train, test, num_factors=10, num_iter=75, reg=0.05, learn_rate=0.005)[0]))

        err_test[fold]= np.sqrt(np.mean((np.array(test[2]) - gravityAlgorithm(train, test, num_factors=10, num_iter=75, reg=0.05, learn_rate=0.005)[1])**2))
        mae_test[fold] = np.mean(np.abs(np.array(test[2]) - gravityAlgorithm(train, test, num_factors=10, num_iter=75, reg=0.05, learn_rate=0.005)[1]))
        print("Fold " + str(fold+1) + ": RMSE_train = " + str(err_train[fold]) + "; RMSE_test = " + str(err_test[fold]))

    print("\n")
    print("Mean error on TRAIN: " + str(np.mean(err_train)))
    print("Mean error on  TEST: " + str(np.mean(err_test)))
    print('MAE on TRAIN: ' + str(np.mean(mae_train)))
    print('MAE on  TEST: ' + str(np.mean(mae_train)))
    print("Matrix factorization runtime:  %s seconds ---" % (time.time() - start_time))
