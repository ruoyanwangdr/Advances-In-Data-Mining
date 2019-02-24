import numpy as np
import pandas as pd
import time

def mean_user(x):
    user_count=np.bincount(x[0])
    zeros_train = np.array(np.where(user_count[1:len(user_count)] == 0))
    non_zero_train = np.array([np.where(user_count[1:len(user_count)] != 0)])
    times_user_train_correct = np.delete(user_count[1:len(user_count)], zeros_train)
    mean_user = np.array(x.groupby([0])[2].mean())
    full = np.repeat(mean_user,times_user_train_correct)
    return np.array(full)

def user_avg(fn):
    df = pd.DataFrame(fn)
    ratings_user=pd.DataFrame(fn)
    ratings_user=ratings_user.append(ratings_user)
    user_average = df.groupby(by=0, as_index=False)[2].mean()
    user_average = user_average.append(user_average)
    global_average = np.mean(fn[:,2])

    nfolds = 5

    err_train=np.zeros(nfolds)
    err_test=np.zeros(nfolds)
    mae_train=np.zeros(nfolds)
    mae_test=np.zeros(nfolds)

    np.random.seed(1)

    seqs=[x%nfolds for x in range(len(fn))]
    np.random.shuffle(seqs)

    start_time = time.time()
    print ('Recommendations from all user averages:')
    for fold in range(nfolds):


        train_set=np.array([x!=fold for x in seqs])
        test_set=np.array([x==fold for x in seqs])
        train_user=pd.DataFrame(ratings_user.iloc[train_set],columns=[0, 1, 2],dtype=int)
        test_user=pd.DataFrame(ratings_user.iloc[test_set],columns=[0, 1, 2],dtype=int)

        # apply the model to the train set:
        err_train[fold] = np.sqrt(np.mean((np.array(train_user[2])-mean_user(train_user))**2))
        err_test[fold] = np.sqrt(np.mean((np.array(test_user[2])-mean_user(test_user))**2))
        mae_train[fold] = np.mean(np.abs(np.array(train_user[2])-mean_user(train_user)))
        mae_test[fold] = np.mean(np.abs(np.array(test_user[2])-mean_user(test_user)))



        print("Fold " + str(fold+1) + ": RMSE_train = " + str(err_train[fold]) + "; RMSE_test = " + str(err_test[fold]))



    print("\n")
    print("Mean error on TRAIN: " + str(np.mean(err_train)))
    print("Mean error on  TEST: " + str(np.mean(err_test)))
    print('MAE on TRAIN:' + str(np.mean(mae_train)))
    print('MAE on  TEST:' + str(np.mean(mae_test)))


    print("User avg runtime:  %s seconds ---" % (time.time() - start_time))
