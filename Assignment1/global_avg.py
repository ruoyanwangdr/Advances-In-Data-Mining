import numpy as np
import pandas as pd
import time

"""
"""

# ratings = np.load('rate.npy')
def global_avg(fn):
    #split data into 5 train and test folds
    nfolds=5

    #allocate memory for results:
    err_train=np.zeros(nfolds)
    err_test=np.zeros(nfolds)
    mae_train=np.zeros(nfolds)
    mae_test=np.zeros(nfolds)
    print ('Recommendations based on global average of all ratings:')
    start_time = time.time()

    #to make sure you are able to repeat results, set the random seed to something:
    np.random.seed(1)

    seqs=[x%nfolds for x in range(len(fn))]
    np.random.shuffle(seqs)

    #for each fold:
    for fold in range(nfolds):
        train_sel=np.array([x!=fold for x in seqs])
        test_sel=np.array([x==fold for x in seqs])
        train=fn[train_sel]
        test=fn[test_sel]

    #calculate model parameters: mean rating over the training set:
        gmr=np.mean(train[:,2])

    #apply the model to the train set:
        err_train[fold] = np.sqrt(np.mean((train[:,2] - gmr)**2))
        mae_train[fold] = np.mean(np.abs(train[:,2] - gmr))

    #apply the model to the test set:
        err_test[fold] = np.sqrt(np.mean((test[:,2] - gmr)**2))
        mae_test[fold] = np.mean(np.abs(test[:,2] - gmr))

    #print errors:
        print("Fold " + str(fold+1) + ": RMSE_train = " + str(err_train[fold]) + "; RMSE_test = " + str(err_test[fold]))

    #print the final conclusion:
    print("\n")
    print("Mean RMS error on TRAIN: " + str(np.mean(err_train)))
    print("Mean RMS error on  TEST: " + str(np.mean(err_test)))
    print('MAE on TRAIN: ' + str(np.mean(mae_train)))
    print('MAE on  TEST: ' + str(np.mean(mae_test)))

    print("global avg runtime: %s seconds ---" % (time.time() - start_time))
