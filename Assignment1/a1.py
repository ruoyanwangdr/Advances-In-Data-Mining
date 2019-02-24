"""
Created 15/09/2018
@author: anon

Assignment 1: Recommender system

Citations: Wolacyzk demo_avg_rating.py and demo_least_squares.py
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
import global_avg
import movie_avg
import user_avg
import combo
import matrix_fact
import matplotlib.pyplot as plt
# import als

# np.set_printoptions(np.inf)
# f_path = ''
# ratings = np.genfromtxt(f_path+'ratings.dat',delimiter='::',dtype='int')
# np.save('rate',ratings)

ratings = np.load('rate.npy')
global_avg.global_avg(ratings)
user_avg.user_avg(ratings)
movie_avg.movie_avg(ratings)
combo.combo(ratings)
matrix_fact.matrix_fact(ratings)
#als
# plt.hist(ratings[:,2],bins=5,fill=False,histtype='step',color='k')
# plt.show()
#print(ratings[:,2])
