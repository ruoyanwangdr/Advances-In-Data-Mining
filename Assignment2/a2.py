'''
@author: anon

Assignment 2: Counting Distinct Elements

Citations: Cardinality Estimation from http://blog.notdot.net/2012/09/Dam-Cool-Algorithms-Cardinality-Estimation
'''

import numpy as np
import random
import math
import time

def getSeed():
    """
    generate random stream of 32-bit numbers

    """
    n = int(1e6)
    a = []
    for x in range(1,n):
        a.append(random.getrandbits(32))
    return a
def rae(size,cardinality):
    """
    relative approximate error
    """
    return np.abs(size-cardinality)/size

def trailing_zeroes(num):
    """
    Counts the number of trailing 0 bits in hashed value 'num'
    """
    # print(num)
    if num == 0:
        return 32 # assuming 32bit inputs
    p = 0
    while (num >> p) & 1 == 0:
        p += 1
    return p

def LogLog(vals,k):
    """
    estimates the number of unique elements in the input set values
    using the durand-flajolet (2003) algorithm.

    arguments:
        values: an iterator of hashable elements to estimate the cardinality of
        k: the number of bits of hash to use as a bucket number

    parameters:
        m: 2**k buckets
        vals: to find out
        M_zeroes: M-elements of data stream, 1 for each bucket
    """
    alpha = 0.79402
    m = 2**k
    M_zeroes = [0]*m # initialize M^(1),...,M^(m) to 0
    # print(M_zeroes)
    for value in vals:
        h = hash(value)
        bucket = h & (m - 1) # mask out the k least significant bits as bucket ID
        h_value= h >> k
        # print bin(h_value)
        M_zeroes[bucket] = max(M_zeroes[bucket], trailing_zeroes(h_value))
    return(2**(float(sum(M_zeroes))/m)*m*alpha)

def pcsa(values):
    '''
    implementation of the flajolet-martin algorithm, or the
    probabilistic counting with stochastic averaging. algorithm described
    in flajolet & martin (1985).
    '''
    R = 0
    max_zeroes = []
    for val in values:
        # print(val,type(val))
        if trailing_zeroes(val) > R:
            R = trailing_zeroes(val)
    print('cardinality estimate: %d' % R)
    return 2**R
    # m = 2**10 #number of hash functions to be tested
    # phi = 0.77351 #magic number phi
    # R_list = [0]*m  #list of highest number of trailing zeroes
    # for h in range(m):
    #     k = 32  #number of bits for each hash value
    #     hash_vals = np.matrix([[np.random.randint(0, 1) for i in range(k)] for j in range(values)]) #randomly generated hash values
    #     R = 0
    #     for value in np.arange(values):
    #         arr = np.array(hash_vals[value,:])[0] #makes an array out of each matrix row
    #         b = ''.join(map(str,arr))  #turns array into string
    #         h_value = int(b,2)  #turns string of binary to integer
    #         R_list[h] = max(R_list[h], trailing_zeroes(h_value))  #keeps track of highest R
    # S = []
    # #here i split
    # for j in np.arange(0,len(R_list),int(math.ceil(np.log2((values))))):
    #     part_list=R_list[j:j+int(math.ceil(np.log2((values))))]
    #     S.append(np.mean(part_list))
    # return((m/phi)*(2**np.median(S)))

"""
iterating k buckets (between 64 bits ie 2^4 and 1024 bits 2^10)
and over m number of buckets in same range,
over n size (10000).
"""
if __name__ == '__main__':
    n = int(1e6) # size of data stream
    # stream = []
    # for i in range(1):
    #     stream.append(getSeed())
    #     np.array(stream)

    stream = [random.getrandbits(32) for i in range(n)] # randomly generated values
    # print('true n:',np.unique(stream))
    # start_time = time.time()
    # pcsa_run = np.array([pcsa(stream) for j in range(1)])
    # pcsa_rae = rae(n,pcsa_run)
    # print('Run time: %s seconds'  %(time.time() - start_time))
    # print('Relative PCSA approximate error: %.3f' % pcsa_rae)
    # # print(pcsa_run,pcsa_rae)

    start_time = time.time()
    loglog_run = np.array([LogLog(stream,10) for j in range(1)])
    loglog_rae = rae(n,loglog_run)
    print('Run time: %s seconds'  %(time.time() - start_time))
    print('Relative LogLog approximate error: %.3f' % loglog_rae)
    #print(loglog_run)
    #print('LogLog: %.3f' % ((np.abs(10000-LogLog(vals,5)/10000) for j in range(1))))
    #print('Run time %s seconds'  % (time.time() - start_time))
