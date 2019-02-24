"""
created on fri 10/12/2018 at 10:11
@author: anon1 & anon2
a3: finding similar users in netflix data. we use lsh and jaccard similarity
to find unique candidate pairs of similar users (lsh) and then confirm them by
calculating their true similarity (jaccard). we load the netflix data, compute
a signature matrix comprising of only rows = num_movies and as many columns as
the original matrix. we then use the banding method for our lsh algorithm to
'hash' column candidate pairs to the same bucket according to from the same
band. signature length is given to be 100 here.

sources:
https://stackoverflow.com/questions/38674027/find-the-row-indexes-of-several-values-in-a-numpy-array
sparse matrices documentation page
https://www.youtube.com/watch?v=bQAYY8INBxg # stanford lecture
https://www.youtube.com/watch?v=bQAYY8INBxg # stanford lecture
MMDS textbook

"""

import itertools
import numpy as np
import scipy.sparse as sps
from timeit import default_timer as timer
import sys as sys

def lsh_similarity(u1, u2, sigm):
    """computes similarity from user1 and user2 signature matrices.
    from the text, we know that sim(u1,u2) = [h(u1) == h(u2)] / h(u1)
    so we implement here
    args:
        u1, u2: user pair
          sigm: signature matrix
    returns:
        lsh_sim: [h(u1) == h(u2)] / h(u1)

    """
    # count equivalent non-zero values for each user
    sig_sim = float(np.count_nonzero(sigm[:, u1] == sigm[:, u2]))
    sig_length = len(sigm[:, u1]) # length of the set
    lsh_sim = sig_sim / sig_length
    return lsh_sim

def jaccard_similarity(u1, u2, dense_matrix):
    """computes jaccard similarity.
    args:
              u1, u2: set user pair
        dense_matrix: user-movie ratings data with 0s included
    returns:
        jaccard similarity

    """
    # jaccard similarity computes similarity for two sets
    # our 'dense' matrix is boolean
    set1 = dense_matrix[:, u1]
    set2 = dense_matrix[:, u2]
    intersect = np.sum(set1 & set2)
    union = np.sum(set1 | set2)
    # intersect = np.sum(dense_matrix[:, u1] & dense_matrix[:, u2])
    # union = np.sum(dense_matrix[:, u1] | dense_matrix[:, u2])
    jsim = np.float(intersect) / np.float(union)
    return jsim

def load_data(path):
    """loads user, movie data and makes csc sparse matrix.
    args:
        path: user inputted path
    returns:
        compressed sparse column matrix, maximum number of users and movies

    """
    # load npy data from path
    data = np.load(path)
    user_data = data[:, 0] # array of users
    movie_data = data[:, 1] # array of movies
    num_users = np.max(user_data) # total number of users
    num_movies = np.max(movie_data) # total number of movies
    row = movie_data
    col = user_data
    ones = np.ones(len(data)) # 1 for each entry where user rated movie

    # shape +1 in each dimension for indexing of .npy vs python standard
    # making the data type int or float doesn't seem to change the matrix
    # itself, but at the end when jaccard_similarity is computed
    # the bitwise operators don't work. so the only information we care about
    # is whether or not a movie is rated by a user, represented by a 1
    sp_matrix = sps.csc_matrix((ones, (row, col)), \
                                shape=(num_movies+1, num_users+1), dtype='b')
    return sp_matrix, num_users, num_movies

def make_signatures(sparse_matrix, num_users, num_movies, seed):
    """minhashing function to create signature matrix for data.
    args:
        sparse_matrix: csc_matrix that was created in load_data
            num_users: maximum number of users
           num_movies: maximum number of movies
                 seed: user inputted random seed. we'll set a new random seed
                       for each random permutation
    returns:
        signature_matrix: 3.3.2 in the text. the ith column of M is replaced by
                          the minhash signature for (the set of) the ith column
                          also a slide from the lecture that details it.
            permutations: recommended number is 100 from the text

    """
    # instructed minimum number of permutations
    # each minhash will have its own random seed
    permutations = 100
    # initialize signature matrix for permutations
    signature_matrix = np.zeros((permutations, num_users))

    for i in range(permutations):
        next_seed = seed*i
        np.random.seed(next_seed)

        # take a random permutation of row order then
        # swap og sparse matrix with randomly permutated row
        # this will be our 'hash' function since we don't actually need one
        rows = np.random.permutation(num_movies)
        swapped_matrix = sparse_matrix[rows, :]

        # find and store index of first 1 found in jth column of hash:
        # user data matrix.indices[indptr] tells you where data starts and stops
        # if hash finds a one in [i, j] it puts that in the signature matrix
        for j in range(num_users):
            # each hash is different because it's a random permutation of rows
            hash = swapped_matrix.indices\
            [swapped_matrix.indptr[j]:swapped_matrix.indptr[j + 1]].min()
            signature_matrix[i, j] = hash
    return signature_matrix, permutations

def lsh(signature_matrix):
    """lsh algorithm to find similar users.

    an implementation of the lsh algorithm found in section 3.4.1 in the text.
    we use a signature length of 100, with 20 bands and 5 rows as per the
    example in the text.
    args:
        signature_matrix: compressed representation of data
    returns:
        candidate_pairs: those pairs which reach the threshold of t > 0.5

    """

    # signature length of 100, 20 bands with 5 rows
    n_bands = 20
    n_rows = 5 # permutations / n_bands

    # buckets for sigs to be hashed into
    buckets = []

    # lsh algorithm from 3.4.1
    current_row = 0
    for i in range(n_bands):
        # slice indices must be integers
        band = signature_matrix[current_row:int(n_rows) + current_row, :]
        current_row += int(n_rows)

        # ravel_multi_index takes a 2D array and returns mapped linear indices
        # to be computed. we have to use ints here as well in order to use
        # ravel_multi_index. we sort the band by its (linear-index equivalent)
        # indices and make a bucket from iterating indices in band
        # similar groups go to similar coordinates after argsorted
        linind = \
        np.ravel_multi_index(band.astype(int),band.max(1).astype(int) + 1)
        argsorted = linind.argsort() # sort the values by index
        arr = linind[argsorted] # entries are index values

        # use above 'hash' to make our buckets
        # we use the same hash function but separate bucket arrays for each
        # band so that columns with the same vector don't get hashed to
        # the same bucket. code below splits each bucket into its own array
        # to specify position later on, iterated through the whole bucket_array
        bucket_array = np.array( \
                    np.split(argsorted, np.nonzero(arr[1:] > arr[:-1])[0] + 1))

        # store only buckets with more than 1 user in it
        for position in range(len(bucket_array)):
            if len(bucket_array[position]) > 1:
                buckets.append(bucket_array[position])

    # make a list of unique buckets only
    # don't want too many or identical buckets
    buckets = list(set(map(tuple, buckets)))
    return buckets

def compute_candidate_pairs(signature_matrix, buckets):
    # establishing candidate pairs with t > 0.5
    candidate_pairs = set()
    for i in range(len(buckets)):
        # set for the combinations of pairs in a bucket
        user_pairs = set(pair for pair in itertools.combinations(buckets[i], 2))

        # new set of elements in user_pairs but not candidate_pairs
        user_pairs = user_pairs.difference(candidate_pairs)
        for j in user_pairs:
            sim = lsh_similarity(j[0], j[1], signature_matrix)
            if sim > 0.5:
                candidate_pairs.add(j)

    return candidate_pairs

def lsh_jaccard(sparse_matrix, lshset):
    """computes jaccard similarity for pairs found in lsh set.

    takes the unique set and computes true jaccard similarity for each pair.
    really, this is a check on our lsh function. as far as i can tell, the
    lsh function really should find lots of pairs with a small amount of false
    positives. this function eliminates false positives and duplicate pairs.
    however, this is not an effective way of combating the problem of false
    negatives. further work should be done to find them. or, multiple runs
    of the algorithm could alleviate things.

    args:
        sparse_matrix: our original csc matrix
                lshet: unique pairs set found by the lsh algorithm
    returns:
        results.txt file documenting found pairs

    """

    og_set = lshset # make a copy to avoid double counting
    lshset = sorted(lshset) # run through our set in order of users

    # have to search through relevent dense ndarray from sparse array
    # since &, | operands do not work on sparse matrices; and,
    # todense() returns only a representation of the sparse matrix
    dense_matrix = sparse_matrix.toarray()
    for pair in lshset:
        if pair[0] < pair[1]:
            jsim = jaccard_similarity(pair[0], pair[1], dense_matrix)
            if jsim > 0.5:
                write_to_file(pair[0], pair[1], jsim)
                # print(f'user-pair found: {pair[0]}, {pair[1]}')
        elif pair[1] < pair[0]:
            # don't double count pairs
            if (pair[1], pair[0]) in og_set:
                continue
            else:
                jsim = jaccard_similarity(pair[0], pair[1], dense_matrix)
                if jsim > 0.5:
                    # always have user1 id < user 2 id
                    write_to_file(pair[1],pair[0], jsim)
                else:
                    continue

def write_to_file(user1, user2, jsim):
    """write pairs to .txt list"""

    with open('results.txt', 'a') as f:
        f.write(f'{user1}    {user2}    {jsim}\n')
    f.close()

def main():
    code_start = timer()
    seed = int(sys.argv[1]) # read in random seed from user
    path = sys.argv[2] # read in specified path from user
    print(f'user random seed: {seed}')
    print(f'user path: {path}')

    # create our csc matrix and find max number of users and movies
    print('creating a compressed sparse column (csc) matrix from '\
          'user-movie data.')
    sparse_matrix, num_users, num_movies = load_data(path)
    print('csc matrix made! please wait a moment for '\
          'candidate pairs to be found...')

    print('utilizing minhashing function to create a signature matrix.')
    # work with a copy because we'll be comparing to computed jaccard
    # similarity for user pairs later on, to eliminate false positives
    # and double-counted pairs
    spm = sparse_matrix

    # create a signature for the sparse matrix and set number of
    # permutations
    sig_m, permutations = make_signatures(sparse_matrix, num_users,
                                          num_movies, seed)
    print('signature matrix is made. beginning to employ the lsh algorithm.')

    # run lsh algorithm to find unique pairs
    # make sure to keep banding technique between 50-150 long signatures
    buckets = lsh(sig_m)
    print('lsh algorithm implemented! we now compute a set of candidate pairs.')

    lshset = compute_candidate_pairs(sig_m, buckets)

    print('now we compute the jaccard similarities for the pairs in this set' \
          ' to ensure similarity and then write to results.txt')
    # run through unique pairs and find true jaccard similarity
    lsh_jaccard(spm, lshset)

    code_end = timer()
    run = (code_end - code_start) / 60
    run_time = np.round(run, decimals = 1)
    print(f'script completed in {run_time} minutes.')

if __name__ == '__main__':
    sys.exit(main())
