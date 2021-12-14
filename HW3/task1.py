import time

from pyspark import SparkContext
import os
import sys
import random
import itertools
import csv

# os.environ['PYSPARK_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'
# input_file_path = '/Users/xinmengqiao/Desktop/DSCI 553/Homework/HW3/publicdata/yelp_train.csv'


def row_hashing(a, b, m, p, idx):
    hashed = ((a*idx+b) % p) % m
    return hashed


def make_bins(num_bins, num_row, sig_list):
    hashed_bins = []
    start_idx = 0
    for bin_idx in range(num_bins):
        end_idx = start_idx + num_row
        lst = sig_list[start_idx:end_idx]
        start_idx = end_idx
        hash_value = hash(tuple(lst))
        hashed_bins.append((bin_idx, hash_value))
    return hashed_bins


def jaccard_distince(candidate_1, candidate_2):
    candidate_1 = set(candidate_1)
    candidate_2 = set(candidate_2)
    jac_dis = float(len(candidate_1 & candidate_2)) / float(len(candidate_1 | candidate_2))
    return jac_dis


if __name__ == '__main__':

    sc = SparkContext().getOrCreate()
    sc.setLogLevel("ERROR")

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    start = time.time()

    raw_rdd = sc.textFile(input_file_path)
    file_title = raw_rdd.first()
    rdd = raw_rdd.filter(lambda row: row != file_title).map(lambda x: (x.split(',')[0], x.split(',')[1])).distinct().persist()

    users = rdd.map(lambda x: x[0]).distinct().collect()

    n = 100
    p = 23333
    m = len(users)
    row = 2
    bins = 50

    # build user index
    user_idx = {}
    for idx, user in enumerate(users):
        user_idx[user] = idx

    user_hashed = {}

    # For each row, generate 100 hashing results to generate the signature matrix
    a = random.randint(1000, 100000)
    b = random.randint(1000, 100000)

    for user in user_idx:
        user_hashed[user] = [row_hashing(a, b, m, p, user_idx[user])]

    for i in range(1, n):
        a = random.randint(1000, 100000)
        b = random.randint(1000, 100000)

        for user in user_idx:
            user_hashed[user].append(row_hashing(a, b, m, p, user_idx[user]))

    # print(user_hashed)

    # generate signature matrix
    full_matrix = rdd.map(lambda x: (x[1], x[0])).map(lambda x: (x[0], [user_hashed[x[1]]])) \
        .reduceByKey(lambda x, y: x+y)

    sig_matrix = full_matrix.map(lambda x: (x[0], [min(u_idx) for u_idx in zip(*x[1])])) \
        .sortBy(lambda x: x[0])

    # print(sig_matrix[:3])

    sample = sig_matrix.collect()[1][1]
    # print(sample)

    sample_bins = make_bins(bins, row, sample)
    # print(sample_bins)

    # divide sig_matrix into bands. Use band_idx, hashed_value as index, business_id as value,
    # group by key, filter those business id that matches others

    band_sig_matrix = sig_matrix.map(lambda x: (x[0], make_bins(bins, row, x[1]))) \
        .flatMap(lambda x: [(values, [x[0]]) for values in x[1]])

    candidate_pairs = band_sig_matrix.reduceByKey(lambda x, y: x+y).filter(lambda x: len(x[1]) > 1) \
        .flatMap(lambda x: [candi for candi in itertools.combinations(x[1], 2)]) \
        .distinct().collect()
    # print('fml', candidate_pairs)

    verify_rdd = rdd.map(lambda x: (x[1], [user_idx[x[0]]])).reduceByKey(lambda x, y: x+y) \
        .map(lambda x: (x[0], list(set(x[1])))).collectAsMap()

    final_output = set()
    for pairs in candidate_pairs:
        if pairs not in final_output:
            sim = jaccard_distince(verify_rdd[pairs[0]], verify_rdd[pairs[1]])
            if sim >= 0.5:
                final_output.add((pairs + (sim,)))

    final_output = sorted(list(final_output), key=lambda x: (x[0], x[1]))

    with open(output_file_path, 'w') as out:
        writer = csv.writer(out)
        writer.writerow(["business_id_1", "business_id_2", "similarity"])
        for rs in final_output:
            writer.writerow(rs)
    out.close()

    end = time.time()
    print(end - start)

