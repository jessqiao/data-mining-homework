import random
import time
import csv
import os
import sys
import binascii
from blackbox import BlackBox
bx = BlackBox()


def generate_parameters(n):
    hash_functions = []

    p_lst = [23335597, 4460899]
    for i in range(n):
        a = random.randint(100, 1000)
        b = random.randint(200, 2000)
        p = p_lst[i]
        hash_functions.append(tuple([a, b, p]))
    return hash_functions


def myhashs(s):
    x = int(binascii.hexlify(s.encode('utf8')), 16)
    hashed_lst = []
    for para in parameters:
        a = para[0]
        b = para[1]
        p = para[2]

        hashed = ((a * x + b) % p) % m
        hashed_lst.append(hashed)

    return hashed_lst


def bloom_filter(batch_data):
    false_positive = 0

    for user in batch_data:
        alter_positions = myhashs(user)
        not_seen = False
        for pos in alter_positions:
            if bit_array[pos] == 0:
                not_seen = True
                bit_array[pos] = 1

        if (not not_seen) and (user not in ground_truth):
            false_positive += 1

    # print(false_positive, len(batch_data))
    return false_positive


if __name__ == '__main__':
    file_name = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file_path = sys.argv[4]

    start = time.time()
    m = 69997
    num_hash = 2

    ground_truth = set()
    bit_array = [0]*m

    parameters = generate_parameters(num_hash)
    # print(parameters)
    fpr_lst = []
    for i in range(num_of_asks):
        stream_users = bx.ask(file_name, stream_size)

        true_neg = len(set(stream_users) - ground_truth)
        false_pos = bloom_filter(stream_users)
        fpr = false_pos / true_neg

        ground_truth.update(set(stream_users))
        fpr_lst.append(fpr)
        # print(false_pos, true_neg)

    with open(output_file_path, 'w') as out:
        writer = csv.writer(out)
        writer.writerow(['Time', 'FPR'])
        for idx, rs in enumerate(fpr_lst):
            writer.writerow((idx, rs))
    out.close()

    end = time.time()

    print(end - start)
