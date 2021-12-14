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

    for i in range(n):
        a = random.randint(100, 1000)
        b = random.randint(200, 2000)
        bit_len = 10
        hash_functions.append(tuple([a, b, bit_len]))
    return hash_functions


def myhashs(s):
    x = int(binascii.hexlify(s.encode('utf8')), 16)
    hashed_lst = []
    for para in parameters:
        a = para[0]
        b = para[1]
        bit_len = para[2]

        hashed = (a * x + b) % ((2 ** bit_len)-1)
        hashed_lst.append(hashed)

    return hashed_lst


def calc_trailing_zero(num):
    count = 0
    bit = bin(num)[2:]
    bit = bit[::-1]

    for b in bit:
        if b == '0':
            count += 1
        else:
            break
    return count


def flajolet_martin(batch_data):
    all_zeros = []
    for user in batch_data:
        hashed_lst = myhashs(user)
        trailing_zeros = [calc_trailing_zero(x) for x in hashed_lst]
        all_zeros.append(trailing_zeros)

    max_trailing_zeros = list(map(max, zip(*all_zeros)))
    predictions = [2 ** num_zero for num_zero in max_trailing_zeros]

    means = []
    for i in range(5):
        start = i * 10
        end = i * 10 + 11
        avg = sum(predictions[start:end]) / 10
        means.append(avg)
    median = means[2]
    median = int(median)
    ground_truth = len(set(stream_users))

    return tuple([median, ground_truth])


if __name__ == '__main__':
    file_name = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file_path = sys.argv[4]

    start = time.time()
    num_hash = 50

    parameters = generate_parameters(num_hash)

    res = []
    for i in range(num_of_asks):
        stream_users = bx.ask(file_name, stream_size)

        a = flajolet_martin(stream_users)
        res.append(a)

    with open(output_file_path, 'w') as out:
        writer = csv.writer(out)
        writer.writerow(['Time', 'Ground Truth', 'Estimation'])
        for idx, rs in enumerate(res):
            writer.writerow((idx, rs[1], rs[0]))
    out.close()

    end = time.time()
    print(end - start)

    sum_pred = sum([x[0] for x in res])
    sum_tru = sum([x[1] for x in res])
    print(sum_pred / sum_tru)

