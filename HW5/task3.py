import random
import time
import csv
import os
import sys
import binascii
from blackbox import BlackBox


if __name__ == '__main__':
    file_name = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file_path = sys.argv[4]

    start = time.time()

    s = 100

    random.seed(553)
    bx = BlackBox()

    fixed_size = []
    n = 0

    samplings = []
    for i in range(num_of_asks):
        stream_users = bx.ask(file_name, stream_size)
        for user in stream_users:
            n += 1
            if n <= s:
                fixed_size.append(user)
            else:
                p = random.random()
                if p <= s/n:
                    pos = random.randint(0, 99)
                    fixed_size[pos] = user
        samplings.append(tuple([n, fixed_size[0], fixed_size[20], fixed_size[40],
                                fixed_size[60], fixed_size[80]]))

    with open(output_file_path, 'w') as out:
        writer = csv.writer(out)
        writer.writerow(['seqnum', '0_id', '20_id', '40_id', '60_id', '80_id'])
        for res in samplings:
            writer.writerow(res)
    out.close()

    end = time.time()
    print(end - start)
    print(len(fixed_size))