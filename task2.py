from pyspark import SparkContext
import os
import json
import time
import sys

# os.environ['PYSPARK_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'



# input_file_path = '/Users/xinmengqiao/Desktop/DSCI 553/Homework/HW1/test_review.json'


def top_business(func_type, reviewRDD, num_par='default'):
    if func_type == "default":
        startTime = time.time()
        length = reviewRDD.glom().map(len).collect()
        number_partition = reviewRDD.getNumPartitions()
        top_business_10 = reviewRDD.map(lambda rev: (rev['business_id'], 1)).reduceByKey(lambda a, b: a + b) \
            .takeOrdered(10, lambda x: (-x[1], x[0]))
        endTime = time.time()
        exe_time = endTime - startTime

    else:
        startTime = time.time()
        reviewRDD_cus = reviewRDD.map(lambda rev: (rev['business_id'], 1)).partitionBy(int(num_par), lambda x: ord(x[0]) + ord(x[-1]))
        number_partition = reviewRDD_cus.getNumPartitions()
        length = reviewRDD_cus.glom().map(len).collect()
        top_business_10 = reviewRDD_cus.reduceByKey(lambda a, b: a + b) \
            .takeOrdered(10, lambda x: (-x[1], x[0]))
        endTime = time.time()
        exe_time = endTime - startTime

    res = {}
    res['n_partition'] = number_partition
    res['n_items'] = length
    res['exe_time'] = exe_time
    return res


if __name__ == "__main__":
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    n_partitions = sys.argv[3]

    # read data
    sc = SparkContext().getOrCreate()
    reviewRDD = sc.textFile(input_file_path).map(lambda rev: json.loads(rev))

    results = {}
    results['default'] = top_business('default', reviewRDD)
    results['customized'] = top_business('customized', reviewRDD, n_partitions)


    with open(output_file_path, 'w+') as outfile:
        json.dump(results, outfile)


