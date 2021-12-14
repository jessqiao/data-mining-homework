from pyspark import SparkContext
import os
import json
import sys

# os.environ['PYSPARK_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'


# input_file_path = '/Users/xinmengqiao/Desktop/DSCI 553/Homework/HW1/test_review.json'


def task_1(reviewRDD):
    res = {}

    # Part A:
    counts = reviewRDD.count()

    # Part B:
    counts_yr = reviewRDD.map(lambda rev: (rev['review_id'], rev['date']))\
        .filter(lambda a: int(a[1][0:4]) == int(2018)).count()

    # Part C:
    counts_user = reviewRDD.map(lambda rev: (rev['user_id'], 1)).reduceByKey(lambda a, b: 1).count()

    # Part D:
    top_user_10 = reviewRDD.map(lambda rev: (rev['user_id'], 1)).reduceByKey(lambda a, b: a + b) \
        .takeOrdered(10, lambda x: (-x[1], x[0]))

    # Part E:
    counts_business = reviewRDD.map(lambda rev: (rev['business_id'], 1)).reduceByKey(lambda a, b: 1).count()

    # Part F:
    top_business_10 = reviewRDD.map(lambda rev: (rev['business_id'], 1)).reduceByKey(lambda a, b: a + b) \
        .takeOrdered(10, lambda x: (-x[1], x[0]))


    res['n_review'] = counts
    res['n_review_2018'] = counts_yr
    res['n_user'] = counts_user
    res['top10_user'] = top_user_10
    res['n_business'] = counts_business
    res['top10_business'] = top_business_10

    return res


if __name__ == "__main__":
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    # read data
    sc = SparkContext().getOrCreate()
    reviewRDD = sc.textFile(input_file_path).map(lambda rev: json.loads(rev))

    results = task_1(reviewRDD)

    with open(output_file_path, 'w+') as outfile:
        json.dump(results, outfile)
