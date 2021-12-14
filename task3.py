from pyspark import SparkContext
import os
import json
import time
import sys

# os.environ['PYSPARK_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'




def adds(a, b):
    return (a[0]+b[0], a[1]+b[1])

def divs(a):
    return a[0] / a[1]



# input_file_path_review = '/Users/xinmengqiao/Desktop/DSCI 553/Homework/HW1/test_review.json'
# input_file_path_business = '/Users/xinmengqiao/Desktop/DSCI 553/Homework/HW1/business.json'


def partA(reviewRDD,businessRDD):
    businessRDD_city = businessRDD.map(lambda x: (x['business_id'], x['city']))

    data_star = reviewRDD.map(lambda rev: (rev['business_id'], rev['stars'])).leftOuterJoin(businessRDD_city)

    data_city = data_star.map(lambda a: (a[1][1], ((a[1][0]), 1)))\
        .reduceByKey(adds)\
        .mapValues(divs).collect()

    return(data_city)

def partB(data_city):

    # Python sort
    start = time.time()
    python_sort = sorted(data_city, key=lambda x: (-x[1], x[0]))
    print(python_sort[:10])
    end = time.time()
    exe_time_m1 = end-start



    # method 2 spark sort
    data_cityRDD = sc.parallelize(data_city)
    start = time.time()
    spark_sort = data_cityRDD.takeOrdered(10, lambda x: (-x[1], x[0]))
    print(spark_sort)
    end = time.time()
    exe_time_m2 = end-start
    # print(spark_sort, 'exe_time', exe_time)

    res = {}
    res['m1'] = exe_time_m1
    res['m2'] = exe_time_m2
    res['reason'] = "When the dataset is small enough that can be saved in main memory, python sort will " \
                    "outperform the spark sort since there will be no data transfer"

    return res
if __name__ == "__main__":
    input_file_path_review = sys.argv[1]
    input_file_path_business = sys.argv[2]
    output_file_path_a = sys.argv[3]
    output_file_path_b = sys.argv[4]


    # read data
    sc = SparkContext().getOrCreate()
    sc.setLogLevel("ERROR")

    reviewRDD = sc.textFile(input_file_path_review).map(lambda rev: json.loads(rev))
    businessRDD = sc.textFile(input_file_path_business).map(lambda x: json.loads(x))

    data_city = partA(reviewRDD, businessRDD)
    data_city_sorted = sorted(data_city, key=lambda x: (-x[1], x[0]))

    exe_time = partB(data_city)

    text_file = open(output_file_path_a, 'w')
    text_file.write('city'+','+'stars'+"\n")
    for city in data_city_sorted:
        text_file.write(city[0]+','+str(city[1])+"\n")
    text_file.close()

    with open(output_file_path_b, 'w+') as outfile:
        json.dump(exe_time, outfile)


