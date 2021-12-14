import time
from pyspark import SparkContext
import os
from pyspark.sql import SparkSession
from graphframes import GraphFrame
import sys
import csv

# os.environ['PYSPARK_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'
os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12")


def edge_cal(vertax, vertax_info, theta):
    edge_info = []
    uid = vertax[0]
    u_set = set(vertax[1])

    for v in vertax_info:
        vid = v[0]
        v_set = set(v[1])

        if uid != vid:
            if len(u_set & v_set) >= theta:
                edge_info.append(vid)
    return tuple([uid, edge_info])


if __name__ == '__main__':
    theta = int(sys.argv[1])
    input_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    sc = SparkContext().getOrCreate()
    sc.setLogLevel("ERROR")
    spark = SparkSession(sc)

    start = time.time()
    raw_rdd = sc.textFile(input_file_path)
    file_title = raw_rdd.first()
    rdd = raw_rdd.filter(lambda row: row != file_title).map(
        lambda x: (x.split(',')[0], x.split(',')[1])).distinct()

    graph_rdd = rdd.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x+y)\
        .map(lambda x: (x[0], list(set(x[1])))).persist()

    v_info = graph_rdd.collect()

    g_rdd = graph_rdd.map(lambda x: edge_cal(x, v_info, theta)).filter(lambda x: len(x[1]) > 0).persist()
    # get the edges
    edge_lst = g_rdd.flatMap(lambda x: [(x[0], val) for val in x[1]])\
        .filter(lambda x: x[0] < x[1]).distinct().toDF(['src', 'dst'])
    # print(edge_lst)
    # print(len(edge_lst))

    v_lst = g_rdd.map(lambda x: (x[0],)).distinct().toDF(['id'])

    print(edge_lst, v_lst)
    # generate graphs
    g = GraphFrame(v_lst, edge_lst)
    result = g.labelPropagation(maxIter=5)

    # find the final community
    community = result.rdd.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x+y).\
        map(lambda x: (sorted(x[1]))).sortBy(lambda x: (len(x), x[0])).collect()

    end = time.time()

    print(end - start)

    # write the final community as output
    with open(output_file_path, 'w+') as f:
        for a in community:
            f.write(str(a)[1:-1] + '\n')



