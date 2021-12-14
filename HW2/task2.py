import time

from pyspark import SparkContext
import os
from itertools import combinations
import math
import sys
import csv

os.environ['PYSPARK_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'
#
# input_file_path = '/Users/xinmengqiao/Desktop/DSCI 553/Homework/HW2/ta_feng_all_months_merged.csv'
intermediate_path = 'pre-process.csv'
# output_file_path = '/Users/xinmengqiao/Desktop/DSCI 553/Homework/HW2/task2_output.csv'
# filter_threshold = 20
# support = 50


def find_candidate_singletons(candidate_items):
    candidate_singletons = set()
    for item in candidate_items:
        for ele in item:
            candidate_singletons.add(ele)
    return candidate_singletons


def find_candidate_elements(basket, candidate_element):
    new = set()
    for ele in candidate_element:
        if ele in set(basket):
            new.add(ele)
    return new


def find_frequent_kplet(data_collection, prev_candidate_element, k, threshold):
    new_candidate = {}
    new_frequent_items = set()
    for basket in data_collection:
        new_candidate_elements = find_candidate_elements(basket, prev_candidate_element)
        for k_plet in combinations(sorted(new_candidate_elements), k):
            new_candidate[k_plet] = new_candidate.get(k_plet, 0) + 1
            if new_candidate[k_plet] >= threshold:
                new_frequent_items.add(k_plet)
    return new_frequent_items


def apriori(data_collection, support, total_baskets):
    # list to store all candidate item sets
    all_frequent = []

    data_collection = list(data_collection)
    # print(data_collection)
    subset_len = len(data_collection)
    threshold = math.ceil((subset_len / total_baskets) * support)
    # print('threshold', threshold)

    # First pass to count and generate frequent singletons first
    items = {}
    frequent_items = set()
    for basket in data_collection:
        for item in basket:
            items[item] = items.get(item, 0) + 1
            if items[item] >= threshold:
                frequent_items.add(tuple([item]))

    all_frequent.extend(sorted(frequent_items))

    # candidate singletons for k=2
    candidate_element = find_candidate_singletons(frequent_items)

    # call function to calculate frequent k-plet, starting with k = 2
    k = 2

    prev_candidate_element = candidate_element
    new_frequent = frequent_items

    while new_frequent:
        new_frequent = find_frequent_kplet(data_collection, prev_candidate_element, k, threshold)
        prev_candidate_element = find_candidate_singletons(new_frequent)
        k += 1
        # print(new_frequent)
        all_frequent.extend(sorted(new_frequent))

    # print(all_frequent)

    return all_frequent


def son_second_pass(data_chunk, first_pass_candidate):
    check_count = {}
    for basket in data_chunk:
        for freq_cand in first_pass_candidate:
            if set(freq_cand).issubset(set(basket)):
                check_count[freq_cand] = check_count.get(freq_cand, 0) + 1

    final_freq = []
    for key in check_count:
        final_freq.append((key, check_count[key]))
    return final_freq


def format_result(input_list):
    output_list = []
    temp_list = []
    cur_len = len(input_list[0])
    for item in input_list:
        if len(item) > cur_len:
            cur_len = len(item)
            output_list.append(','.join(temp_list))
            output_list.append('\n\n')
            temp_list=[]
        if cur_len == 1:
            temp_list.append('(\'%s\')' % item[0])
        else:
            temp_list.append(str(item))
    output_list.append(','.join(temp_list))
    return ''.join(output_list)


def generate_processed_data(process_data, intermediate_path):
    with open(intermediate_path, 'w+') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(['DATE-CUSTOMER_ID', 'PRODUCT_ID'])
        for item in process_data:
            writer.writerow([item[0], item[1]])



if __name__ == '__main__':
    filter_threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    sc = SparkContext().getOrCreate()
    sc.setLogLevel("ERROR")

    raw_rdd = sc.textFile(input_file_path)
    file_title = raw_rdd.first()
    rdd = raw_rdd.filter(lambda row: row != file_title).map(lambda x: [x.split(',')[0].strip('"') \
        , x.split(',')[1].strip('"').lstrip('0'), x.split(',')[5].strip('"').lstrip('0')]) \
        .map(lambda x: ((x[0]+'-'+x[1]), x[2]))

    # print(rdd.take(20))
    process_data = rdd.collect()

    generate_processed_data(process_data, intermediate_path)
    start = time.time()
    processed_rdd = sc.textFile(intermediate_path)
    header = processed_rdd.first()

    input_rdd = processed_rdd.filter(lambda row: row != header).map(lambda x: (x.split(',')[0], [x.split(',')[1]])) \
        .reduceByKey(lambda x, y: x + y).map(lambda x: list(set(x[1]))) \
        .filter(lambda x: len(x) > filter_threshold)

    # print(input_rdd[:10])

    total_len = input_rdd.count()

    # first pass candidate item sets
    first_pass_can = input_rdd.mapPartitions(lambda x: apriori(x, support, total_len)) \
        .distinct().sortBy(lambda x: (len(x), x)).collect()
    # print(first_pass_can)
    candidates = format_result(first_pass_can)

    # second pass check frequent items
    second_pass_freq = input_rdd.mapPartitions(lambda x: son_second_pass(x, first_pass_can)) \
        .reduceByKey(lambda x, y: x + y).filter(lambda x: x[1] >= support) \
        .sortBy(lambda x: (len(x[0]), x[0])).map(lambda x: x[0]).collect()

    # print(second_pass_freq)
    frequent_itemsets = format_result(second_pass_freq)

    with open(output_file_path, 'w+') as output:
        output.write('Candidates:\n')
        output.write(candidates)
        output.write('\n\n')
        output.write('Frequent Itemsets:\n')
        output.write(frequent_itemsets)

    end = time.time()

    duration = end - start
    print('duration:', duration)