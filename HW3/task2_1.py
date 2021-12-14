import time
from pyspark import SparkContext
import os
import numpy as np
import csv
import sys


# os.environ['PYSPARK_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/xinmengqiao/opt/anaconda3/envs/py36/bin/python'

# train_file_path = '/Users/xinmengqiao/Desktop/DSCI 553/Homework/HW3/publicdata/yelp_train.csv'
# test_file_path = '/Users/xinmengqiao/Desktop/DSCI 553/Homework/HW3/publicdata/yelp_test.csv'


def adds(a, b):
    return (a[0]+b[0], a[1]+b[1])


def divs(a):
    return a[0] / a[1]


def pearson_corr(item_1, item_2, users, ratings):
    item_1_rating = []
    item_2_rating = []
    for user in users:
        item_1_rating.append(float(ratings[(user, item_1)]))
        item_2_rating.append(float(ratings[(user, item_2)]))

    top = np.sum((item_1_rating - np.mean(item_1_rating)) * (item_2_rating - np.mean(item_2_rating)))
    bottom = np.sqrt(np.sum((item_1_rating - np.mean(item_1_rating)) ** 2) * np.sum((item_2_rating - np.mean(item_2_rating)) ** 2))

    if bottom != 0:
        pearson_sim = top / bottom
        return pearson_sim
    return 0


def find_candidate_similarity(all_cand_items, pred_user, pred_business, item_all_users_dict, all_similarity, ratings):
    candidate_similarity = []

    for candidate_item in all_cand_items:
        users_for_cand_item = set(item_all_users_dict[candidate_item])
        users_for_pred = set(item_all_users_dict[pred_business])

        # find the corated users
        corated_users = users_for_cand_item & users_for_pred
        # print(candidate_item, pred_business)
        # print(len(corated_users))

        if len(corated_users) < 3:
            continue
        if tuple(sorted([candidate_item, pred_business])) in all_similarity:
            sim = all_similarity[tuple(sorted([candidate_item, pred_business]))]
        else:
            sim = pearson_corr(candidate_item, pred_business, corated_users, ratings)
            # add case amplification
            sim = sim * abs(sim) ** (2.5 - 1)
            all_similarity[tuple(sorted([candidate_item, pred_business]))] = sim
        candidate_similarity.append((candidate_item, sim))

    return candidate_similarity


def predict_rating(prediction_pair, item_all_users_dict, user_all_items_dict, item_avg_rating, user_avg_rating, global_avg,
                   all_similarity, ratings):
    pred_user = prediction_pair[0]
    pred_business = prediction_pair[1]

    if pred_user not in user_all_items_dict.keys() and pred_business not in item_all_users_dict.keys():
        pred = global_avg
    elif pred_user not in user_all_items_dict.keys():
        pred = item_avg_rating[pred_business]
    elif pred_business not in item_all_users_dict.keys():
        pred = user_avg_rating[pred_user]
    else:
        all_candidates = user_all_items_dict[pred_user]
        all_simi = find_candidate_similarity(all_candidates, pred_user, pred_business, item_all_users_dict,
                                             all_similarity, ratings)
        if len(all_simi) >= 5:
            all_simi.sort(key=lambda x: x[1], reverse=True)
            final_cand = all_simi[:5]
            a, b = 0.0, 0.0
            for candi in final_cand:
                candidate = candi[0]
                similarity = float(candi[1])
                rating = float(ratings[(pred_user, candidate)])
                sim_rat = similarity * rating
                a += sim_rat
                b += similarity
            if b == 0 or b < 3:
                pred = item_avg_rating[pred_business]
            else:
                pred = a/abs(b)
                if pred > 5 or pred < 1:
                    pred = item_avg_rating[pred_business]
        else:
            pred = user_avg_rating[pred_user]
    return pred

if __name__ == '__main__':

    sc = SparkContext().getOrCreate()
    sc.setLogLevel("ERROR")

    start = time.time()

    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    training_set = sc.textFile(train_file_path)
    testing_set = sc.textFile(test_file_path)

    train_file_title = training_set.first()
    test_file_title = testing_set.first()
    train_rdd = training_set.filter(lambda row: row != train_file_title).map(lambda x: (x.split(',')[0], x.split(',')[1], float(x.split(',')[2]))).persist()
    test_rdd = testing_set.filter(lambda row: row != test_file_title).map(lambda x: (x.split(',')[0], x.split(',')[1]))

    # prepare useful dictionary for calculation \
    # user-to-item dictionary: it has all users as key, and their rated business as values
    user_all_items = train_rdd.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x+y) \
        .map(lambda x: (x[0], list(set(x[1])))).collect()
    user_all_items_dict = {}
    for item in user_all_items:
        user_all_items_dict[item[0]] = item[1]

    # item-to-user dictionary: it has all business_id as key, and people who rated the business as value
    item_all_users_dict = train_rdd.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x+y) \
        .map(lambda x: (x[0], list(set(x[1])))).collectAsMap()

    # rating dictionary: the (user_id, bid) tuple is the key, and ratings is the value
    ratings = train_rdd.map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()

    # AVG rating matrix for items
    item_avg_rating = train_rdd.map(lambda x: (x[1], (x[2], 1))).reduceByKey(adds).mapValues(divs).collectAsMap()
    # print(item_avg_rating)

    # AVG rating matrix for users
    user_avg_rating = train_rdd.map(lambda x: (x[0], (float(x[2]), 1))).reduceByKey(adds).mapValues(divs).collectAsMap()

    # Global avg for all ratings
    global_avg = train_rdd.map(lambda x: (float(x[2]))).mean()

    # similarity matrix: the sorted(bid1, bid2) is key, and pearson similarity is values
    all_similarity = {}

    # sample = test_rdd.take(1)[0]
    # a = predict_rating(sample)

    # prediction on the testing set
    res = test_rdd.map(lambda x: (x[0], x[1], predict_rating(x, item_all_users_dict, user_all_items_dict,
                                                             item_avg_rating, user_avg_rating, global_avg,
                                                             all_similarity, ratings))).collect()

    with open(output_file_path, 'w') as out:
        writer = csv.writer(out)
        writer.writerow(["user_id", "business_id", "prediction"])
        for rs in res:
            writer.writerow(rs)
    out.close()

    end = time.time()
    print(end - start)
