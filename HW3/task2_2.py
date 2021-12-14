import pandas as pd
import time
from pyspark import SparkContext
import json
import datetime as dt
from xgboost import XGBRegressor
import csv
import sys

# folder_path = '/Users/xinmengqiao/Desktop/DSCI 553/Homework/HW3_P2/HW3StudentData/'


def adds(a, b):
    return (a[0]+b[0], a[1]+b[1])


def divs(a):
    return a[0] / a[1]


def calc_days(join_date):
    tod = dt.date.today()
    join_date = dt.datetime.strptime(join_date, '%Y-%m-%d').date()
    diff = (tod - join_date).days
    return diff


def calc_std_u(user):
    mean = user_avg_rating[user[0]]
    value = (user[2] - mean) ** 2
    return value


def calc_std_b(item):
    mean = biz_avg_rating[item[1]]
    value = (item[2] - mean) ** 2
    return value


def generate_features(row):
    user_id = row[0]
    biz_id = row[1]
    # rating = row[2]

    b_info = biz_info_dict.get(biz_id, (biz_star_avg, biz_review_avg))
    u_info = user_info_dict.get(user_id, (user_review_avg, 3, 0, 0))

    # u_mean = user_avg_rating.get(user_id, global_avg)
    # u_std = user_std_rating.get(user_id, global_std)
    # b_mean = biz_avg_rating.get(biz_id, global_avg)
    # b_std = biz_std_rating.get(biz_id, global_std)

    checks = check_in.get(biz_id, 0)
    photo = photos.get(biz_id, 0)

    all_features = (checks, photo) + b_info + u_info
    return all_features


if __name__ == '__main__':
    folder_path = sys.argv[1]
    test_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    sc = SparkContext().getOrCreate()
    sc.setLogLevel("ERROR")

    start = time.time()

    training_set = sc.textFile(folder_path + '/yelp_train.csv')
    testing_set = sc.textFile(test_file_path)
    train_file_title = training_set.first()
    test_file_title = testing_set.first()
    train_rdd = training_set.filter(lambda row: row != train_file_title)\
        .map(lambda x: (x.split(',')[0], x.split(',')[1], float(x.split(',')[2]))).persist()
    test_rdd = testing_set.filter(lambda row: row != test_file_title)\
        .map(lambda x: (x.split(',')[0], x.split(',')[1]))

    # business information with bid as key, calculate some means for missing values
    biz_info = sc.textFile(folder_path + '/business.json').map(lambda x: json.loads(x)) \
        .map(lambda x: (x['business_id'], (x['stars'], x['review_count'])))

    biz_star_avg = biz_info.map(lambda x: x[1][0]).mean()
    biz_review_avg = biz_info.map(lambda x: x[1][1]).mean()

    biz_info_dict = biz_info.collectAsMap()

    # user information with user_id as key, calculate some means for missing values
    user_info = sc.textFile(folder_path + '/user.json').map(lambda x: json.loads(x)) \
        .map(lambda x: (x['user_id'], (x['review_count'], x['average_stars'], x['fans'],
                                       x['useful']+x['funny']+x['cool'])))

    user_review_avg = user_info.map(lambda x: x[1][0]).mean()
    user_fan_avg = 0
    user_reaction_avg = 0

    user_info_dict = user_info.collectAsMap()

    # check-in number
    check_in = sc.textFile(folder_path + '/checkin.json').map(lambda x: json.loads(x)).\
        map(lambda x: (x['business_id'], sum(list(x['time'].values())))).collectAsMap()

    # photo number
    photos = sc.textFile(folder_path + '/photo.json').map(lambda x: json.loads(x)).\
        map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x+y).collectAsMap()

    # user rating avg and std
    user_avg_rating = train_rdd.map(lambda x: (x[0], (float(x[2]), 1))).reduceByKey(adds).mapValues(divs).collectAsMap()
    user_std_rating = train_rdd.map(lambda x: (x[0], (calc_std_u(x), 1))).reduceByKey(adds).mapValues(divs).collectAsMap()

    # business rating avg and std
    biz_avg_rating = train_rdd.map(lambda x: (x[1], (float(x[2]), 1))).reduceByKey(adds).mapValues(divs).collectAsMap()
    biz_std_rating = train_rdd.map(lambda x: (x[1], (calc_std_b(x), 1))).reduceByKey(adds).mapValues(divs).collectAsMap()

    # global rating average/std
    global_avg = train_rdd.map(lambda x: (float(x[2]))).mean()
    global_std = train_rdd.map(lambda x: (float(x[2]))).stdev()

    # combine features
    training_set = train_rdd.map(lambda x: generate_features(x)+(x[2],)).collect()
    testing_set = test_rdd.map(lambda x: generate_features(x)+(x[0], x[1])).collect()
    # print(training_set[:10])
    train_df = pd.DataFrame(training_set)
    test_df = pd.DataFrame(testing_set)

    # train model using XGBRegressor
    X = train_df.iloc[:, :-1].values
    y = train_df.iloc[:, -1].values

    xgb = XGBRegressor(
        n_estimators=200,
        learning_rate=0.025,
        max_depth=5,
        random_state=100
    )

    # use trained model to predict the test dataset
    xgb.fit(X, y)
    res = xgb.predict(test_df.iloc[:, :-2].values)

    with open(output_file_path, 'w') as out:
        writer = csv.writer(out)
        writer.writerow(["user_id", "business_id", "prediction"])
        for i in range(len(res)):
            writer.writerow((testing_set[i][-2], testing_set[i][-1], res[i]))
    out.close()

    end = time.time()
    print('time', end-start)

