from pyspark import SparkContext, SparkConf
import sys
import json
import time

train_file = 'train_review.json'
test_file = 'test_review.json'
model_file = 'task3item.model'
output_file = 'task3item.predict'
cf_type = 'item_based'

ITEM_BASED = 'item_based'
USER_BASED = 'user_based'
N = 7

def predict_itembased(iterator, rating_dict):
    for bid, uid, neighbors in iterator:
        numereator = 0
        denominator = 0
        n = 0
        for neighbor, sim in neighbors:
            if (uid, neighbor) in rating_dict.keys():
                numereator += rating_dict[(uid, neighbor)] * sim
                denominator += abs(sim)
                n += 1
                if n >= N:
                    break
        # if denominator == 0:
        #     pred_res = biz_avg_dict[bid]
        # else:
        if denominator != 0:
            pred_res = numereator/denominator
            yield (uid, bid, pred_res)

def predict_userbased(iterator, rating_dict, usr_avg_dict):
    for uid, bid, neighbors in iterator:
        numerator = 0
        denominator = 0
        n = 0
        for neighbor, sim in neighbors:
            usr_dict = rating_dict[neighbor]
            if bid in usr_dict.keys():
                numerator += (usr_dict[bid] - sum(usr_dict.values())/len(usr_dict)) * sim
                denominator += abs(sim)
                n += 1
                if n >= N:
                    break
        # if n < N-2 or denominator == 0:
        #     pred_res = usr_avg_dict[uid]
        # else:
        if n >= N and denominator != 0:
            pred_res = sum(rating_dict[uid].values())/len(rating_dict[uid]) + numerator/denominator
            if pred_res > 5.0:
                pred_res = 5.0
            yield (uid, bid, pred_res)


start_time = time.time()

conf = SparkConf().setAppName('inf553_hw3_task3predict').setMaster('local[*]').set('spark.driver.memory', '4G')
sc = SparkContext(conf=conf)

if cf_type == ITEM_BASED:

    # get ratings from the train data
    train_data = sc.textFile(train_file)\
        .map(lambda s: json.loads(s))\
        .map(lambda s: ((s['user_id'], s['business_id']), s['stars']))\
        .collect()
    rating_dict = dict(train_data)

    # build models
    # deal with models into 2 parts
    model1 = sc.textFile(model_file)\
        .map(lambda s: json.loads(s))\
        .map(lambda s: (s['b1'], (s['b2'], s['sim'])))\
        .persist()

    model2 = model1.map(lambda s: (s[1][0], (s[0], s[1][1])))\
        .persist()

    # load target pairs
    test_pairs = sc.textFile(test_file)\
        .map(lambda s: json.loads(s))\
        .map(lambda s: (s['business_id'], s['user_id']))\
        .persist()

    # combine target pairs with models together
    neighbors_part = test_pairs.join(model1).persist()

    # go predicting with the models
    pred = test_pairs.join(model2)\
        .union(neighbors_part)\
        .map(lambda s: ((s[0], s[1][0]), s[1][1])) \
        .groupByKey() \
        .mapValues(list) \
        .map(lambda s: (s[0][0], s[0][1], sorted(s[1], key=lambda x: x[1], reverse=True))) \
        .mapPartitions(lambda s: predict_itembased(s, rating_dict))\
        .map(lambda s: dict({'user_id': s[0], 'business_id': s[1], 'stars': s[2]}))\
        .collect()

    with open(output_file, 'w') as f:
        for piece in pred:
            f.write(json.dumps(piece) + '\n')
    print('Duration: ', time.time()-start_time)

elif cf_type == USER_BASED:

    # get ratings from the train data
    train_data = sc.textFile(train_file) \
        .map(lambda s: json.loads(s)) \
        .map(lambda s: (s['user_id'], (s['business_id'], s['stars']))) \
        .groupByKey()\
        .mapValues(list)\
        .map(lambda s: (s[0], dict(s[1])))\
        .collect()
    rating_dict = dict(train_data)

    # build models
    # deal with models into 2 parts
    model1 = sc.textFile(model_file) \
        .map(lambda s: json.loads(s)) \
        .map(lambda s: (s['u1'], (s['u2'], s['sim']))) \
        .persist()

    model2 = model1.map(lambda s: (s[1][0], (s[0], s[1][1]))) \
        .persist()

    # load target pairs
    test_pairs = sc.textFile(test_file) \
        .map(lambda s: json.loads(s)) \
        .map(lambda s: (s['user_id'], s['business_id'])) \
        .persist()

    # combine target pairs with models together
    neighbors_part = test_pairs.join(model1).persist()

    # go predicting with the models
    pred = test_pairs.join(model2) \
        .union(neighbors_part) \
        .map(lambda s: ((s[0], s[1][0]), s[1][1])) \
        .groupByKey() \
        .mapValues(list) \
        .map(lambda s: (s[0][0], s[0][1], sorted(s[1], key=lambda x: x[1], reverse=True))) \
        .mapPartitions(lambda s: predict_userbased(s, rating_dict)) \
        .map(lambda s: dict({'user_id': s[0], 'business_id': s[1], 'stars': s[2]})) \
        .collect()

    with open(output_file, 'w') as f:
        for piece in pred:
            f.write(json.dumps(piece) + '\n')
    print('Duration: ', time.time()-start_time)