from pyspark import SparkContext, SparkConf
import sys
import json
from itertools import combinations
import math
import time

train_file = 'train_review.json'
model_file = 'task3user.model'
cf_type = 'user_based'

ITEM_BASED = 'item_based'
USER_BASED = 'user_based'

signature_len = 35
band_n = 35

def pearson_cor(iterator, busi_dict):
    for b_id1, b_id2 in iterator:
        b1_dict = busi_dict[b_id1]
        b2_dict = busi_dict[b_id2]
        co_rated = set(b1_dict.keys()) & set(b2_dict.keys())
        r1_corated_total = 0
        r2_corated_total = 0
        for key in co_rated:
            r1_corated_total += b1_dict[key]
            r2_corated_total += b2_dict[key]
        r1_ave = r1_corated_total / len(co_rated)
        r2_ave = r2_corated_total / len(co_rated)
        numerator = 0
        dnmnt_b1 = 0
        dnmnt_b2 = 0
        for cor in co_rated:
            numerator += (b1_dict[cor] - r1_ave) * (b2_dict[cor] - r2_ave)
            dnmnt_b1 += pow((b1_dict[cor] - r1_ave), 2)
            dnmnt_b2 += pow((b2_dict[cor] - r2_ave), 2)

        denominator = math.sqrt(dnmnt_b1 * dnmnt_b2)
        if denominator != 0:
            pearson = numerator / denominator
            if 0 < pearson <= 1:
                yield (b_id1, b_id2, pearson)

def minhash(iterator, signature_len, biz_n):
    for uid, biz_list in iterator:
        signature = []
        for i in range(1, signature_len + 1):
            min_value = biz_n
            for biz_index in biz_list:
                # (2749 * biz_index + 5323 + i * ((937 * biz_index + 4093) % 97))
                min_value = min(min_value, ((49+i)*biz_index+323) % biz_n)
            signature.append(min_value)
        yield (uid, signature)

def lsh(iterator, signature_len, band_n):
    r = signature_len / band_n
    for uid, signature in iterator:
        for b in range(band_n):
            sig_band = signature[int(b * r):int((b + 1) * r)]
            value = ''
            for sig in sig_band:
                value += str(sig)
            yield ((b, value), uid)

def jaccard(iterator, user_dict):
    for u1, u2 in iterator:
        b1_list = user_dict[u1].keys()
        b2_list = user_dict[u2].keys()
        inter = set(b1_list) & set(b2_list)
        if len(inter) >= 3:
            union = set(b1_list) | set(b2_list)
            sim = len(inter)/len(union)
            if sim >= 0.01:
                yield (u1, u2)

conf = SparkConf().setAppName('inf553_hw3_task3').setMaster('local[*]')
sc = SparkContext(conf=conf)
start_time = time.time()

if cf_type == ITEM_BASED:
    review = sc.textFile(train_file) \
        .map(lambda s: json.loads(s)) \
        .persist()

    business_rate = review.map(lambda s: ((s['business_id'], s['user_id']), s['stars'])) \
        .aggregateByKey((0, 0), lambda u, v: (u[0] + v, u[1] + 1), lambda U, V: (U[0] + V[0], U[1] + V[1])) \
        .mapValues(lambda v: v[0] / v[1]) \
        .map(lambda s: (s[0][0], (s[0][1], s[1]))) \
        .groupByKey() \
        .mapValues(list) \
        .map(lambda s: (s[0], dict(s[1]))) \
        .collect()
    business_dict = dict(business_rate)

    business_pairs = review.map(lambda s: (s['user_id'], s['business_id'])) \
        .distinct() \
        .groupByKey() \
        .mapValues(list) \
        .filter(lambda s: len(s[1]) > 1) \
        .flatMap(lambda s: combinations(sorted(s[1]), 2)) \
        .map(lambda s: ((s[0], s[1]), 1)) \
        .reduceByKey(lambda u, v: u + v) \
        .filter(lambda s: s[1] >= 3) \
        .map(lambda s: s[0]) \
        .mapPartitions(lambda s: pearson_cor(s, business_dict)) \
        .map(lambda s: dict((('b1', s[0]), ('b2', s[1]), ('sim', s[2])))) \
        .collect()

    print(len(business_pairs))
    # ground truth pairs: 55,7683
    with open(model_file, 'w') as f:
        for piece in business_pairs:
            f.write(json.dumps(piece) + '\n')

    print('Duration: ', time.time() - start_time)

elif cf_type == USER_BASED:
    review = sc.textFile(train_file) \
        .map(lambda s: json.loads(s)) \
        .persist()

    # transform users into indices
    bizs = review.map(lambda s: s['business_id'])\
        .distinct()\
        .zipWithIndex()\
        .collect()
    bizs_dict = dict(bizs)

    # build a dict with user_id and business_id mapping to ratings
    ratings = review.map(lambda s: ((s['user_id'], s['business_id']), s['stars'])) \
        .aggregateByKey((0, 0), lambda u, v: (u[0] + v, u[1] + 1), lambda U, V: (U[0] + V[0], U[1] + V[1])) \
        .mapValues(lambda v: v[0] / v[1]) \
        .map(lambda s: (s[0][0], (s[0][1], s[1])))\
        .persist()
    rating_dict = dict(ratings.groupByKey()\
                       .mapValues(list)\
                       .map(lambda s: (s[0], dict(s[1])))\
                       .collect())

    # process minhash, LSH
    # calculate Jaccard similarity and Pearson correlation
    usr_jaccard = ratings.map(lambda s: (s[0], bizs_dict[s[1][0]]))\
        .groupByKey()\
        .mapValues(list)\
        .mapPartitions(lambda s: minhash(s, signature_len, len(bizs_dict)))\
        .mapPartitions(lambda s: lsh(s, signature_len, band_n))\
        .groupByKey()\
        .mapValues(list)\
        .filter(lambda s: len(s[1]) > 1) \
        .flatMap(lambda s: list(combinations(sorted(s[1]), 2))) \
        .distinct() \
        .mapPartitions(lambda s: jaccard(s, rating_dict))\
        .mapPartitions(lambda s: pearson_cor(s, rating_dict)) \
        .map(lambda s: dict((('u1', s[0]), ('u2', s[1]), ('sim', s[2])))) \
        .collect()

    print(len(usr_jaccard))
    # ground truth pairs: 78,3235
    with open(model_file, 'w') as f:
        for piece in usr_jaccard:
            f.write(json.dumps(piece) + '\n')

    print('Duration: ', time.time() - start_time)