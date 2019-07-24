import numpy as np
from collections import defaultdict
from six import iteritems
import time
import json


def get_data():
    all_dict = defaultdict(list)
    for line in open(r'../train'):
        line = line.replace('\n', '')
        line = line.split(' ')
        user = line[0]
        items = line[1:]
        all_dict[user] += items

    for line in open(r'../dev'):
        line = line.replace('\n', '')
        line = line.split(' ')
        user = line[0]
        items = line[1:]
        all_dict[user] += items

    train_dict = dict()
    test_dict = dict()
    for idx, (k, v) in enumerate(iteritems(all_dict)):
        if not v:
            continue

        if len(v) > 1:
            rand_idx = np.random.randint(1, len(v))
        else:
            rand_idx = np.random.randint(0, len(v))

        train_dict[k] = {
            'train': v[:rand_idx],
            'label': v[rand_idx:]
        }

    return train_dict, test_dict, all_dict


def get_metadata():
    meta_dict = defaultdict()
    meta_dict.default_factory = meta_dict.__len__

    read_dict = defaultdict()
    read_dict.default_factory = meta_dict.__len__

    now_time = time.time() * 1000

    meta_read_dict = defaultdict()
    for line in open(r'../metadata.json'):
        try:
            line = json.loads(line)
            reads = line.get('keyword_list', [])
            keyword_list = []
            for read in reads:
                keyword_list.append(read_dict[read])

            item = line.get('id', '')
            item_key = meta_dict[item]

            reg = line.get('reg_ts', 0)
            meta_read_dict[item] = {'key': item_key, 'read': keyword_list, 'reg': round(reg / now_time, 4)}

        except Exception:
            continue

    return meta_dict, read_dict, meta_read_dict


def get_follow():
    follow_dict = defaultdict()
    follow_dict.default_factory = follow_dict.__len__

    for line in open(r'../users.json'):
        line = json.loads(line)
        following_list = line.get('following_list', [])
        for following in following_list:
            _ = follow_dict[following]

    follow_len = len(follow_dict)
    follow_user = defaultdict()

    for line in open(r'../users.json'):
        line = json.loads(line)
        user = line.get('id')
        following_list = line.get('following_list', [])
        if user:
            follow_user_list = []
            for following in following_list:
                follow_user_list.append(follow_dict[following])
            follow_user[user] = follow_user_list

    return follow_user, follow_len
