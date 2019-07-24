from tqdm import tqdm
from popular_model.loader import get_metadata, get_data, get_follow
from popular_model.recommender_model import PopularModel
import numpy as np


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')

    batch_size = 50
    train_data, test_data, all_data = get_data()
    meta_data, read_data, meta_read_data = get_metadata()
    follow_data, follow_len = get_follow()

    meta_data_len = len(meta_data)
    read_data_len = len(read_data)
    model = PopularModel(batch_size, meta_data_len, read_data_len, follow_len)


    def max_batch_func(batch_data: list):
        max_len = 1
        for batch_datum in batch_data:
            if max_len < len(batch_datum):
                max_len = len(batch_datum)

        batch = []
        for batch_datum in batch_data:
            batch_pad = np.array(np.pad(np.array(batch_datum), (0, max_len - len(batch_datum)), mode='constant') == 0) * meta_data_len
            batch_value = np.pad(np.array(batch_datum), (0, max_len - len(batch_datum)), mode='constant')
            batch_value += batch_pad
            batch.append(batch_value)

        return batch


    def train_func():
        cnt = 0
        sum_predict = []
        train_value_batch = []
        train_keyword_batch = []
        label_value_batch = []
        follow_value_batch = []
        for k, v in tqdm(train_data.items()):
            if not (v.get('train') and v.get('label')):
                continue

            train_value_list = []
            train_keyword_list = []
            for train_value in v.get('train'):
                if train_value in meta_data:
                    train_value_idx = meta_read_data[train_value].get('key')
                    train_value_list.append(train_value_idx)
                    train_keyword_list += meta_read_data[train_value].get('read')
            train_value_list = list(set(train_value_list))

            if not train_value_list:
                train_value_list.append(meta_data_len)
            if not train_keyword_list:
                train_keyword_list.append(meta_data_len)

            label_value_list = []
            reg_value_list = []
            for label_value in v.get('label'):
                if label_value in meta_data:
                    label_value_idx = meta_read_data[label_value].get('key')
                    reg_value = meta_read_data[label_value].get('reg')
                    label_value_list.append(label_value_idx)
                    reg_value_list.append(reg_value)
            if not reg_value_list:
                reg_value_list.append(0)

            follow_value_list = []
            if k in follow_data:
                follow_value_list += follow_data[k]
            else:
                follow_value_list.append(follow_len)

            reg_mean = np.mean(reg_value_list, dtype=np.float16)
            labels_mean = np.array(np.bincount(label_value_list, minlength=meta_data_len) >= 1,
                              dtype=np.float64) * 0.9 + 0.05
            labels_mean = labels_mean * reg_mean

            label_value_batch.append(labels_mean)
            train_value_batch.append(train_value_list)
            train_keyword_batch.append(train_keyword_list)
            follow_value_batch.append(follow_value_list)

            if len(train_value_batch) == batch_size:
                inputs = max_batch_func(train_value_batch)
                labels = label_value_batch
                keywords = max_batch_func(train_keyword_batch)
                follows = max_batch_func(follow_value_batch)
                model.model_train(inputs, keywords, labels, follows)

                if cnt % 10 == 0:
                    result = model.predict(inputs, keywords, follows)
                    sum_predict.append((np.sum(labels * result) / np.sum(labels)))

                if cnt % 100 == 0:
                    sum_predict_result = np.mean(sum_predict)
                    print(sum_predict_result)
                    sum_predict = []

                label_value_batch = []
                train_value_batch = []
                train_keyword_batch = []
                follow_value_batch = []
                cnt += 1


    for _ in range(1):
        train_func()

    model.model_save('')
    model.close()
