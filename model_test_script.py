from tqdm import tqdm
from popular_model.loader import get_metadata, get_data, get_follow
from popular_model.recommender_model import PopularModel
import numpy as np


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')

    train_data, test_data, all_data = get_data()
    meta_data, read_data, meta_read_data = get_metadata()
    follow_data, follow_len = get_follow()

    meta_data_len = len(meta_data)
    read_data_len = len(read_data)

    model = PopularModel(1, meta_data_len, read_data_len, follow_len)
    model.model_load('')
    meta_key = np.array(list(meta_data.keys()))

    with open(r'./most_list.txt') as f:
        most_list = f.read()

    with open(r'./tmp/dev.users') as f:
        recom_dev_data = f.read()
        recom_dev_data = recom_dev_data.split('\n')
        recom_dev_data = recom_dev_data[:3000]

    with open(r'./tmp/nf.recommender.txt', 'w') as f:
        for user in tqdm(recom_dev_data):
            user = user.replace('\n', '')
            try:
                k = user
                if k not in all_data:
                    raise ImportError
                v = all_data[user]

                test_value_list = []
                test_keyword_list = []
                for test_value in v:
                    if test_value in meta_read_data:
                        test_value_idx = meta_read_data[test_value].get('key')
                        test_value_list.append(test_value_idx)
                        test_keyword_list += meta_read_data[test_value].get('read')

                follow_value_list = []
                if k in follow_data:
                    follow_value_list += follow_data[k]
                else:
                    follow_value_list.append(follow_len)

                predict_result = model.predict([test_value_list], [test_keyword_list], [follow_value_list])
                sort_idx = np.argsort(predict_result)[::-1][:400]
                score_result = meta_key[sort_idx]

                result = [str(x) for x in score_result if str(x) not in ' '.join(v)][:100]
                f.write('%s %s\n' % (k, ' '.join(result)))
            except Exception as e:
                f.write('%s %s\n' % (k, most_list))