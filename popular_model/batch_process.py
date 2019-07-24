"""
실패 실패 ㅠㅠ
"""

import json
import numpy as np

from typing import Tuple, List, Dict
from konlpy.tag import Mecab
from six import iteritems

mecab = Mecab()


class EmbeddingVocabulary(object):
    def __init__(self, load_path=None):
        self._load_embedding_path = load_path
        self._mecab = Mecab()

        # load된 데이터
        self._embedding_voca = None
        self._voca_size = None

        # load 작업
        self._load_embedding()

    @property
    def embedding_voca(self) -> Dict:
        return self._embedding_voca

    @property
    def embedding_voca_size(self) -> int:
        return self._voca_size

    def embedding_tokenizer(self, sentence: str) -> List:
        tokens = self._mecab.morphs(sentence)

        embedding_token_result = []
        for token in tokens:
            embedding_token = self._embedding_voca.get(token, self.embedding_voca_size)
            embedding_token_result.append(embedding_token)

        return embedding_token_result

    def _load_embedding(self):
        with open(self._load_embedding_path) as f:
            data = f.read()
            self._embedding_voca = json.loads(data)
        self._voca_size = len(self._embedding_voca)


class MetaDataBatch(object):
    def __init__(self, load_path=None, embedding_model=None):
        self._load_meta_path = load_path
        self._ev: EmbeddingVocabulary = embedding_model

        # load된 데이터
        self._max_meta_data_len = 0
        self._max_title_len = 0
        self._meta_title_ids = []
        self._meta_title_embedding = []
        self._meta_data = {}

        # load 작업
        self._load_meta_labels()

    @property
    def max_title_length(self):
        return self._max_title_len

    @property
    def max_meta_data_length(self):
        return self._max_meta_data_len

    @property
    def title_ids(self) -> List:
        return self._meta_title_ids

    @property
    def title_embedding(self) -> List:
        return self._meta_title_embedding

    @property
    def meta_data(self):
        return self._meta_data

    def _load_meta_labels(self):
        meta_data_dict = dict()
        max_length = 0
        for lines in open(self._load_meta_path):
            meta_data = json.loads(lines)
            for k, v in meta_data.items():
                token_numbers = self._ev.embedding_tokenizer(v.get('title', ''))

                if max_length < len(token_numbers):
                    max_length = len(token_numbers)

                meta_data_dict[k] = {'title': token_numbers}

        title_ids = []
        title_embedding = []
        embedding_meta_data_dict = {}

        for k, v in iteritems(meta_data_dict):
            title_ids.append(k)

            value = np.pad(
                np.array(v.get('title')),
                [0, max_length - len(v.get('title'))],
                mode='constant'
            )
            title_embedding.append(value)
            embedding_meta_data_dict[k] = value

        self._max_title_len = max_length
        self._max_meta_data_len = len(meta_data_dict)
        self._meta_title_ids = title_ids
        self._meta_title_embedding = title_embedding
        self._meta_data = embedding_meta_data_dict


class TrainBatch(object):
    def __init__(self, load_path=None, embedding_model=None):
        self._load_path = load_path
        self._ev: EmbeddingVocabulary = embedding_model

        # load된 데이터
        self._train_data = None

        # load 작업
        self._load_data()

    def unpack_batch(self, batch_size: int):
        scope = 'title'
        result = self._title_batch(batch_size, scope)
        return self._zero_padding(zip(*result))

    def _load_data(self):
        train_data_dict = dict()
        positive_title = []

        print('load start')
        for idx, sentence in enumerate(open(self._load_path)):
            try:
                sentence = json.loads(sentence)
            except json.decoder.JSONDecodeError:
                continue

            positive_title_tmp = []
            for title in sentence['explanation']:
                if title.get('title'):
                    positive_title_tmp.append(title.get('title'))

            if positive_title_tmp:
                positive_title.append(positive_title_tmp)
        print('load complete')

        train_data_dict['title'] = positive_title
        self._train_data = train_data_dict

    def _zero_padding(self, batch_data: zip):
        titles, sub_titles, positive = batch_data

        new_titles = self._title_zero_padding(titles)
        new_sub_titles = self._sub_title_zero_padding(sub_titles)

        return new_titles, new_sub_titles, positive

    @staticmethod
    def _title_zero_padding(titles: List[List]):
        max_title_len = 0
        for title_list in titles:
            for title in title_list:
                if len(title) > max_title_len:
                    max_title_len = len(title)

        new_titles = []
        for title_list in titles:
            new_tmp_titles = []
            for title in title_list:
                zero_padding = np.pad(np.array(title), [0, max_title_len - len(title)], mode='constant')
                new_tmp_titles.append(zero_padding)

            new_titles.append(new_tmp_titles)

        return new_titles

    @staticmethod
    def _sub_title_zero_padding(sub_titles: List):
        max_sub_title_len = 0
        for sub_title in sub_titles:
            if len(sub_title) > max_sub_title_len:
                max_sub_title_len = len(sub_title)

        new_sub_titles = []
        for sub_title in sub_titles:
            zero_padding = np.pad(np.array(sub_title), [0, max_sub_title_len - len(sub_title)], mode='constant')
            new_sub_titles.append(np.array(zero_padding))

        return new_sub_titles

    def _title_batch(self, batch_size: int, scope='title') -> Tuple:
        cnt = 0
        if scope not in ['title', 'sub_title']:
            raise NotImplemented

        while True:
            cnt += 1
            main_p_title, sub_p_title = self._title_choice(True, scope)
            yield main_p_title, sub_p_title, 1
            if batch_size == cnt:
                break

            cnt += 1
            main_n_title, sub_n_title = self._title_choice(False, scope)
            yield main_n_title, sub_n_title, 0
            if batch_size == cnt:
                break

    def _title_choice(self, pn=True, scope='title'):
        match_idx = 0
        if not pn:
            match_idx = 1

        title_choices_index = np.random.randint(0, len(self._train_data[scope]) - 1, 2)

        main_title = np.random.choice(
            self._train_data[scope][title_choices_index[0]], 3
        )
        sub_title = np.random.choice(
            self._train_data[scope][title_choices_index[match_idx]], 1
        )
        sub_title = sub_title[0]

        return self._match_list_embedding(main_title), self._match_embedding(sub_title)

    def _match_list_embedding(self, titles: list):
        embedding_number = []
        for title in titles:
            embedding_number_tmp = self._ev.embedding_tokenizer(str(title))
            embedding_number.append(embedding_number_tmp)

        return embedding_number

    def _match_embedding(self, title: str):
        return self._ev.embedding_tokenizer(str(title))


class DevBatch(object):
    def __init__(self, load_path=None, meta_data=None):
        self._load_dev_path = load_path
        self._mb: MetaDataBatch = meta_data

        # load된 데이터
        self._dev_data = {}

        # load 작업
        self._load_dev_data()

    @property
    def dev_data(self):
        return self._dev_data

    def _load_dev_data(self):
        cnt = 0
        for line in open(self._load_dev_path):
            line_split = line.split(' ')
            user = line_split[0]
            items = line_split[1:]

            item_embedding = []
            for item in items:
                item_tmp_emb = self._mb.meta_data.get(item, None)
                if item_tmp_emb is not None:
                    item_embedding.append(item_tmp_emb)

            self._dev_data[user] = [item_embedding]
            cnt += 1
            if cnt == 1000:
                break


if __name__ == "__main__":
    pass
