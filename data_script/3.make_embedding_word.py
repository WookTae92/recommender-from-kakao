import json
from konlpy.tag import Mecab
from six import itervalues
from collections import defaultdict
from typing import List, Dict, AnyStr


def _mecab_word_template(embed_key: Dict, title: AnyStr):
    mecab_result = mecab.morphs(str(title))
    for word in mecab_result:
        if embed_key[word]:
            pass


def _update_keyword_list(embed_key: Dict, keyword_list: List):
    for word in keyword_list:
        if embed_key[word]:
            pass


if __name__ == "__main__":
    READ_PATH = r"../tmp/title_metadata"
    SAVE_PATH = r"../tmp/embedding_word"

    mecab = Mecab()
    embedding_keys = defaultdict(int)
    embedding_keys.default_factory = embedding_keys.__len__

    for idx, line in enumerate(open(READ_PATH, 'r')):
        json_line = json.loads(line)

        for v in itervalues(json_line):
            _mecab_word_template(embedding_keys, v['title'])
            _mecab_word_template(embedding_keys, v['sub_title'])
            _update_keyword_list(embedding_keys, v['keyword_list'])

    with open(SAVE_PATH, 'w') as f:
        f.write(json.dumps(embedding_keys))
