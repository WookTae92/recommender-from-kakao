import json

from typing import List, Dict
from konlpy.tag import Mecab


class EmbeddingVocabulary(object):
    def __init__(self, load_path=None):
        self._load_embedding_path = load_path
        self._embedding_voca = None
        self._voca_size = None
        self._mecab = Mecab()

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
        self._embedding_voca_size = len(self._embedding_voca)