import json
from six import iteritems
import tqdm


def _save_to_text(data, save_path):
    with open(save_path, 'a') as f:
        for key, datum in iteritems(data):
            save_format = {
                key: datum
            }
            json_save_format = json.dumps(save_format)
            f.write(json_save_format + '\n')


if __name__ == "__main__":
    READ_PATH = r'../res/metadata.json'
    SAVE_PATH = r'../tmp/title_metadata'

    title_meta_data = dict()
    for line in tqdm.tqdm(open(READ_PATH, 'r')):
        json_line = json.loads(line)
        title_meta_value_format = {
            'title': json_line.get('title'),
            'sub_title': json_line.get('sub_title'),
            'keyword_list': json_line.get('keyword_list', [])
        }

        title_meta_data[json_line['id']] = title_meta_value_format

    _save_to_text(title_meta_data, SAVE_PATH)
