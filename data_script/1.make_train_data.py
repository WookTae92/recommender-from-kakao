from data_script.util import iterate_data_files
import tqdm
import json
from concurrent.futures import ProcessPoolExecutor


def _metadata_maker(path):
    _metadata_restore = dict()
    for line in tqdm.tqdm(open(path, 'r')):
        line_to_json = json.loads(line)
        _metadata_restore[line_to_json['id']] = {
            'title': line_to_json.get('title', ''),
            'sub_title': line_to_json.get('sub_title', ''),
            'keyword_list': line_to_json.get('keyword_list', ''),
            'user_id': line_to_json.get('user_id', '')

        }
    return _metadata_restore


def _save_to_text(data, save_path):
    with open(save_path, 'a') as f:
        for datum in data:
            for line in datum:
                dump_line = json.dumps(line)
                f.write(dump_line + '\n')


if __name__ == "__main__":
    READ_PATH = r'../res/metadata.json'
    TODATE = 2018100100
    FROMDATE = 2019022200
    SAVE_PATH = r'../tmp/new_train'

    files = sorted([path for path, _ in iterate_data_files(TODATE, FROMDATE)])
    metadata_restore = _metadata_maker(READ_PATH)


    def find_by_user_id_meta(path):
        user_id_by_meta = []
        for line in open(path, 'r'):
            lines = line.split(' ')

            if len(lines) <= 2:
                user_id_by_meta.append(
                    {
                        'user_id': lines[0],
                        'explanation': [{
                            'title': '',
                            'sub_title': '',
                            'keyword_list': '',
                            'user_id': '',
                        }]
                    }
                )

            explanation = []
            for line_id in lines[1:-1]:
                if line_id in metadata_restore:
                    line_id_meta = metadata_restore[line_id]
                    explanation.append(line_id_meta)
                else:
                    explanation.append(
                        {
                            'title': '',
                            'sub_title': '',
                            'keyword_list': '',
                            'user_id': '',
                        }
                    )
            user_id_dict = dict()
            user_id_dict['id'] = lines[0]
            user_id_dict['explanation'] = explanation
            user_id_by_meta.append(user_id_dict)

        return user_id_by_meta


    with ProcessPoolExecutor(max_workers=5) as executer:
        file_list = []

        for idx, file_path in enumerate(tqdm.tqdm(files)):
            file_list.append(file_path)
            if idx % 10 == 0:
                map_results = executer.map(find_by_user_id_meta, file_list)
                file_list = []
                _save_to_text(map_results, SAVE_PATH)

        map_results = executer.map(find_by_user_id_meta, file_list)
        _save_to_text(map_results, SAVE_PATH)
