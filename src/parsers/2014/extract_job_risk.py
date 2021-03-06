import os
import sys

import utils

if __name__ == '__main__':
    print('main')
    if len(sys.argv) < 2:
        raise Exception('Enter the path of the directory')

    path = sys.argv[1]

    if not os.path.isdir(path):
        raise Exception('Invalid path : ', path)
    print('path', path)
    for idx, filename in enumerate(os.listdir(path)):
        # print(filename)
        tags, text = utils.extract_tags_risk(os.path.join(path, filename))

        words = utils.extract_words(text, tags, idx)

        df = utils.map_output(words)
        name = filename.split('.')[0]
        utils.save_to_csv('.', f'{name}.csv', df)