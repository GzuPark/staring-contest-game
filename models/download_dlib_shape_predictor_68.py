from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bz2
import os
from urllib import request


def download(filename, filepath):
    url = 'https://raw.githubusercontent.com/davisking/dlib-models/master/{}.bz2'.format(filename)
    print("[INFO] Start to download")

    request.urlretrieve(url, '{}.bz2'.format(filepath))

    print("[INFO] Downloaded: {}.bz2".format(filename))


def decompress(filename, filepath):
    print("[INFO] Start to decompress")

    zipfile = bz2.BZ2File('{}.bz2'.format(filepath))
    data = zipfile.read()
    open(filepath, 'wb').write(data)

    print("[INFO] Decompressed: {}".format(filename))


def clear(filepath):
    os.remove('{}.bz2'.format(filepath))


def main():
    realpath = os.path.dirname(os.path.realpath(__file__))
    filename = 'shape_predictor_68_face_landmarks.dat'
    filepath = os.path.join(realpath, filename)
    params = {
        'filename': filename,
        'filepath': filepath,
    }

    download(**params)
    decompress(**params)
    clear(filepath)


if __name__ == '__main__':
    main()
