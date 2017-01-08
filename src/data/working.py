#!/usr/bin/env python3
"""Script to build a working set of images

TODO: define query bounding boxes and extract query images
"""
import argparse
import os
import sys
from PIL import Image

parser = argparse.ArgumentParser(description=
                                 'Extract feature representations')
parser.add_argument('--data-dir',
                    help='Data directory')
parser.add_argument('--out-dir',
                    help='Output directory')

prefix = 'raw/notary_charters/notary_charters/'
working_set = [
    'BayHStA-KUPassauStNikola_14220924_r.jpg',
    'BayHStA-KUPassauStNikola_14230223_r.jpg',
    'BayHStA-KUPassauStNikola_14550521_A_r.jpg',
    'BayHStA-KUPassauStNikola_14550521_B_r.jpg',
    'BayHStA-KUPassauStNikola_14560412_r.jpg',
    'BayHStA-Passau-Domkapitel_13180113_00166_r.jpg',
    'BayHStA-Passau-Domkapitel_13430505_00352_r.jpg',
    'BayHStA-Passau-Domkapitel_13531211_00444_r.jpg',
    'BayHStA-Passau-Domkapitel_13791013_00743_r.jpg',
    'de_bayhsta_ku-rohr_0388_14860716_e8aaabef-e110-4576-ae88-4421473ce600_r.jpg',
    'HGW-Urkunden_14400630_00144_r.jpg',
    'HHSTA_Salzburg_1283_03_26.jpg',
    'HHSTA_Salzburg_1318_12_05-3-5.jpg',
    'HHSTA_Salzburg_1338_11_20.jpg',
    'HHSTA_Salzburg_1376_07_15.jpg',
    'HHSTA_Salzburg_1376_07_15.jpg',
    'HHSTA_Salzburg_1376_12_01.jpg',
    'HHSTA_Salzburg_1376_12_01.jpg',
    'HHSTA_Salzburg_1381_10_18.jpg',
    'HHSTA_Salzburg_1381_10_18.jpg',
    'HHSTA_Salzburg_1395_06_28-1.jpg',
    'HHSTA_Salzburg_1395_06_28-2.jpg',
    'HHSTA_Salzburg_1395_06_28-2.jpg',
    'HHSTA_Salzburg_1396-1403-15.jpg',
    'HHSTA_Salzburg_1397_07_09-2.jpg',
    'HHSTA_Salzburg_1399_06_17-1400_04_04.jpg',
    'HHSTA_Salzburg_1420_04_15.jpg',
    'HHSTA_Salzburg_1420_11_15.jpg',
    'HHSTA_Salzburg_1428_01_25.jpg',
    'HHSTA_Salzburg_1432_11_24-2.jpg',
    'HHSTA_Salzburg_1433_11_02-2.jpg',
    'HHSTA_Salzburg_1433_11_02-3.jpg',
]

MAX_SIZE = 1000  # Maximum height or width of the images (resizes if necessary)

def main(args):
    args = parser.parse_args(args)

    for idx, path in enumerate(working_set):
        path = os.path.join(args.data_dir, prefix, path)
        out_path = os.path.join(args.out_dir, 
                                '{:02d}.jpg'.format(idx+1))

        img = Image.open(path)
        img.thumbnail((MAX_SIZE, MAX_SIZE))
        img.save(out_path)

if __name__ == '__main__':
    main(sys.argv[1:])
