#!/usr/bin/env python3
"""Script to build a working set of images

TODO: define query bounding boxes and extract query images
"""
import argparse
import os
import sys
from PIL import Image

from notary_charters.annotations import write_labeled_annotations

parser = argparse.ArgumentParser(description='Build working set of images')
parser.add_argument('--data-dir', required=True, help='Data directory')
parser.add_argument('--out-dir', required=True, help='Output directory')

prefix = 'raw/notary_charters/notary_charters/'
WORKING_SET = [
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
    'HHSTA_Salzburg_1376_12_01.jpg',
    'HHSTA_Salzburg_1381_10_18.jpg',
    'HHSTA_Salzburg_1393_01_11-2.jpg',
    'HHSTA_Salzburg_1395_06_28-1.jpg',
    'HHSTA_Salzburg_1395_06_28-2.jpg',
    'HHSTA_Salzburg_1396-1403-15.jpg',
    'HHSTA_Salzburg_1397_07_09-2.jpg',
    'HHSTA_Salzburg_1397_08_16.jpg',
    'HHSTA_Salzburg_1399_06_17-1400_04_04.jpg',
    'HHSTA_Salzburg_1420_04_15.jpg',
    'HHSTA_Salzburg_1420_11_15.jpg',
    'HHSTA_Salzburg_1428_01_25.jpg',
    'HHSTA_Salzburg_1432_11_24-2.jpg',
    'HHSTA_Salzburg_1433_11_02-2.jpg',
    'HHSTA_Salzburg_1433_11_02-3.jpg',
    'SLA-OU_14520816_r.jpg',
    'StadtAWo_Abt1AI_0348_14080829_r.JPG',
]

QUERIES = {
    'BayHStA-KUPassauStNikola_14220924_r.jpg': [
        ('vase', (116, 577, 116+98, 577+121))
    ],
    'BayHStA-KUPassauStNikola_14230223_r.jpg': [
        ('vase', (123, 411, 125+123, 411+158))
    ],
    'BayHStA-KUPassauStNikola_14550521_A_r.jpg': [
        ('cross', (143, 722, 143+156, 722+163))
    ],
    'BayHStA-KUPassauStNikola_14550521_B_r.jpg': [
        ('cross', (108, 816, 108+193, 816+163))
    ],
    'BayHStA-KUPassauStNikola_14560412_r.jpg': [
        ('cross', (98, 637, 98+192, 637+189))
    ]
}

MAX_SIZE = 1000  # Maximum height or width of the images (resizes if necessary)

def main(args):
    args = parser.parse_args(args)

    queries_per_label = {}
    query_dir = os.path.join(args.out_dir, 'query')
    if not os.path.exists(query_dir):
        os.mkdir(query_dir)

    for idx, file in enumerate(WORKING_SET):
        path = os.path.join(args.data_dir, prefix, file)
        out_name = '{:02d}.jpg'.format(idx+1)
        out_path = os.path.join(args.out_dir, out_name)

        img = Image.open(path)
        img.thumbnail((MAX_SIZE, MAX_SIZE))
        img.save(out_path)

        if file in QUERIES:
            for label, bbox in QUERIES[file]:
                crop = img.crop(bbox)

                prev_queries = queries_per_label.get(label, [])
                query_name = '{}_{}'.format(len(prev_queries), out_name)

                label_dir = os.path.join(query_dir, label)
                if not os.path.exists(label_dir):
                    os.mkdir(label_dir)

                crop.save(os.path.join(label_dir, query_name))

                prev_queries.append((query_name, bbox))
                queries_per_label[label] = prev_queries

    labeled_annotations = []
    for label, queries in sorted(queries_per_label.items()):
        for query in queries:
            query_name, bbox = query
            labeled_annotations.append((query_name, bbox, label))

    query_file = os.path.join(query_dir, 'labeled_queries.csv')
    write_labeled_annotations(query_file, labeled_annotations)

if __name__ == '__main__':
    main(sys.argv[1:])
