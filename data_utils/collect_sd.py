import os
import sys
from sd_utils import collect_point_label, BASE_DIR

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
anno_paths = [os.path.join('./data/sight_distance/', p) for p in anno_paths]

output_folder = './data/sight_distance'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for anno_path in anno_paths:
    print(anno_path)
    try:
        elements = anno_path.split('/')
        print(elements)
        out_filename = elements[-2]+'.npy' # Street1.npy
        collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
    except:
        print(anno_path, 'ERROR!!')