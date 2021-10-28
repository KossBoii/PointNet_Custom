import sys
import os
from data_utils import int_to_rgb

def process_annos(anno_file, class_list, output_path):
    data = {}
    id2class = {id: cls for id, cls in enumerate(class_list)}
    count_obj_per_class = {cls: 0 for cls in class_list}
    f = open(anno_file, 'r')

    while True:
        line = f.readline()
        if line.startswith('DATA'):
            break
    
    while True:
        line = f.readline()
        if not line:    # if the line is empty
            break

        # parsing the annotations
        vals = line.split()
        assert(len(vals) == 6)
        x = float(vals[0])
        y = float(vals[1])
        z = float(vals[2])
        rgb_int = int(vals[3])
        label = int(vals[4])
        obj_num = int(vals[5])

        r,g,b = int_to_rgb(rgb_int)

        if obj_num not in data.keys():
            data[obj_num] = []
        
        data[obj_num].append(
            (x, y, z, r, g, b, label)
        )
    
    print(sorted(data.keys()))
    print(f'There are total of {len(data.keys())} objects')
    for obj_id in sorted(data.keys()):
        if obj_id == -1:
            print(f'{obj_id} is passed')
            continue
        temp_data = data[obj_id]
        temp_id = 0
        label_id = None
        while True:
            if temp_data[temp_id][-1] != 0:     # if the label_id is not 'void'
                label_id = temp_data[temp_id][-1]
                break
            else:                               # if the label_id keep being 'void'
                temp_id = temp_id + 1
        label_in_str = id2class[label_id]

        file_name = label_in_str + '_' + str(count_obj_per_class[label_in_str] + 1) + '.txt'
        f1 = open(os.path.join(output_path, file_name), 'w')
        for i in range(len(temp_data)):
            x = temp_data[i][0]
            y = temp_data[i][1]
            z = temp_data[i][2]
            r = temp_data[i][3]
            g = temp_data[i][4]
            b = temp_data[i][5]
            f1.write(f'{x} {y} {z} {r} {g} {b}\n')
        count_obj_per_class[label_in_str] = count_obj_per_class[label_in_str] + 1
        f1.close()
    
    f.close()

if __name__ == '__main__':
    class_list = ['void', 'road', 'car', 'building', 'tree']
    BASE_PATH = './data/sight_distance/Street2/'
    annos_path = os.path.join(BASE_PATH, 'Annotations/pcd_2_annos.pcd')
    output_path = os.path.join(BASE_PATH, 'Annotations/')
    process_annos(annos_path, class_list, output_path)
