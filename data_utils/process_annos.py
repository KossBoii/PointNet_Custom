import sys
import os
from data_utils import int_to_rgb

def correct_annos(annos_file, objnum_2_label_list, output_path):
    lines = None
    with open(annos_file, 'r') as f:
        lines = f.readlines()
        
        for id in range(len(lines)):
            if not lines[id][0].isnumeric():         # first character is not number
                continue
            else:                   # first character is number ==> point cloud data
                vals = lines[id].split()
                assert(len(vals) == 6)
                x = float(vals[0])
                y = float(vals[1])
                z = float(vals[2])
                rgb_int = int(vals[3])
                label = int(vals[4])
                obj_num = int(vals[5])

                if label != objnum_2_label_list[obj_num]:
                    lines[id] = f"{x} {y} {z} {rgb_int} {objnum_2_label_list[obj_num]} {obj_num}\n"

    with open(os.path.join(output_path, "corrected_annos.pcd"), 'w') as f1:
        f1.writelines("%s" % line for line in lines)

# def correct_annos(annos_file, class_list, output_path):
#     data = {}
#     id2class = {id: cls for id, cls in enumerate(class_list)}
#     count_obj_per_class = {cls: 0 for cls in class_list}
#     f = open(annos_file, 'r')

#     obj_label_list = {}
#     lines = f.readlines()
#     for line_num in range(len(lines)):
#         if not lines[line_num][0].isnumeric():         # first character is not number
#             continue
#         else:                   # first character is number ==> point cloud data
#             vals = lines[line_num].split()
#             assert(len(vals) == 6)
#             x = float(vals[0])
#             y = float(vals[1])
#             z = float(vals[2])
#             rgb_int = int(vals[3])
#             label = int(vals[4])
#             obj_num = int(vals[5])

#             if obj_num not in obj_label_list.keys():
#                 obj_label_list[obj_num] = []
            
#             if label not in obj_label_list[obj_num]:
#                 obj_label_list[obj_num].append(label)

#     print(obj_label_list)
#     # verify the correctness of label:
#     need_correct_list = []
#     for obj_id in obj_label_list.keys():
#         assert(len(obj_label_list[obj_id]) <= 2)
#         if len(obj_label_list[obj_id]) == 1:
#             # if obj_id == -1 and obj_label_list[obj_id][0] != 0:
#             #     need_correct_list.append(obj_id)
#             continue
#         else:               # number of labels == 2
#             need_correct_list.append(obj_id)
    
#     print(need_correct_list)
#     for label in need_correct_list:
#         obj_label_list[label].remove(0)
#         assert(len(obj_label_list[obj_id]) == 1 and obj_label_list[obj_id][0] != 0)
    
#     # correct the annotation
#     if len(need_correct_list) != 0:
#         for line_num in range(len(lines)):
#             if not lines[line_num][0].isnumeric():         # first character is not number
#                 continue
#             else:                   # first character is number ==> point cloud data
#                 vals = lines[line_num].split()
#                 assert(len(vals) == 6)
#                 x = float(vals[0])
#                 y = float(vals[1])
#                 z = float(vals[2])
#                 rgb_int = int(vals[3])
#                 label = int(vals[4])
#                 obj_num = int(vals[5])

#                 if obj_num != 0 and label == 0:
#                     lines[line_num] = f"{x} {y} {z} {rgb_int} {obj_label_list[obj_num][0]} {obj_num}\n"
#                 # if obj_num in need_correct_list and label != obj_label_list[obj_num][0]:
#                 #     lines[line_num] = f"{x} {y} {z} {rgb_int} {obj_label_list[obj_num][0]} {obj_num}"

#     with open(os.path.join(output_path, "corrected_annos.pcd"), 'w') as f1:
#         f1.writelines("%s" % line for line in lines)

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

        print(f"obj_id: {obj_id}, {len(data[obj_id])} points")

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

    # objnum_2_label_list = {
    #     -1: 0,
    #     0: 4, 1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4,
    #     11: 4, 12: 4, 13: 4, 14: 4, 15: 4, 16: 4, 17: 4, 18: 4, 19: 4, 20: 4, 
    #     21: 1, 22: 2, 23: 1, 24: 3, 25: 3
    # }
    # correct_annos(annos_path, objnum_2_label_list, output_path)
