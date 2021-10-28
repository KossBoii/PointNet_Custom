import numpy as np
import h5py
import os
import sys

def rgbint_to_rgb(rgb_int):
  blue =  rgb_int & 255
  green = (rgb_int >> 8) & 255
  red =   (rgb_int >> 16) & 255
  return (red, green, blue)

def convert_pcd(ascii_pcd):         # in form of '_____.pcd'
    with open(ascii_pcd) as f:
        lines = f.readlines()
    
    with open('annotations.pcd', 'w') as out:
        for i in range(len(lines)):
            if lines[i][0].isdigit():
                temp = lines[i].split(' ')
                intensity = temp[0]
                rgb = temp[1]
                x = temp[2]
                y = temp[3]
                z = temp[4][:-1]

                # convert RGB Int to separate RGB values
                red, green, blue = rgbint_to_rgb(int(rgb))
                # '%f %f %'
                # output = '' + str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(red) + ' ' + str(green) + ' ' + str(blue) + ' ' + str(intensity) + '\n'
                out.write('{} {} {} {} {} {} {}\n'.format(x, y, z, red, green, blue, intensity))
            else:
            if lines[i][:6] == 'FIELDS':
                out.write('FIELDS x y z r g b intensity\n')
            elif lines[i][:4] == 'SIZE':
                out.write('SIZE 4 4 4 4 4 4 4\n')
            elif lines[i][:4] == 'TYPE':
                out.write('TYPE F F F U U U F\n')
            elif lines[i][:5] == 'COUNT':
                out.write('COUNT 1 1 1 1 1 1 1\n')
            else:
                out.write(lines[i])

    out.close()

def add_color_annotations(src_pcd, annos_pcd):
    count = 0
    class_map = {}

    with open(src_pcd) as f1:       // contains information regarding color
    lines = f1.readlines()

    with open(annos_pcd) as f2:         // contains the labels information
        lines1 = f2.readlines()

    with open('annos_rgb.pcd', 'w') as out:
        for i in range(len(lines)):
            if lines[i][0].isdigit():
                temp = lines[i].split(' ')
                temp1 = lines1[i].split(' ')
                
                x = temp[0]
                y = temp[1]
                z = temp[2]
                r = temp[3]
                g = temp[4]
                b = temp[5]
                intensity = temp[6][:-1]

                x1 = temp1[0]
                y1 = temp1[1]
                z1 = temp1[2]
                label = temp1[3]
                object_id = temp1[4][:-1]

            if label not in class_map:
                class_map[label] = count
                count = count + 1
            else:
                label = class_map[temp1[3]]

            if x == x1 and y == y1 and z == z1:
                out.write('{} {} {} {} {} {} {} {} {}\n'.format(x, y, z, red, green, blue, intensity, label, object_id))
            else:
            if lines[i][:6] == 'FIELDS':
                out.write('FIELDS x y z r g b intensity label object\n')
            elif lines[i][:4] == 'SIZE':
                out.write('SIZE 4 4 4 4 4 4 4 4 4\n')
            elif lines[i][:4] == 'TYPE':
                out.write('TYPE F F F U U U F I I\n')
            elif lines[i][:5] == 'COUNT':
                out.write('COUNT 1 1 1 1 1 1 1 1 1\n')
            else:
                out.write(lines[i])

    out.close()   
    return class_map

def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def normalize_data():
    raise NotImplementedError