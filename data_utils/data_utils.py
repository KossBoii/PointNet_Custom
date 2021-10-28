import os
import numpy as np

x_shift = 0
y_shift = 0
z_shift = 0
z_max = 106.68

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))

def rgb_to_int(r, g, b):
    return (r<<16) + (g<<8) + b

def int_to_rgb(rgb_int):
    blue =  rgb_int & 255
    green = (rgb_int >> 8) & 255
    red =   (rgb_int >> 16) & 255
    return red, green, blue

def rgb_to_hex(rgb):
    # rgb = hex(((r&0x0ff)<<16)|((g&0x0ff)<<8)|(b&0x0ff))
    return '%02x%02x%02x' % rgb

def write_xyzrgb(save_dir, file_name, points, colors):
    points = np.asarray(points)
    colors = np.asarray(colors)
    assert(points.shape[0] == colors.shape[0])
    f = open(os.path.join(save_dir, file_name), 'w')

    for i in range(points.shape[0]):
        r = int(colors[i][0] * 255)
        g = int(colors[i][1] * 255)
        b = int(colors[i][2] * 255)
        f.write(f'{points[i][0]} {points[i][1]} {points[i][2]} {r} {g} {b}\n')
    f.close()

def write_pcd_xyzrgb(save_dir, file_name, points, colors):
    points = np.asarray(points)
    colors = np.asarray(colors)
    assert(points.shape[0] == colors.shape[0])
    f = open(os.path.join(save_dir, file_name), 'w')
    
    f.write('VERSION .7\n')
    f.write('FIELDS x y z rgb\n')
    f.write('SIZE 4 4 4 4\n')
    f.write('TYPE F F F U\n')
    f.write('COUNT 1 1 1 1\n')
    f.write('WIDTH ' + str(points.shape[0]) + '\n')
    f.write('HEIGHT 1\n')
    f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
    f.write('POINTS ' + str(points.shape[0]) + '\n')
    f.write('DATA ascii\n')

    for i in range(points.shape[0]):
        r = int(colors[i][0] * 255)
        g = int(colors[i][1] * 255)
        b = int(colors[i][2] * 255)        
        # rgb = rgb_to_hex((r, g, b))
        rgb = rgb_to_int(r, g, b)
        f.write(f'{points[i][0]} {points[i][1]} {points[i][2]} {rgb}\n')
    f.close()

def write_pcd(save_dir, file_name, data, type):
    assert(type == 'pts' or type == 'xyzrgbi')
    f = open(os.path.join(save_dir, file_name), 'w')

    if type == 'pts':
        f.write(str(len(data)) + '\n')
    elif type == 'xyzrgbi':
        f.write('VERSION .7\n')
        f.write('FIELDS x y z r g b i\n')
        f.write('SIZE 4 4 4 4 4 4 4\n')
        f.write('TYPE F F F U U U F\n')
        f.write('COUNT 1 1 1 1 1 1 1\n')
        f.write('WIDTH ' + str(len(data)) + '\n')
        f.write('HEIGHT 1\n')
        f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        f.write('POINTS ' + str(len(data)) + '\n')
        f.write('DATA ascii\n')

    for i in range(len(data)):
        f.write(data[i] + '\n')
    f.close()

def read_velodyne(input, datas, filter=False, type='pcd'):
    print(f'Reading {input}')
    assert(type == 'pts' or type == 'xyzrgb' or type == 'pcd')
    f = open(input, 'r')
    col_names = f.readline().split()
    num_pts = int(f.readline())
    count = 0

    while True:
        line = f.readline()
        if not line:    # if the line is empty
            break

        vals = line.split()
        assert(len(vals) == 12)
        x = float(vals[0]) - x_shift
        y = float(vals[1]) - y_shift
        z = float(vals[2]) - z_shift
        r = vals[3]
        g = vals[4]
        b = vals[5]
        gps_time = vals[9]
        intensity = int(float(vals[10]))

        count += 1
        if not filter:
            if z > z_max:
                pass
        
        pts_time = int(gps_time.split('.')[0])
        if pts_time not in datas.keys():
            datas[pts_time] = []
        
        if type == 'pts':
            datas[pts_time].append(f'{x} {y} {z} {intensity} {r} {g} {b}')
        elif type == 'xyzrgb':
            datas[pts_time].append(f'{x} {y} {z} {r} {g} {b}')
        elif type == 'pcd':
            datas[pts_time].append(f'{x} {y} {z} {r} {g} {b} {intensity}')
    
    assert(count == num_pts)
