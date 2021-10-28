def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))
    
def rgb_to_hex(r, g, b):
    # rgb = hex(((r&0x0ff)<<16)|((g&0x0ff)<<8)|(b&0x0ff))

    rgb = (r<<16) + (g<<8) + b
    return rgb
    # return '%02x%02x%02x' % rgb

if __name__ == '__main__':
    print(rgb_to_hex(101, 91, 76))
    # colorRGB = int(rgb_to_hex((115, 200, 50)), base=16)
    
    # r = (colorRGB >> 16) & 0x0000ff
    # g = (colorRGB >> 8) & 0x0000ff
    # b = (colorRGB ) & 0x0000ff

    # print(r)
    # print(g)
    # print(b)
