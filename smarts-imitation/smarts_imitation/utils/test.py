import math
import numpy as np

def _legalize_angle(angle):
    return angle % (2 * math.pi)

def get_index(degree, n):
    radians = degree * 2 * math.pi / 360.0

    partition_size = math.pi * 2.0 / n

    new_angle = radians + partition_size / 2.0
    new_angle = _legalize_angle(new_angle)

    index = int(new_angle / partition_size)
    return index


for i in range(0, 360):
    print(i, get_index(i, 8))
