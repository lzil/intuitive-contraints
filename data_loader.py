import sys, random
import os
import time
import math

import pymunk
import pymunk.pygame_util

import pickle

import pygame
from pygame.locals import *
from pygame.color import *

import numpy as np
import matplotlib.pyplot as plt

from scene_tools import *

import argparse

def main(args):
    viz = args['visualize']
    noise = args['noise']
    file = args['file']
    no_lines = args['no_lines']

    # get file id which is determined by type of constraint and timestamp
    file_id = file.split('/')[-1].split('.')[0]
    simtype = file_id.split('_')[0]

    # actually load the file
    with open(file, 'rb') as f:
        data = pickle.load(f)

    obj_data = data['obj']
    con_data = data['con']
    # con_data[0][3] = (150,101,0.083)

    num_obj = len(obj_data[0])

    scene = Scene(noise)
    scene.add_bodies_from_rep(obj_data[0])
    scene.add_constraints_from_rep(con_data)

    # draw data happening without any constraints
    if no_lines:
        scene.visualize_obj_data(obj_data)
    elif viz:
        locations = scene.run_and_visualize(min(len(obj_data), TIME_LIMIT), label='data_loader')
    else:
        locations = scene.run_and_record(min(len(obj_data), TIME_LIMIT))

    # assert len(locations) == len(obj_data)

def parse_args():
    parser = argparse.ArgumentParser(description='Load scene.')

    parser.add_argument('file')
    parser.add_argument('-v', '--visualize', help='Visualize system.', action='store_true')
    parser.add_argument('-x', '--no_lines', help='No constraint connections shown', action='store_true')
    parser.add_argument('-n', '--noise', help='amount of noise, in [collision, dynamic] form', nargs=2, type=float,default=[0,0])
    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

   