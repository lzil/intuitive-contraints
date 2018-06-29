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
    noise = args['noise']
    file = args['file']
    file2 = args['file_secondary']
    no_lines = args['no_lines']

    # get file id which is determined by type of constraint and timestamp
    file_id = file.split('/')[-1].split('.')[0]

    # actually load the file
    with open(file, 'rb') as f:
        data = pickle.load(f)

    space_data = data['space']
    locs_data = data['locs']

    num_obj = len(locs_data[0])

    
    if file2:
        with open(file2, 'rb') as f:
            data2 = pickle.load(f)
        con_data2 = data2['constraints'][-1]

        ch = scene.space.add_collision_handler(2, 3)
        def begin(arbiter, space, data):
            return False
        ch.begin = begin

        pygame.init()
        screen = pygame.display.set_mode(scene.size)
        pygame.display.set_caption('comparison with snapback')
        clock = pygame.time.Clock()
        draw_options = pymunk.pygame_util.DrawOptions(screen)

        skip = 20
        sim_length = 15
        for j in range(len(locs_data) // skip - 1):
            scene.reset_space()
            scene.add_bodies_from_rep(locs_data[j * skip])
            scene.add_constraints_from_rep(con_data)
            scene.add_bodies_from_rep(locs_data[j * skip], col_type='ball_red')
            scene.add_constraints_from_rep(con_data2, num_obj)

            for i in range(sim_length):
                screen.fill((255,255,255))
                for event in pygame.event.get():
                    if event.type in [QUIT, K_ESCAPE]:
                        sys.exit(0)
                scene.space.debug_draw(draw_options)
                pygame.display.flip()
                clock.tick(TICK / 2)
                scene.space.step(1/50.0)


    # draw data happening without any constraints shown
    elif no_lines:
        scene = SimulationScene(space=space_data, noise=noise)
        pygame.init()
        screen = pygame.display.set_mode(scene.size)
        pygame.display.set_caption('visualization (no constraints shown)')
        clock = pygame.time.Clock()
        draw_options = pymunk.pygame_util.DrawOptions(screen)

        for con in scene.space.constraints:
            scene.space.remove(con)

        for t in range(len(locs_data)):
            screen.fill((255,255,255))
            for event in pygame.event.get():
                if event.type in [QUIT, K_ESCAPE]:
                    sys.exit(0)
            
            clock.tick(TICK)
            scene.space.debug_draw(draw_options)
            pygame.display.flip()
            scene.update_body_locations(locs_data[t])

        
    else:
        scene = SimulationScene(space=space_data, noise=noise)
        scene.run_and_visualize(min(len(locs_data), TIME_LIMIT), label='visualization (with constraints)')


def parse_args():
    parser = argparse.ArgumentParser(description='Load scene.')

    parser.add_argument('file')
    parser.add_argument('-f', '--file_secondary', help='Second file')
    parser.add_argument('-x', '--no_lines', help='No constraint connections shown', action='store_true')
    parser.add_argument('-n', '--noise', help='amount of noise, in [collision, dynamic] form', nargs=2, type=float,default=[0,0])
    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

   