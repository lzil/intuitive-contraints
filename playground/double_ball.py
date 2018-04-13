
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

import argparse

parser = argparse.ArgumentParser(description='Constraint system arguments.')
# types: inertial, circular, harmonic
parser.add_argument('type', help='Type of constraint system.')
parser.add_argument('-v', '--visualize', help='Visualize system.', action='store_true')
args = vars(parser.parse_args())

simtype = args['type']
viz = args['visualize']
# simtype = 'inertial'

now = int(time.time())
data_folder = os.path.join('scenes', simtype) 
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

def create_ball(space, mass, radius, position, velocity=(0,0), angular_velocity=0):
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = position
    body.velocity = velocity
    body.angular_velocity = angular_velocity
    shape = pymunk.Circle(body, radius)
    space.add(body, shape)
    return shape

def main():
    if viz:
        pygame.init()
        screen = pygame.display.set_mode((1000, 1000))
        pygame.display.set_caption("double_ball")
        clock = pygame.time.Clock()
        print(clock)
        draw_options = pymunk.pygame_util.DrawOptions(screen)

    space = pymunk.Space()
    space.gravity = (0.0, 0.0)

    item_details = []
    items = []

    def add_ball(pos, vel):
        shape = create_ball(space, 1, 10, pos, vel, 0)
        item_details.append(['obj', len(items), None])
        items.append(shape)
        return shape

    # ticks_to_next_ball = 10

    if simtype == 'none':
        pos = [(random.randint(300,700), random.randint(300,700)) for i in range(2)]
        vel = [(random.randint(-200,200), random.randint(-200,200)) for i in range(2)]
        shapes = [add_ball(pos[i], vel[i]) for i in range(2)]

    elif simtype == 'pin':
        pos = [(random.randint(300,700), random.randint(300,700)) for i in range(2)]
        vel = [(random.randint(-400,400), random.randint(-400,400)) for i in range(2)]
        shapes = [add_ball(pos[i], vel[i]) for i in range(2)]

        pin_joint = pymunk.PinJoint(shapes[0].body, shapes[1].body, (0,0), (0,0))
        item_details.append(['con', len(items), ['pin', items.index(shapes[0]), items.index(shapes[1])]])
        items.append(pin_joint)
        space.add(pin_joint)

    elif simtype == 'slide':
        pos = [(random.randint(300,700), random.randint(300,700)) for i in range(2)]
        vel = [(random.randint(-400,400), random.randint(-400,400)) for i in range(2)]
        shapes = [add_ball(pos[i], vel[i]) for i in range(2)]

        cur_dist = ((pos[1][0] - pos[0][0]) ** 2 + (pos[1][1] - pos[0][1]) ** 2) ** (1/2)
        mn = random.random() * cur_dist
        mx = (1 + random.random()) * cur_dist

        slide_joint = pymunk.SlideJoint(shapes[0].body, shapes[1].body, (0,0), (0,0), mn, mx)
        item_details.append(['con', len(items), ['slide', items.index(shapes[0]), items.index(shapes[1]), mn, mx]])
        items.append(slide_joint)
        space.add(slide_joint)

    elif simtype == 'spring-undamped':
        pos = [(random.randint(300,700), random.randint(300,700)) for i in range(2)]
        vel = [(random.randint(-200,200), random.randint(-200,200)) for i in range(2)]
        shapes = [add_ball(pos[i], vel[i]) for i in range(2)]

        rest_length = random.randint(0,200)
        stiffness = random.randint(4,80)
        spring_joint = pymunk.DampedSpring(shapes[0].body, shapes[1].body, (0,0), (0,0), rest_length, stiffness, 0)
        item_details.append(['con', len(items), ['spring', items.index(shapes[0]), items.index(shapes[1]), rest_length, stiffness, 0]])
        items.append(spring_joint)
        space.add(spring_joint)

    elif simtype == 'spring-damped':
        pos = [(random.randint(300,700), random.randint(300,700)) for i in range(2)]
        vel = [(random.randint(-200,200), random.randint(-200,200)) for i in range(2)]
        shapes = [add_ball(pos[i], vel[i]) for i in range(2)]

        rest_length = random.randint(0,200)
        stiffness = random.randint(10,120)
        damping = 0.1 / random.random()
        spring_joint = pymunk.DampedSpring(shapes[0].body, shapes[1].body, (0,0), (0,0), rest_length, stiffness, damping)
        item_details.append(['con', len(items), ['spring', items.index(shapes[0]), items.index(shapes[1]), rest_length, stiffness, damping]])
        items.append(spring_joint)
        space.add(spring_joint)

    else:
        sys.exit("ERROR: non-valid simulation type")

    locations = []
    timestep = 0
    while True:
        if not viz and timestep >= 200:
            break
        timestep += 1
        if viz:
            screen.fill((255,255,255))
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    sys.exit(0)

        space.step(1/50.0)

        if viz:
            space.debug_draw(draw_options)
            pygame.display.flip()
            clock.tick(50)
        else:
            step_obj = []
            for item in item_details:
                if item[0] == 'obj':
                    pos = tuple(items[item[1]].body.position)
                    vel = tuple(items[item[1]].body.velocity)
                    step_obj.append([item[1], pos, vel])
            locations.append(step_obj)
            
    if not viz:
        with open(os.path.join(data_folder,simtype + '_' + str(now) + '.pkl'), 'wb') as f:
            constraints = []
            for item in item_details:
                if item[0] == 'con':
                    constraints.append([item[1]] + item[2])
            save_data = {'obj': locations, 'con': constraints}
            pickle.dump(save_data, f)


if __name__ == '__main__':
    main()