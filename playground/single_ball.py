
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

def add_ball(space, mass, radius, position, velocity=(0,0), angular_velocity=0):
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
        pygame.display.set_caption("Ball moving")
        clock = pygame.time.Clock()
        print(clock)
        draw_options = pymunk.pygame_util.DrawOptions(screen)

    space = pymunk.Space()
    space.gravity = (0.0, 0.0)

    objects = []

    # ticks_to_next_ball = 10

    if simtype == 'inertial':
        x_pos = random.randint(200,800)
        y_pos = random.randint(200,800)
        x_vel = random.randint(-500,500)
        y_vel = random.randint(-500,500)
        shape = add_ball(space, 1, 20, (x_pos, y_pos), (x_vel, y_vel), 0)

        objects.append(shape)

    elif simtype == 'circular':
        rotation_center_body = pymunk.Body(body_type = pymunk.Body.STATIC)
        joint_x = random.randint(300,700)
        joint_y = random.randint(300,700)
        rotation_center_body.position = (joint_x,joint_y)

        x_pos = random.randint(200,800)
        y_pos = random.randint(200,800)
        vel = random.randint(200,1200)

        string = (joint_x - x_pos, joint_y - y_pos)
        mag = math.sqrt(string[0] ** 2 + string[1] ** 2)
        velocity = (string[1] / mag * vel, string[0] / mag * vel)
        shape = add_ball(space, 1, 20, (x_pos, y_pos), velocity, 0)

        rotation_center_joint = pymunk.PinJoint(shape.body, rotation_center_body, (0,0), (0,0))
        space.add(rotation_center_joint)

        objects.append(shape)

    elif simtype == 'harmonic':
        harmonic_center_body = pymunk.Body(body_type = pymunk.Body.STATIC)
        joint_x = random.randint(300,700)
        joint_y = random.randint(300,700)
        harmonic_center_body.position = (joint_x,joint_y)

        x_pos = random.randint(300,700)
        y_pos = random.randint(300,700)
        vel = random.randint(2,20)

        string = (joint_x - x_pos, joint_y - y_pos)
        mag = math.sqrt(string[0] ** 2 + string[1] ** 2)
        velocity = (string[1] / mag * vel, string[0] / mag * vel)
        shape = add_ball(space, 1, 20, (x_pos, y_pos), velocity, 0)

        stiffness = random.randint(10,120)
        harmonic_center_joint = pymunk.DampedSpring(shape.body, harmonic_center_body, (0,0), (0,0), 0, stiffness, 0)
        space.add(harmonic_center_joint)

        objects.append(shape)

    else:
        sys.exit("ERROR: non-valid simulation type")

    locations = []
    timestep = 0
    while True:
        if not viz and timestep >= 100:
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
        locations.append([obj.copy() for obj in objects])


        # objects_to_remove = []
        # for obj in objects:
        #     print (obj)
        #     if obj.body.position.y > 150:
        #         objects_to_remove.append(obj)

        # for obj in objects_to_remove:
        #     space.remove(obj, obj.body)
        #     objects.remove(obj)
        if viz:
            space.debug_draw(draw_options)
            pygame.display.flip()
            clock.tick(50)

    with open(os.path.join(data_folder,simtype + '_' + str(now) + '.pkl'), 'wb') as f:
        pickle.dump(locations, f)


if __name__ == '__main__':
    main()