
import sys, random
import os

import pymunk
import pymunk.pygame_util
import math
import numpy as np


import pygame
from pygame.locals import *
from pygame.color import *

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
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    pygame.display.set_caption("Ball moving")
    clock = pygame.time.Clock()

    space = pymunk.Space()
    space.gravity = (0.0, 0.0)

    objects = []
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    

    rotation_center_body = pymunk.Body(body_type = pymunk.Body.STATIC)
    joint_x = random.randint(300,700)
    joint_y = random.randint(300,700)
    rotation_center_body.position = (joint_x,joint_y)


    x_pos = random.randint(300,700)
    y_pos = random.randint(300,700)

    vel = random.randint(200,1200)

    string = (joint_x - x_pos, joint_y - y_pos)
    mag = math.sqrt(string[0] ** 2 + string[1] ** 2)
    velocity = (string[1] / mag * vel, string[0] / mag * vel)

    shape = add_ball(space, 1, 20, (x_pos, y_pos), velocity, 0)

    rotation_center_joint = pymunk.PinJoint(shape.body, rotation_center_body, (0,0), (0,0))
    space.add(rotation_center_joint)

    
    objects.append(shape)
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)

        space.step(1/50.0)
        screen.fill((255,255,255))

        # objects_to_remove = []
        # for obj in objects:
        #     print (obj)
        #     if obj.body.position.y > 150:
        #         objects_to_remove.append(obj)

        # for obj in objects_to_remove:
        #     space.remove(obj, obj.body)
        #     objects.remove(obj)
        space.debug_draw(draw_options)

        pygame.display.flip()
        clock.tick(50)


if __name__ == '__main__':
    main()