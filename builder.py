
import sys, random
import os
import time
import math

import pymunk
import pymunk.pygame_util
import pygame

import pickle

import numpy as np
import matplotlib.pyplot as plt

from scene_tools import *
import argparse

def main():

    b_scene = BuilderScene()

    pygame.init()
    screen = pygame.display.set_mode(b_scene.size) 
    # clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    running = True
    clicked_body = None
    modes = ['pin', 'spring (short)', 'spring (medium)', 'spring (long)']
    mode = 0
    shps = ['circle', 'rectangle']
    shp = 0

    orig_drag_pos = (0,0)

    font = pygame.font.Font(None, 36)
    text_setting = font.render('Setting (m): ', True, THECOLORS['grey20'])
    text_shape = font.render('Shape (n): ', True, THECOLORS['grey20'])
    

    candidates = []

    while running:
        screen.fill(THECOLORS['white'])
        b_scene.space.debug_draw(draw_options)

        screen.blit(text_setting,(35,975))
        screen.blit(text_shape,(535,975))
        text_setting_actual = font.render(modes[mode], True, THECOLORS['steelblue'])
        text_shape_actual = font.render(shps[shp], True, THECOLORS['steelblue'])
        screen.blit(text_setting_actual,(195,975))
        screen.blit(text_shape_actual,(695,975))

        pos = pymunk.pygame_util.from_pygame(pygame.mouse.get_pos(), screen)
        near_query = b_scene.space.point_query_nearest(pos, 0, pymunk.ShapeFilter())
        if pos[0] < b_scene.size[0] - 20 and pos[0] > 20 and pos[1] < b_scene.size[1] - 20 and pos[1] > 20:
            within_bounds = True
        else:
            within_bounds = False

        shape = None
        if near_query is not None:
            shape = near_query.shape
            if shape.body.body_type != pymunk.Body.STATIC:
                p = pymunk.pygame_util.to_pygame(shape.body.position, screen)
                pygame.draw.circle(screen, THECOLORS["orange"], p, 5, 0)

        if clicked_body is not None:
            clicked_body.position = sub_vec(pos, sub_vec(start_pos,orig_drag_pos))
            b_scene.space.reindex_shapes_for_body(clicked_body)

        for event in pygame.event.get():
            if event.type == QUIT or \
                event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):  
                running = False
            elif event.type == KEYDOWN:
                if event.key is K_m: # change the constraint type that is produced by clicking
                    mode += 1
                    if mode >= len(modes):
                        mode = 0
                elif event.key is K_n: # change the shape that is produced
                    shp += 1
                    if shp >= len(shps):
                        shp = 0
                elif event.key is K_s: # save the space as a _real.pkl file
                    b_scene.set_body_data()
                    save_space = b_scene.get_space()
                    s_scene = SimulationScene(space=save_space)
                    locations = s_scene.run_and_record(steps=TIME_LIMIT)
                    now = int(time.time())

                    stimuli_folder = os.path.join('stimuli') 
                    if not os.path.exists(stimuli_folder):
                        os.makedirs(stimuli_folder)

                    data_name = os.path.join(stimuli_folder, '{}_real.pkl'.format(now))
                    with open(data_name, 'wb') as f:
                        save_data = {'space': save_space, 'locs': locations}
                        pickle.dump(save_data, f)
                        print('Saved {}'.format(data_name))

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and within_bounds:
                start_time = time.time()
                start_pos = pos
                
                if shape is not None:
                    clicked_body = shape.body
                    orig_drag_pos = clicked_body.position
                    start_onshape = True
                else:
                    start_onshape = False
                    

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                end_time = time.time()
                total_time = end_time - start_time
                if start_onshape:
                    clicked_body = None
                    if pos == start_pos:
                        shape.color = THECOLORS['cadetblue']
                        candidates.append(shape.body)
                        if len(candidates) == 2:
                            for c in candidates:
                                list(c.shapes)[0].color = THECOLORS['lightblue']
                            if candidates[0] != candidates[1]:
                                b_scene.remove_constraint(candidates[0], candidates[1])
                                b_scene.add_constraint_by_mode(candidates[0], candidates[1], mode)
                            candidates = []

                elif shape is None:
                    if len(candidates) == 1:
                        list(candidates[0].shapes)[0].color = THECOLORS['lightblue']
                        b_scene.add_background_constraint(candidates[0], mode, pos)
                        candidates = []
                    elif shp == 0:
                        if total_time >= 0.1:
                            r = max(10, abs((pos[0] - start_pos[0])) / 2)
                            center = avg_vec(start_pos, pos)
                            b_scene.add_ball(center, r=r)
                        else:
                            b_scene.add_ball(pos)
                    elif shp == 1:
                        if total_time >= 0.1:
                            b_scene.add_block(start_pos, pos)
                        else:
                            b_scene.add_block(pos)
                        
                

                
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3 and within_bounds:
                if near_query is not None:
                    shape = near_query.shape
                    for con in shape.body.constraints:
                        if con in b_scene.space.constraints:
                            b_scene.space.remove(con)
                    if shape.body in candidates:
                        candidates = []
                    b_scene.space.remove(shape.body, shape)
        
        pygame.display.flip()

    # b_scene.run_builder()


main()