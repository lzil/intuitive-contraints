
import sys, random
import os
import time
import math

import pymunk
import pymunk.pygame_util

import pickle

import numpy as np
import matplotlib.pyplot as plt

from scene_tools import *
import argparse

def main(args):
    con_type = args['constraint']
    num_constraints = args['num_constraints']
    num_obj = args['balls']
    viz = args['visualize']
    reps = args['number']
    noise = args['noise']

    scene = Scene(noise)

    for _ in range(reps):
        
        scene.reset_space()
        # generate some random positions of balls in the scene
        pos = np.random.randint(300, 700, [num_obj, 2])
        vel = np.random.randint(-500, 500, [num_obj, 2])
        bodies = [scene.add_ball(pos[i], vel[i]) for i in range(num_obj)]

        # choose two shapes to be constrained by something
        if num_constraints > num_obj * (num_obj - 1) / 2:
            print('You want too many constraints. Exiting')
            sys.exit()

        constraints = []
        for i in range(num_constraints):
            while True:
                cs = set(np.random.choice(bodies, 2, False))
                if cs not in constraints:
                    constraints.append(cs)
                    break
        print(constraints)
        # if num_obj <= 1:
        #     con_type = 'none'
        # else:
        #     cs = np.random.choice(bodies, 2, False)

        for cs in constraints:
            cs = list(cs)
            if con_type == 'none':
                pass
            elif con_type == 'pin':
                scene.add_pin_constraint(cs[0], cs[1])
            elif con_type == 'slide':
                scene.add_slide_constraint(cs[0], cs[1])
            elif con_type == 'spring':
                scene.add_spring_constraint(cs[0], cs[1])
            # elif con_type == 'func':
            #     scene.add_func_constraint(cs[0], cs[1])
        # else:
        #     raise ValueError("non-valid constraint type")

        # run the physics simulation forward
        if viz:
            locations = scene.run_and_visualize(label='multiple_ball')
        else:
            locations = scene.run_and_record(steps=TIME_LIMIT)

            assert len(locations) == TIME_LIMIT

        # save the simulation data; id is current time
        now = int(time.time())

        stimuli_folder = os.path.join('stimuli', con_type) 
        if not os.path.exists(stimuli_folder):
            os.makedirs(stimuli_folder)

        plots_folder = os.path.join('plots', con_type)
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        dt_name = os.path.join(stimuli_folder, '{}_real.pkl'.format(now))
        with open(dt_name, 'wb') as f:
            constraints = scene.get_constraint_rep()
            save_data = {'obj': locations, 'con': constraints}
            pickle.dump(save_data, f)
            print('Saved {}'.format(dt_name))
            # print(save_data)


        # get mutual distances and plot them
        distances = {}
        for t in locations:
            for i in range(num_obj):
                for j in range(i + 1, num_obj):
                    d = dist(t[i][1], t[j][1])
                    if (i,j) in distances:
                        distances[(i,j)].append(d)
                    else:
                        distances[(i,j)] = [d]

        fig = plt.figure()
        plt.title('pairwise object distances')
        plt.xlabel('timestep')
        plt.ylabel('distance')
        for pair, d in distances.items():
            plt.plot(d, label=str(pair))
        plt.legend(loc='lower right')
        fig_name = os.path.join(plots_folder, '{}_real.png'.format(now))
        print('Saved ' + fig_name)
        plt.savefig(fig_name)

    

def parse_args():
    parser = argparse.ArgumentParser(description='Constraint system arguments.')
    parser.add_argument('constraint', help='Type of constraint system.')
    parser.add_argument('-b', '--balls', help='Number of balls.', type=int)
    parser.add_argument('-c', '--num_constraints', help='Number of constraints', type=int, default=1)
    parser.add_argument('-v', '--visualize', help='Visualize system.', action='store_true')
    parser.add_argument('-n', '--noise', help='amount of noise, in [collision, dynamic] form', nargs=2, type=float, default=[0,0])
    parser.add_argument('--number', type=int, help='Number of scenes to generate.', default=1)
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

