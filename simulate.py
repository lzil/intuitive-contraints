
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
    con_type = 'mixed'

    num_obj = args['balls']
    viz = args['visualize']
    reps = args['number']
    noise = args['noise']

    max_degree = args['max_degree']
    if max_degree == None:
        max_degree = num_obj - 1
    num_constraints = args['num_constraints']
    assert num_constraints >= 0 and num_constraints <= num_obj * (num_obj - 1) / 2 and num_constraints <= num_obj * max_degree // 2
    num_pin_constraints = args['num_pin_constraints']
    if num_pin_constraints is None:
        num_pin_constraints = np.random.randint(0, num_constraints)
    assert num_pin_constraints >= 0 and num_pin_constraints <= num_constraints
    num_spring_constraints = num_constraints - num_pin_constraints


    scene = Scene(noise)

    for _ in range(reps):
        
        scene.reset_space()
        # generate some random positions of balls in the scene
        pos = np.random.randint(350, 650, [num_obj, 2])
        vel = np.random.randint(-200, 200, [num_obj, 2])
        bodies = [scene.add_ball(pos[i], vel[i]) for i in range(num_obj)]

        distances = []
        for i in range(num_obj):
            for j in range(i + 1, num_obj):
                distances.append((dist(bodies[i].position, bodies[j].position), (i, j)))

        other_probs, other_ids = zip(*distances)
        other_probs = [1/i for i in other_probs]
        other_probs = [i / sum(other_probs) for i in other_probs]
        chosen_constraints = [other_ids[i] for i in np.random.choice(len(other_probs), p=other_probs, size=len(other_probs), replace=False)]

        use_counts = [0] * num_obj
        filtered_constraints = []
        for con in chosen_constraints:
            if use_counts[con[0]] + 1 <= max_degree and use_counts[con[1]] + 1 <= max_degree:
                use_counts[con[0]] += 1
                use_counts[con[1]] += 1
                filtered_constraints.append(con)

        final_constraints = filtered_constraints[:num_constraints]
        #random.shuffle(final_constraints)

        for i in range(num_pin_constraints):
            pair = final_constraints[i]
            obj1, obj2 = [scene.bodies[j] for j in pair]
            scene.add_pin_constraint(obj1, obj2)
        for i in range(num_spring_constraints):
            pair = final_constraints[num_pin_constraints + i]
            obj1, obj2 = [scene.bodies[j] for j in pair]
            scene.add_spring_constraint(obj1, obj2)


        # constraints = []
        # for i in range(num_constraints):
        #     while True:
        #         cs = set(np.random.choice(bodies, 2, False))
        #         use_count = [0,0]
        #         cs_list = list(cs)
        #         for j in constraints:
        #             if cs_list[0] in j:
        #                 use_count[0] += 1
        #             if cs_list[1] in j:
        #                 use_count[1] += 1
        #         if use_count[0] >= max_degree or use_count[1] >= max_degree:
        #             continue
                        
        #         if cs not in constraints:
        #             constraints.append(cs)
        #             break

        # for i in range(num_pin_constraints):
        #     cs = list(constraints[i])
        #     scene.add_pin_constraint(cs[0], cs[1])
        # for i in range(num_spring_constraints):
        #     cs = list(constraints[num_pin_constraints + i])
        #     scene.add_spring_constraint(cs[0], cs[1])

        # run the physics simulation forward
        if viz:
            locations = scene.run_and_visualize(label='constraints simulation')
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
    parser.add_argument('-f', '--file', help='Input json path')
    parser.add_argument('-b', '--balls', help='Number of balls.', type=int)
    parser.add_argument('-c', '--num_constraints', help='Number of constraints', type=int, default=1)
    parser.add_argument('-p', '--num_pin_constraints', help='Number of pin constraints', type=int)
    parser.add_argument('-d', '--max_degree', help='Maximum degree per object', type=int)
    parser.add_argument('-v', '--visualize', help='Visualize system.', action='store_true')
    parser.add_argument('-n', '--noise', help='amount of noise, in [collision, dynamic] form', nargs=2, type=float, default=[0,0])
    parser.add_argument('--number', type=int, help='Number of scenes to generate.', default=1)
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

