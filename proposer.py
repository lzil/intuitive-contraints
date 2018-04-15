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

import numpy
import matplotlib.pyplot as plt
from matplotlib import cm

from scene_tools import *
import argparse

import pymc3 as pm

import pdb


def find_candidates(scene, obj_data):
    #locations_no_constraints = scene.run_and_record(TIME_LIMIT)
    old_locs, snapback_locs = scene.get_snapback_locs(obj_data)
    num_obj = len(obj_data[0])
    cost_list = []
    for i in range(num_obj):
        cost = get_cost(snapback_locs, old_locs, obj_id=i)
        cost_list.append(cost)
    obj_sorted = list(np.argsort(np.asarray(cost_list)))
    candidates = []
    # print(obj_sorted)
    threshold = 1 #max(cost_list) / 4
    print('Threshold: {}'.format(threshold))
    for ind, cost in enumerate(cost_list):
        if cost > threshold:
            candidates.append(ind)

    return candidates


spring_priors = [
    scipy.stats.norm(100, 50),  # rest length
    scipy.stats.norm(60, 40)    # stiffness
]
spring_multipliers = (20, 5)
spring_start = (100, 60)


# metropolis-hastings algorithm
def metropolis(scene, save_state, data, ids, start, priors, mults, value_func, proposal_func, niter=10000, nburn=0):
    current = start
    # keep list of explored data points
    post = [current]
    value_current = value_func(scene, save_state, data, ids, current)
    # initial temperature
    T = .5
    #mult = 0.999
    # some mathematical expression so that there's an exponential decrease
    mult = np.exp(np.log(0.005) / niter)
    for i in range(niter + nburn):
        # propose a new point to jump to
        proposed = proposal_func(current, mults)
        value_proposed = value_func(scene, save_state, data, ids, proposed, whole_scene=False)
        # pdb.set_trace()
        # print(value_proposed, proposed)
        # pdb.set_trace()
        # calculate probability of jumpint to new point
        fb = np.log(value_current)
        fa = np.log(value_proposed)
        delta = fa - fb
        # acceptance criterion depends on tempeature
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current = proposed
            value_current = value_proposed
        T *= mult
        post.append(current)

    return post[nburn:]


# get the value at a certain point
def get_value(scene, save_state, data, ids, params, whole_scene=True, eps=1e-6):
    scene.load_state(save_state)

    p_obj = [scene.bodies[i] for i in ids]
    spring_joint = scene.add_spring_constraint(p_obj[0], p_obj[1], params)

    # use snapback rule to get cost
    old_locs, new_locs = scene.get_snapback_locs(data)

    total_cost = 0
    if whole_scene:
        total_cost = get_cost(old_locs, new_locs)
    else:
        for i in ids:
            cost = get_cost(old_locs, new_locs, i)
            total_cost += cost
        total_cost /= len(ids)
    return np.exp(-total_cost)


# given a certain point, propose a new point to jump to in next iteration of MH
def make_proposal(params, multipliers):
    assert len(params) == len(multipliers)
    new_params = []
    for ind, p in enumerate(spring_priors):
        while True:
            dec = np.random.normal(0,1)
            proposal_point = params[ind] + dec * multipliers[ind]
            if p.pdf(proposal_point) / p.pdf(p.median()) < np.random.random() or proposal_point < 0:
                continue
            else:
                new_params.append(proposal_point)
                break
    return new_params


def main(args):
    file = args['file']
    noise = args['noise']
    viz = args['visualize']

    # get file id which is determined by type of constraint and timestamp
    file_id = os.path.basename(file).split('_')[0]
    con_type = os.path.dirname(file)[-1]

    with open(file, 'rb') as f:
        data = pickle.load(f)
    obj_data = data['obj']
    con_data = data['con']

    num_obj = len(obj_data[0])

    scene = Scene(noise)
    scene.add_bodies_from_rep(obj_data[0])

    # save current state because will need to run many times later for simulations
    save_state = scene.save_state()

    while True:
        #pdb.set_trace()

        # find which objects likely involve some sort of constraint
        objects_constrained = find_candidates(scene, obj_data)
        # sys.exit()

        print('Constrained objects: {}'.format(objects_constrained))
        if len(objects_constrained) < 2:
            print("Only {} above threshold; can't create constraints.".format(len(objects_constrained)))
            sys.exit(0)
        

        do_mh = True
        mh_data = {}
        if do_mh:
            candidate_pairs = []

            # creating list of possible constraint pairs: candidate_pairs
            cons = [(i[1], i[2]) for i in scene.get_constraint_rep()]
            num_constrained = len(objects_constrained)
            scene.load_state(save_state)
            for i in range(num_constrained): 
                for j in range(i + 1, num_constrained):
                    obj1 = scene.bodies[objects_constrained[i]]
                    obj2 = scene.bodies[objects_constrained[j]]
                    d = dist(obj1.position, obj2.position)
                    if (objects_constrained[i], objects_constrained[j]) not in cons and (objects_constrained[j], objects_constrained[i]) not in cons:
                        candidate_pairs.append((d, objects_constrained[i], objects_constrained[j]))
            candidate_pairs = sorted(candidate_pairs)


            scene.load_state(save_state)
            old_locs, new_locs = scene.get_snapback_locs(obj_data)
            scene_prob = np.exp(-get_cost(old_locs, new_locs))

            
            # print(candidate_pairs)
            print(con_data)
            print('Cost of scene without new constraints: {}'.format(scene_prob))

            pair_parameters = []

            print('Pair \tData   \tValue')
            for pair_tuple in candidate_pairs:
                pair = pair_tuple[1:]

                # run them MH algorithm with some number of steps
                trial_data = metropolis(scene, save_state, obj_data, pair, spring_start, spring_priors, spring_multipliers, get_value, make_proposal, niter=40)
                trial_data_round = [int(i) for i in trial_data[-1]]
                trial_value = get_value(scene, save_state, obj_data, pair, trial_data_round)
                print('{}\t{}\t{}'.format(pair, trial_data_round, trial_value))

                # save the data from this run
                mh_data[pair] = trial_data
                pair_parameters.append([trial_value, list(pair), trial_data_round])

            pair_parameters = sorted(pair_parameters, reverse=True)
            # best_scene_prob = get_value(scene, save_state, obj_data, None, None)

            if pair_parameters[0][0] > scene_prob:
                best_pair = pair_parameters[0][1]
                best_params = pair_parameters[0][2]
                print('Best pair: {} with parameters {} and prob {}'.format(best_pair, best_params, pair_parameters[0][0]))
                scene.load_state(save_state)
                scene.add_spring_constraint(scene.bodies[best_pair[0]], scene.bodies[best_pair[1]], best_params + [0])
                save_state = scene.save_state()
            else:
                print('Best pair prob {} is less than scene prob {}'.format(pair_parameters[0][0], scene_prob))
                break
                
                
            with open(os.path.join(os.path.dirname(file), '{}_metropolis.pkl'.format(file_id)), 'wb') as f:
                pickle.dump(mh_data, f)
                print('Produced {}_metropolis.pkl'.format(file_id))
        else:
            # load from a file and just plot
            with open(os.path.join(os.path.dirname(file), '{}_metropolis.pkl'.format(file_id)), 'rb') as f:
                mh_data = pickle.load(f)


    # graph the data from this run
    
    

    fig, ax = plt.subplots(1, len(mh_data))
    fig.suptitle('density')
    #fig.ylabel('damping')
    nbins = 20
    #fig.hexbin(xys[0], xys[1], gridsize=nbins, cmap=fig.cm.BuGn_r)
    c = 0
    for pair, dt in mh_data.items():
        xys = [[],[]]
        for i in dt:
            xys[0].append(i[0])
            xys[1].append(i[1])
        ax[c].set_title(str(pair))
        ax[c].set_xlabel('rest_length')
        ax[c].set_ylabel('stiffness')
        ax[c].hist2d(xys[0], xys[1], bins=nbins, cmap=cm.BuGn_r)
        c += 1

    fig.savefig('metro.png')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Load scene.')
    parser.add_argument('file')
    parser.add_argument('-v', '--visualize', help='Visualize system.', action='store_true')
    parser.add_argument('-n', '--noise', help='amount of noise, in [collision, dynamic] form', nargs=2, type=float,default=[0,0])
    args = vars(parser.parse_args())

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

