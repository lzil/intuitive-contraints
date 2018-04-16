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


def get_constraint_order(scene, obj_data):
    #locations_no_constraints = scene.run_and_record(TIME_LIMIT)
    old_locs, snapback_locs = scene.get_snapback_locs(obj_data)
    num_obj = len(obj_data[0])
    cost_list = []
    for i in range(num_obj):
        cost = get_cost(snapback_locs, old_locs, obj_id=i)
        cost_list.append([cost, i])
    # obj_sorted = list(np.argsort(np.asarray(cost_list)))
    # candidates = []
    # print(obj_sorted)
    # threshold = 1 #max(cost_list) / 4
    # print('Threshold: {}'.format(threshold))
    # for ind, cost in enumerate(cost_list):
    #     if cost > threshold:
    #         candidates.append([cost, ind])

    cost_list = sorted(cost_list, reverse=True)
    return cost_list


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
    value_current = value_func(scene, save_state, data, ids, current, whole_scene=False)
    post = [current]
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
        # print(np.exp(delta), np.exp(delta/T))
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current = proposed
            value_current = value_proposed

        T *= mult
        post.append(current)

    return post[nburn:]


# get the value at a certain point
def get_value(scene, save_state, data, ids, params, whole_scene=True, eps=1e-6):
    scene.load_state(save_state)

    if ids not in scene.constraints:
        p_obj = [scene.bodies[i] for i in ids]
        spring_joint = scene.add_spring_constraint(p_obj[0], p_obj[1], params)
    else:
        scene.constraints[ids].rest_length = params[0]
        scene.constraints[ids].stiffness = params[1]

    # use snapback rule to get cost
    old_locs, new_locs = scene.get_snapback_locs(data)

    total_cost = 0
    if whole_scene:
        total_cost = get_cost(old_locs, new_locs)
    else:
        # total_cost = get_cost(old_locs, new_locs, ids)
        for i in ids:
            cost = get_cost(old_locs, new_locs, i)
            total_cost += cost
        total_cost /= len(ids)
    return np.exp(-total_cost)

def get_scene_value(scene, save_state, data):
    scene.load_state(save_state)
    old_locs, new_locs = scene.get_snapback_locs(data)
    return np.exp(-get_cost(old_locs, new_locs))


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


def get_average_distances(dt):
    num_objs = len(dt[0])

    distances = {}
    for i in range(num_objs):
        for j in range(i + 1, num_objs):
            distances[(i,j)] = []

    for step in range(len(dt)):
        for i in range(num_objs):
            for j in range(i + 1, num_objs):
                distances[(i,j)].append(dist(dt[step][i][1], dt[step][j][1]))

    final_dict = {}
    for i in range(num_objs):
        final_dict[i] = []
    for key, val in distances.items():
        distances[key] = sum(val) / len(val)
        final_dict[key[0]].append((distances[key], key[1]))
        final_dict[key[1]].append((distances[key], key[0]))

    for key, val in final_dict.items():
        final_dict[key] = sorted(final_dict[key])

    return final_dict

def guess_single_constraint():
        distances = get_average_distances(obj_data)
        # print(distances)
        # sys.exit()

        num_objs = 0
        candidate_pairs = []
        for _, obj in objects_constrained:
            num_objs += 1
            candidate_pairs += list(filter(lambda x: obj in x[1] and x[0] < 300 and x[1] not in [y[1] for y in candidate_pairs], distances))
            # print(candidate_pairs)
            if num_objs >= 2:
                break

        scene.load_state(save_state)
        old_locs, new_locs = scene.get_snapback_locs(obj_data)
        scene_prob = np.exp(-get_cost(old_locs, new_locs))

        
        # print(candidate_pairs)
        print(con_data)
        print('Cost of scene without new constraints: {}'.format(scene_prob))

        pair_parameters = []

        print('Pair \tData   \tValue')
        for pair_tuple in candidate_pairs:
            pair = pair_tuple[1]

            # run them MH algorithm with some number of steps
            trial_data = metropolis(scene, save_state, obj_data, pair, spring_start, spring_priors, spring_multipliers, get_value, make_proposal, niter=60)
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
            #added_constraints.append(best_pair)
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

def main(args):
    start_time = time.time()
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

    print_time('startup time', start_time)

    distances = get_average_distances(obj_data)

    graph = {}

    best_value = 0
    best_constraints = []
    values = []
    constraints = []

    iteration = 0
    while True:
        iteration += 1
        #pdb.set_trace()

        # find which objects likely involve some sort of constraint
        loop_start_time = time.time()
        objects_constrained = get_constraint_order(scene, obj_data)
        # print(objects_constrained)
        # print_time('objects_constrained', loop_start_time)
        # sys.exit()

        # print('Constrained objects: {}'.format(objects_constrained))
        # if len(objects_constrained) < 2:
        #     print("Only {} above threshold; can't create constraints.".format(len(objects_constrained)))
        #     sys.exit(0)

        # get one object to fixate on based on probability distribution of costs
        objects_constrained_probs = [i[0] for i in objects_constrained]
        objects_constrained_probs = [i / sum(objects_constrained_probs) for i in objects_constrained_probs]
        objects_constrained_sels = [i[1] for i in objects_constrained]
        # print(objects_constrained_sels, objects_constrained_probs)
        chosen_obj = np.random.choice(objects_constrained_sels, p=objects_constrained_probs)



        scene.load_state(save_state)
        old_locs, new_locs = scene.get_snapback_locs(obj_data)
        scene_prob = np.exp(-get_cost(old_locs, new_locs))
        # print(con_data)
        print('\nIteration {} with scene probability: {}'.format(iteration, scene_prob))


        # chosen_obj = 6
        # print(distances)
        if chosen_obj not in graph:
            graph[chosen_obj] = [[None, (100, 60)]] * num_obj
        other_distances = distances[chosen_obj]
        other_probs, other_ids = zip(*other_distances)
        other_probs = [1/i for i in other_probs]
        other_probs = [i / sum(other_probs) for i in other_probs]
        chosen_others = np.random.choice(other_ids, p=other_probs, size=4, replace=False)
        # print(obj_probs, obj_ids)
        for other_obj in chosen_others:
            if other_obj not in graph:
                graph[other_obj] = [[None, (100, 60)]] * num_obj
            connection_info = graph[chosen_obj][other_obj]
            if other_obj == chosen_obj:
                continue
            pair = (min(chosen_obj, other_obj), max(chosen_obj, other_obj))


            # run them MH algorithm with some number of steps
            trial_data = metropolis(scene, save_state, obj_data, pair, connection_info[1], spring_priors, spring_multipliers, get_value, make_proposal, niter=10)
            trial_best = tuple([int(i) for i in trial_data[-1]])
            trial_value = get_value(scene, save_state, obj_data, pair, trial_best, whole_scene=True)
            # print('{}\t{}\t{}'.format(pair, trial_best, trial_value))
            # sys.exit()

            on_prob = trial_value / (scene_prob + trial_value)
            graph[chosen_obj][other_obj] = [on_prob, trial_best]
            graph[other_obj][chosen_obj] = [on_prob, trial_best]
            scene.load_state(save_state)
            if np.random.random() < on_prob and trial_best[1] > 10:
                
                
                if pair in scene.constraints:
                    scene.constraints[pair].rest_length = trial_best[0]
                    scene.constraints[pair].stiffness = trial_best[1]
                    print('Updating {} with parameters {} and prob {}'.format(pair, trial_best, trial_value))
                else:
                    print('Adding {} with params {} and prob {}'.format(pair, trial_best, trial_value))
                    scene.add_spring_constraint(scene.bodies[pair[0]], scene.bodies[pair[1]], list(trial_best) + [0])
                    #added_constraints.append(best_pair)
            else:
                if pair in scene.constraints:
                    scene.remove_constraint(pair)
                    save_state = scene.save_state()
                    new_value = get_scene_value(scene, save_state, obj_data)
                    delete_prob = new_value / (new_value + trial_value)
                    if np.random.random() > delete_prob:
                        scene.add_spring_constraint(scene.bodies[pair[0]], scene.bodies[pair[1]], list(trial_best) + [0])
                        print('Updating {} with parameters {} and prob {}'.format(pair, trial_best, trial_value))
                    else:
                        print('Deleting {}'.format(pair))
            save_state = scene.save_state()
            
            val = get_scene_value(scene, save_state, obj_data)
            values.append(val)
            current_constraints = [(i, scene.constraints[i].rest_length, scene.constraints[i].stiffness) for i in scene.constraints.keys()]
            constraints.append(current_constraints)
            if val > best_value:
                best_value = val
                best_constraints = current_constraints
                print('New best found with prob {}'.format(val))
        if not iteration % 5:
            # print('new scene probability is {}'.format(scene_prob))
            
            print('correct constraints: {}'.format(con_data))
            print('current constraints: {}'.format(list(scene.constraints.keys())))
            print('best so far: {}'.format(best_constraints))

        if iteration > 50 or best_value > 0.5:
            break


            # save the data from this run
            #mh_data[pair] = trial_data
            #pair_parameters.append([trial_value, list(pair), trial_data_round])

        # print(objects_constrained_probs)

        #for _, obj in objects_constrained:


        
        # added_constraints = []

        do_mh = True
        mh_data = {}
        if do_mh:

            distances = get_average_distances(obj_data)
            # print(distances)
            # sys.exit()

            num_objs = 0
            candidate_pairs = []
            for _, obj in objects_constrained:
                num_objs += 1
                candidate_pairs += list(filter(lambda x: obj in x[1] and x[0] < 300 and x[1] not in [y[1] for y in candidate_pairs], distances))
                # print(candidate_pairs)
                if num_objs >= 2:
                    break
                # sys.exit()




            # candidate_pairs = []

            # # creating list of possible constraint pairs: candidate_pairs
            # cons = [(i[1], i[2]) for i in scene.get_constraint_rep()]
            # num_constrained = len(objects_constrained)
            # scene.load_state(save_state)
            # for i in range(num_constrained): 
            #     for j in range(i + 1, num_constrained):
            #         obj1 = scene.bodies[objects_constrained[i]]
            #         obj2 = scene.bodies[objects_constrained[j]]
            #         d = dist(obj1.position, obj2.position)
            #         if (objects_constrained[i], objects_constrained[j]) not in cons and (objects_constrained[j], objects_constrained[i]) not in cons:
            #             candidate_pairs.append((d, objects_constrained[i], objects_constrained[j]))
            # candidate_pairs = sorted(candidate_pairs)


            scene.load_state(save_state)
            old_locs, new_locs = scene.get_snapback_locs(obj_data)
            scene_prob = np.exp(-get_cost(old_locs, new_locs))

            
            # print(candidate_pairs)
            print(con_data)
            print('Cost of scene without new constraints: {}'.format(scene_prob))

            pair_parameters = []

            print('Pair \tData   \tValue')
            for pair_tuple in candidate_pairs:
                pair = pair_tuple[1]

                # run them MH algorithm with some number of steps
                trial_data = metropolis(scene, save_state, obj_data, pair, spring_start, spring_priors, spring_multipliers, get_value, make_proposal, niter=60)
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
                #added_constraints.append(best_pair)
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

