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

import pdb

file = ''
file_id = ''
locs_data = None
con_data = None
num_obj = 0


# get an ordering of constraints based on how far they are from the given data
def get_constraint_order(scene, data):
    old_locs, snapback_locs = scene.get_snapback_locs(data)
    num_obj = len(data[0])
    cost_list = []
    for i in range(num_obj):
        cost = get_cost(snapback_locs, old_locs, obj_id=i)
        cost_list.append([cost, i])

    cost_list = sorted(cost_list, reverse=True)
    return cost_list


# metropolis-hastings algorithm
def metropolis(scene, save_state, data, ids, start, priors, mults, value_func, proposal_func, niter=MCMC_SAMPLE_NUM, nburn=0):
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


# get the value at a certain point while adding/modifying a spring, given observed data
def get_value_spring(scene, save_state, data, ids, params, whole_scene=True, eps=1e-6):
    scene.load_state(save_state)

    p_obj = [scene.bodies[i] for i in ids]
    if ids not in scene.constraints:
        scene.add_spring_constraint(p_obj[0], p_obj[1], params)
    else:
        if type(scene.constraints[ids]) == pymunk.constraint.DampedSpring:
            scene.constraints[ids].rest_length = params[0]
            scene.constraints[ids].stiffness = params[1]
        else:
            scene.remove_constraint(ids)
            scene.add_spring_constraint(p_obj[0], p_obj[1], params)

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
    return np.exp(-total_cost) * (0.9 ** len(scene.constraints))

# get the value at a certain point while adding/modifying a pin, given observed data
def get_value_pin(scene, save_state, data, ids, whole_scene=True, eps=1e-6):
    scene.load_state(save_state)

    p_obj = [scene.bodies[i] for i in ids]
    if ids not in scene.constraints:
        scene.add_pin_constraint(p_obj[0], p_obj[1])
    else:
        if type(scene.constraints[ids]) == pymunk.constraint.DampedSpring:
            scene.remove_constraint(ids)
            scene.add_pin_constraint(p_obj[0], p_obj[1])

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
    # modulation of cost by the number of constraints to minimize the number of constraints
    return np.exp(-total_cost) * (0.9 ** len(scene.constraints))


# get the probability value for a given scene, given the observed data
def get_value_scene(scene, save_state, data):
    scene.load_state(save_state)
    old_locs, new_locs = scene.get_snapback_locs(data)
    return np.exp(-get_cost(old_locs, new_locs)) * (0.9 ** len(scene.constraints))


# given a certain point, propose a new point to jump to in next iteration of MH
def make_proposal(params, multipliers, limits=[0, 0]):
    assert len(params) == len(multipliers)
    new_params = []
    for ind, p in enumerate(SPRING_PRIORS):
        while True:
            dec = np.random.normal(0,1)
            proposal_point = params[ind] + dec * multipliers[ind]
            if p.pdf(proposal_point) / p.pdf(p.median()) < np.random.random() or proposal_point < limits[ind]:
                continue
            else:
                new_params.append(proposal_point)
                break
    return new_params

# average distances of all pairs of objects 
def get_average_distances(dt):

    distances = {}
    for i in range(num_obj):
        for j in range(i + 1, num_obj):
            distances[(i,j)] = []

    for step in range(len(dt)):
        for i in range(num_obj):
            for j in range(i + 1, num_obj):
                distances[(i,j)].append(dist(dt[step][i][1], dt[step][j][1]))

    final_dict = {}
    for i in range(num_obj):
        final_dict[i] = []
    for key, val in distances.items():
        distances[key] = sum(val) / len(val)
        final_dict[key[0]].append((distances[key], key[1]))
        final_dict[key[1]].append((distances[key], key[0]))

    for key, val in final_dict.items():
        final_dict[key] = sorted(final_dict[key])

    return final_dict


# for a scene with just one constraint
def guess_single_constraint(scene, save_state, mh=True):

    objects_constrained = [i[1] for i in get_constraint_order(scene, locs_data)[:2]]
    pair = (min(objects_constrained), max(objects_constrained))

    print(scene.get_constraint_rep())

    mh_data = {}

    if mh:

        trial_data = metropolis(scene, save_state, locs_data, pair, SPRING_START, SPRING_PRIORS, SPRING_MULTS, get_value_spring, make_proposal)
        trial_best = [int(i) for i in trial_data[-1]]
        trial_value = get_value_spring(scene, save_state, locs_data, pair, trial_best)

        mh_data[pair] = trial_data
        print(trial_best)

        with open(os.path.join(os.path.dirname(file), '{}_metropolis.pkl'.format(file_id)), 'wb') as f:
            pickle.dump(mh_data, f)
            print('Produced {}_metropolis.pkl'.format(file_id))
    else:
        # load from a file and just plot
        with open(os.path.join(os.path.dirname(file), '{}_metropolis.pkl'.format(file_id)), 'rb') as f:
            mh_data = pickle.load(f)


    fig, ax = plt.subplots(1, len(mh_data), squeeze=False)
    fig.suptitle('density')
    nbins = 30
    #fig.hexbin(xys[0], xys[1], gridsize=nbins, cmap=fig.cm.BuGn_r)
    c = 0
    for pair, dt in mh_data.items():
        xys = [[],[]]
        for i in dt:
            xys[0].append(i[0])
            xys[1].append(i[1])
        ax[0,c].set_title(str(pair))
        ax[0,c].set_xlabel('rest_length')
        ax[0,c].set_ylabel('stiffness')
        ax[0,c].hist2d(xys[0], xys[1], bins=nbins, cmap=cm.BuGn_r)
        c += 1

    fig.savefig(os.path.join(os.path.dirname(file), '{}.png'.format(file_id)))
    plt.show()


def guess_scene_constraints(scene, save_state, mh=True):

    if mh:
        graph = {}
        for i in range(num_obj):
            graph[i] = [['spring', SPRING_START]] * num_obj

        values = []
        constraints = []
        best_value = 0
        best_constraints = []

        distances = get_average_distances(locs_data)

        num_per_iteration = min(4, num_obj - 1)
        num_per_iteration = 1

        iteration = 0
        while True:
            iteration += 1
            scene_prob = get_value_scene(scene, save_state, locs_data)
            print('\nIteration {} with scene probability: {}'.format(iteration, scene_prob))

            # find which objects likely involve some sort of constraint
            loop_start_time = time.time()
            objects_constrained = get_constraint_order(scene, locs_data)
            # print(objects_constrained)


            # get one object to fixate on based on probability distribution of costs
            objects_constrained_probs = [i[0] for i in objects_constrained]
            objects_constrained_probs = [i / sum(objects_constrained_probs) for i in objects_constrained_probs]
            objects_constrained_sels = [i[1] for i in objects_constrained]
            chosen_obj = np.random.choice(objects_constrained_sels, p=objects_constrained_probs)



            # get selection of other objects to make pairings based on inverse distribution of distances
            other_distances = distances[chosen_obj]
            other_probs, other_ids = zip(*other_distances)
            other_probs = [1/i for i in other_probs]
            other_probs = [i / sum(other_probs) for i in other_probs]
            #chosen_others = np.random.choice(other_ids, p=other_probs, size=num_per_iteration, replace=False)

            distance_probs = zip(other_ids, other_probs)
            constrain_probs = zip(objects_constrained_sels, objects_constrained_probs)

            mixed_probs = [1] * num_obj
            for i in distance_probs:
                mixed_probs[i[0]] *= i[1]
            for i in constrain_probs:
                mixed_probs[i[0]] *= i[1]

            other_probs_mixed = [mixed_probs[i] for i in other_ids]
            other_probs_mixed = [i / sum(other_probs_mixed) for i in other_probs_mixed]
            chosen_others = np.random.choice(other_ids, p=other_probs_mixed, size=num_per_iteration, replace=False)


            for other_obj in chosen_others:
                connection_info = graph[chosen_obj][other_obj]
                pair = (min(chosen_obj, other_obj), max(chosen_obj, other_obj))

                # get a good setting of parameters given particular constraint
                trial_data = metropolis(scene, save_state, locs_data, pair, connection_info[1], SPRING_PRIORS, SPRING_MULTS, get_value_spring, make_proposal)
                trial_best = tuple([int(i) for i in trial_data[-1]])
                trial_value = get_value_spring(scene, save_state, locs_data, pair, trial_best, whole_scene=True)

                pin_value =  get_value_pin(scene, save_state, locs_data, pair, whole_scene=True)
                if pin_value > trial_value:
                    final_constraint = 'pin'
                    on_prob = pin_value / (scene_prob + pin_value)
                else:
                    final_constraint = 'spring'
                    # print('{}\t{}\t{}'.format(pair, trial_best, trial_value))
                    on_prob = trial_value / (scene_prob + trial_value)
                    


                # update the constraint and parameters
                scene.load_state(save_state)
                # creating/editing the spring or the pin based on what we got for the best value earlier
                if np.random.random() < on_prob and trial_best[1] > 10:
                    if pair in scene.constraints:
                        scene.remove_constraint(pair)
                    b1, b2 = scene.bodies[pair[0]], scene.bodies[pair[1]]
                    if final_constraint == 'spring':
                        scene.add_spring_constraint(b1, b2, list(trial_best) + [0])
                        print('{}: spring with parameters {} and prob {}'.format(pair, trial_best, trial_value))
                        graph[chosen_obj][other_obj] = ['spring', trial_best]
                        graph[other_obj][chosen_obj] = ['spring', trial_best]
                    else:
                        scene.add_pin_constraint(b1, b2)
                        print('{}: pin with prob {}'.format(pair, pin_value))
                        graph[chosen_obj][other_obj] = ['pin', trial_best]
                        graph[other_obj][chosen_obj] = ['pin', trial_best]
                else:
                    if pair in scene.constraints:
                        # test what happens without the constraint
                        scene.remove_constraint(pair)
                        save_state = scene.save_state()
                        new_value = get_value_scene(scene, save_state, locs_data)
                        final_value = trial_value if final_constraint == 'spring' else pin_value
                        delete_prob = new_value / (new_value + final_value)
                        if np.random.random() < delete_prob:
                            # potentially delete the constraint from the graph
                            print('{}: DELETE'.format(pair))
                        else:
                            # otherwise just update with the new parameters
                            b1, b2 = scene.bodies[pair[0]], scene.bodies[pair[1]]
                            if final_constraint == 'spring':
                                scene.add_spring_constraint(b1, b2, list(trial_best) + [0])
                                print('{}: spring with parameters {} and prob {}'.format(pair, trial_best, trial_value))
                            else:
                                scene.add_pin_constraint(b1, b2)
                                print('{}: pin with prob {}'.format(pair, pin_value))
                save_state = scene.save_state()
                
                # log values and constraints
                val = get_value_scene(scene, save_state, locs_data)

                if iteration < 50:
                    threshold = 4
                else:
                    threshold = 8

                if val < best_value / threshold:
                    print('resetting')
                    scene.reset_space()
                    scene.add_bodies_from_rep(locs_data[0])
                    scene.add_constraints_from_rep(best_constraints)
                    save_state = scene.save_state()

                values.append(val)
                current_constraints = [
                    [
                        'spring' if type(scene.constraints[i]) == pymunk.constraint.DampedSpring else 'pin',
                        i[0],
                        i[1],
                        [
                            scene.constraints[i].rest_length,
                            scene.constraints[i].stiffness,
                            0
                        ] if type(scene.constraints[i]) == pymunk.constraint.DampedSpring else None
                    ] for i in scene.constraints.keys()
                ]
                constraints.append(current_constraints)
                if val > best_value:
                    best_value = val
                    best_constraints = current_constraints
                    print('NEW BEST: {}'.format(val))

            if not iteration % 5:            
                print('correct constraints: {}'.format(scene.get_constraint_rep()))
                print('current constraints: {}'.format(list(scene.constraints.keys())))
                print('best so far: {}'.format(best_constraints))
                continue
                scene.load_state(save_state)
                for con in scene.constraints:
                    if type(scene.constraints[con]) == pymunk.constraint.DampedSpring:
                        t = 'spring'
                        par = [scene.constraints[con].rest_length, scene.constraints[con].stiffness]
                    else:
                        t='pin'
                    # test what happens without the constraint
                    old_value = get_value_scene(scene, save_state, locs_data)
                    scene.remove_constraint(con)

                    save_state = scene.save_state()
                    new_value = get_value_scene(scene, save_state, locs_data)
                    #final_value = trial_value if final_constraint == 'spring' else pin_value
                    delete_prob = new_value / 2 / (new_value + old_value)
                    if np.random.random() < delete_prob:
                        # potentially delete the constraint from the graph
                        print('{}: DELETE'.format(con))
                    else:
                        # otherwise just update with the new parameters
                        b1, b2 = scene.bodies[con[0]], scene.bodies[con[1]]
                        if t == 'spring':
                            scene.add_spring_constraint(b1, b2, list(par) + [0])
                            print('{}: spring with parameters {} and prob {}'.format(con, par, trial_value))
                        else:
                            scene.add_pin_constraint(b1, b2)
                            print('{}: pin with prob {}'.format(con, pin_value))
                    save_state = scene.save_state()

            if iteration >= 100 or best_value > 0.4:
                print('Finished! Best parameters found: {}'.format(best_constraints))
                break

        mh_data = {
            'values': values,
            'constraints': constraints,
            'graph': graph,
            'best_value': best_value,
            'best_constraints': best_constraints
        }

        with open(os.path.join(os.path.dirname(file), '{}_metropolis.pkl'.format(file_id)), 'wb') as f:
            pickle.dump(mh_data, f)
            print('Produced {}/{}_metropolis.pkl'.format(os.path.dirname(file), file_id))

    else:
        # load from a file and just plot
        with open(os.path.join(os.path.dirname(file), '{}_metropolis.pkl'.format(file_id)), 'rb') as f:
            mh_data = pickle.load(f)

        con_rep = mh_data['best_constraints']
        # pdb.set_trace()
        scene.add_constraints_from_rep(con_rep)
        scene.run_and_visualize()

    values = mh_data['values']


    fig, ax = plt.subplots(1, 1)
    nbins = 20
    #fig.hexbin(xys[0], xys[1], gridsize=nbins, cmap=fig.cm.BuGn_r)
    c = 0
    ax.set_title('log probability over time')
    ax.set_xlabel('time')
    ax.set_ylabel('scene value')

    ax.set_yscale('log')

    ax.plot(values)

    fig.savefig(os.path.join(os.path.dirname(file), '{}.png'.format(file_id)))
    plt.show()




def main(args):
    global file
    global file_id
    global locs_data
    global con_data
    global num_obj
    global use_mh

    start_time = time.time()
    file = args['file']
    noise = args['noise']
    viz = args['visualize']
    use_mh = args['use_params']

    # get file id which is determined by timestamp
    file_id = os.path.basename(file).split('_')[0]
    # con_type = os.path.dirname(file)[-1]

    with open(file, 'rb') as f:
        data = pickle.load(f)


    space_data = data['space']
    locs_data = data['locs']
    # con_data = data['con']

    num_obj = len(locs_data[0])

    
    scene = SimulationScene(space=space_data, noise=noise)
    # scene.add_bodies_from_rep(locs_data[0])

    # save current state because will need to run many times later for simulations
    save_state = scene.get_state()

    if args['single']:
        guess_single_constraint(scene, save_state, not use_mh)
    else:
        guess_scene_constraints(scene, save_state, not use_mh)




def parse_args():
    parser = argparse.ArgumentParser(description='Load scene.')
    parser.add_argument('file')
    parser.add_argument('-p', '--use_params', action='store_true')
    parser.add_argument('-s', '--single', help='Use single constraint solver.', action='store_true')
    parser.add_argument('-v', '--visualize', help='Visualize system.', action='store_true')
    parser.add_argument('-n', '--noise', help='amount of noise, in [collision, dynamic] form', nargs=2, type=float,default=[0,0])
    args = vars(parser.parse_args())

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

