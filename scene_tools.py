import sys, random
import os
import time
import math

import pymunk
import pymunk.pygame_util

import pickle

import pdb

import pygame
from pygame.locals import *
from pygame.color import *

import numpy as np

import scipy.stats

# limit for artificial simulation of scenes
TIME_LIMIT = 360

TICK = 20
MCMC_SAMPLE_NUM = 20
GRAVITY = 0 #-2000.0
ELASTICITY = 0.99

SPRING_PRIORS = [
    scipy.stats.norm(50, 50),  # rest length
    scipy.stats.norm(90, 50)    # stiffness
]
SPRING_LIMITS = ((0, 100), (40, 140))
SPRING_MULTS = (5, 5)
SPRING_START = (50, 90)



# manages the space, bodies, walls, and running of physics all in one class
class Scene:
    def __init__(self, noise=(0,0), walls=True):
        self.space = pymunk.Space()
        self.bodies = []
        self.space.gravity = (0.0, GRAVITY)
        self.size = [1000, 1000]
        self.collision_types = {
            'wall': 1,
            'ball': 2,
            'ball_red': 3
        }
        self.noise = {
            'collision': noise[0],
            'dynamic': noise[1]
        }
        self.verbose = False
        self.update_collision_handler()
        if walls:
            self.add_walls()

        self.constraints = {}
        # self.func_constraints = []

    # add walls on the four borders of the space
    def add_walls(self):
        bottom_wall = pymunk.Segment(self.space.static_body, (0,0), (self.size[0],0), 40)
        left_wall = pymunk.Segment(self.space.static_body, (0,0), (0,self.size[1]), 40)
        top_wall = pymunk.Segment(self.space.static_body, (0,self.size[1]), (self.size[0],self.size[1]), 40)
        right_wall = pymunk.Segment(self.space.static_body, (self.size[0],self.size[1]), (self.size[0],0), 40)
        walls = [bottom_wall, left_wall, top_wall, right_wall]
        for w in walls:
            w.friction = 0.6
            w.color = THECOLORS['grey']
        for wall in walls:
            wall.elasticity = ELASTICITY
        self.space.add([walls])

    # add a small ball shape and body to the space
    def add_ball(self, pos, vel=(0,0), col_type='ball'):
        mass = 1
        radius = 20
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = pos
        body.velocity = vel
        shape = pymunk.Circle(body, radius)
        shape.collision_type = self.collision_types[col_type]
        shape.color = THECOLORS['lightblue']
        if col_type == 'ball_red':
            shape.color=(255,100,100,0)
        shape.elasticity = ELASTICITY
        self.space.add(body, shape)
        self.bodies.append(body)
        return body

    def add_pin_constraint(self, b1, b2, params=None):
        if self.verbose:
            print('Adding pin to {}-{}'.format(self.bodies.index(b1), self.bodies.index(b2)))
        pin_joint = pymunk.PinJoint(b1, b2, (0,0), (0,0))
        indices = (self.bodies.index(b1), self.bodies.index(b2))
        self.constraints[(min(indices), max(indices))] = pin_joint
        self.space.add(pin_joint)
        return pin_joint

    # def add_slide_constraint(self, b1, b2, params=None):
    #     if params == None:
    #         cur_dist = dist(b1.position, b2.position)
    #         params = [random.random() * cur_dist, (1 + random.random()) * cur_dist]

    #     if self.verbose:
    #         print('Adding slide to {}-{}; {}, {}'.format(self.bodies.index(b1), self.bodies.index(b2), params[0], params[1]))
    #     slide_joint = pymunk.SlideJoint(b1, b2, (0,0), (0,0), params[0], params[1])
    #     self.space.add(slide_joint)
    #     return slide_joint

    def add_spring_constraint(self, b1, b2, params=None):
        if params == None:
            rest_length = np.random.randint(SPRING_LIMITS[0][0], SPRING_LIMITS[0][1])
            stiffness = np.random.randint(SPRING_LIMITS[1][0], SPRING_LIMITS[1][1])
            # damping = np.random.gamma(1, 0.5)
            damping = 0
            params = [rest_length, stiffness, damping]
        if self.verbose:
            print('Adding spring to {}-{}; {}, {}'.format(self.bodies.index(b1), self.bodies.index(b2), params[0], params[1]))
        indices = (self.bodies.index(b1), self.bodies.index(b2))
        spring_joint = pymunk.DampedSpring(b1, b2, (0,0), (0,0), params[0], params[1], 0)
        self.constraints[(min(indices), max(indices))] = spring_joint
        self.space.add(spring_joint)
        return spring_joint

    def remove_constraint(self, pair):
        con = self.constraints[pair]
        self.space.remove(con)
        del self.constraints[pair]


    # def add_func_constraint(self, b1, b2, params=None):
    #     func_joint = FuncConstraint(b1, b2, params)
    #     self.func_constraints.append(func_joint)
    #     self.space.add(func_joint)
    #     return func_joint



    # returns representation of the constraints in the scene
    def get_constraint_rep(self):
        rep = []
        for con in self.space.constraints:
            if type(con) == pymunk.constraint.PinJoint:
                rep.append(['pin', self.bodies.index(con.a), self.bodies.index(con.b), None])
            # elif type(con) == FuncConstraint:
            #     rep.append(['func', self.bodies.index(con.a), self.bodies.index(con.b), (con.func_rep)])
            # elif type(con) == pymunk.constraint.SlideJoint:
            #     rep.append(['slide', self.bodies.index(con.a), self.bodies.index(con.b), (con.min, con.max)])
            elif type(con) == pymunk.constraint.DampedSpring:
                rep.append(['spring', self.bodies.index(con.a), self.bodies.index(con.b), (con.rest_length, con.stiffness, con.damping)])
        return rep

    # returns representation of the bodies (balls) in the scene
    def get_body_rep(self):
        rep = []
        for i, body in enumerate(self.bodies):
            rep.append((i, tuple(body.position), tuple(body.velocity)))
        return rep

    # given representation, add the corresponding balls
    def add_bodies_from_rep(self, data, col_type='ball'):
        for obj in data:
            self.add_ball(obj[1], obj[2], col_type)

    # given representation, add the corresponding constraints
    def add_constraints_from_rep(self,data,offset=0):
        for con in data:
            c1 = self.bodies[con[1] + offset]
            c2 = self.bodies[con[2] + offset]
            if con[0] == 'pin':
                self.add_pin_constraint(c1, c2, None)
            # elif con[0] == 'slide':
            #     self.add_slide_constraint(c1, c2, con[3])
            elif con[0] == 'spring':
                self.add_spring_constraint(c1, c2, con[3])
            # elif con[0] == 'func':
            #     self.add_func_constraint(c1, c2, con[3])

    # returns a whole representation of the balls in the space
    def save_state(self):
        return (self.get_body_rep(), self.get_constraint_rep())

    # load the body and constraint states from a rep
    def load_state(self, state, col_type='ball'):
        self.reset_space()
        self.add_bodies_from_rep(state[0], col_type)
        self.add_constraints_from_rep(state[1])

    # get rid of everything in the space, as if nothing had happened
    def reset_space(self):
        self.bodies = []
        self.constraints = {}
        self.space.remove(self.space.bodies, self.space.shapes, self.space.constraints)
        self.add_walls()

    # add collision noise to the space
    def update_collision_handler(self):
        h = self.space.add_default_collision_handler()
        def pre_solve(arbiter, space, data):
            dt = arbiter.contact_point_set
            if len(dt.points) > 0:
                dt.normal = dt.normal.rotated(gauss(self.noise['collision'] / 10))
                dt.points[0].distance = 0
                arbiter.contact_point_set = dt
            return True

        h.pre_solve = pre_solve

    # hit all balls around by a little bit
    def apply_dynamic_noise(self):
        for obj in self.bodies:
            obj.apply_impulse_at_local_point(
                (gauss(self.noise['dynamic']),
                gauss(self.noise['dynamic'])))


    # def apply_func_constraints(self):
    #     for fcon in self.func_constraints:
    #         fcon.apply_func()

    # run the scene forward and record locations of all moves
    def run_and_record(self, steps):
        self.update_collision_handler()
        locations = []
        for t in range(steps):
            locations.append(self.get_body_rep())
            self.space.step(1/50.0)
            # self.apply_func_constraints()
            self.apply_dynamic_noise()
        return locations

    # snapback rule - every ~skip~ steps, use correct locations and velocity and run for ~sim_length~ steps
    def get_snapback_locs(self, obj_data, skip=20, sim_length=10):
        self.update_collision_handler()
        old_locs = []
        new_locs = []
        cons = self.get_constraint_rep()
        for j in range(len(obj_data) // skip - 1):
            self.reset_space()
            self.add_bodies_from_rep(obj_data[j * skip])
            self.add_constraints_from_rep(cons)
            for i in range(sim_length):
                self.space.step(1/50.0)
                # self.apply_func_constraints()
                self.apply_dynamic_noise()
            old_locs.append(obj_data[j * skip + sim_length])
            new_locs.append(self.get_body_rep())

        return old_locs, new_locs


    # run the scene forward while showing what's going on in pygame
    def run_and_visualize(self, steps=10000, label='visualization', tick=20):
        pygame.init()
        screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(label)
        clock = pygame.time.Clock()
        draw_options = pymunk.pygame_util.DrawOptions(screen)

        self.update_collision_handler()
        # locations = []
        for t in range(steps):
            screen.fill((255,255,255))
            for event in pygame.event.get():
                if event.type in [QUIT, K_ESCAPE]:
                    sys.exit(0)
            self.space.debug_draw(draw_options)
            pygame.display.flip()
            clock.tick(TICK)
            # locations.append(self.get_body_rep())
            self.space.step(1/50.0)
            # self.apply_func_constraints()
            self.apply_dynamic_noise()
            if self.verbose and not t % 10:
                print('step {}'.format(t))
        # pdb.set_trace()
        # return locations

    # def visualize_obj_data(self, obj_data, label='visualization'):
    #     pygame.init()
    #     screen = pygame.display.set_mode(self.size)
    #     pygame.display.set_caption(label)
    #     clock = pygame.time.Clock()
    #     draw_options = pymunk.pygame_util.DrawOptions(screen)

    #     for t in range(len(obj_data)):
    #         screen.fill((255,255,255))
    #         for event in pygame.event.get():
    #             if event.type in [QUIT, K_ESCAPE]:
    #                 sys.exit(0)
    #         self.space.debug_draw(draw_options)
    #         pygame.display.flip()
    #         clock.tick(TICK)
    #         self.reset_space()
    #         self.add_bodies_from_rep(obj_data[t])

            # self.space.step(1/50.0)
            # self.apply_func_constraints()
            # self.apply_dynamic_noise()
            # if self.verbose and not t % 10:
            #     print('step {}'.format(t))

# from InputBox import InputBox

class BuilderScene(Scene):
    def __init__(self):
        super(BuilderScene, self).__init__()


    def run(self):
        pygame.init()
        screen = pygame.display.set_mode(self.size) 
        clock = pygame.time.Clock()
        draw_options = pymunk.pygame_util.DrawOptions(screen)

        # input_box1 = InputBox(40, self.size[1] - 40, 80, self.size[1])
        # input_boxes = [input_box1]

        running = True
        mode = 'pin'

        candidates = []

        while running:
            screen.fill((255,255,255))
            self.space.debug_draw(draw_options)
            # for box in input_boxes:
            #     box.draw(screen)

            mouse_pos = pymunk.pygame_util.from_pygame(pygame.mouse.get_pos(), screen)
            near_query = self.space.point_query_nearest(mouse_pos, 0, pymunk.ShapeFilter())
            if mouse_pos[0] < self.size[0] - 40 and mouse_pos[0] > 40 and mouse_pos[1] < self.size[1] - 40 and mouse_pos[1] > 40:
                press_within_boundaries = True
            else:
                press_within_boundaries = False

            if near_query is not None:
                shape = near_query.shape
                if shape.body.body_type != pymunk.Body.STATIC:
                    # TODO: make this work for not just circles
                    r = shape.radius + 4
                    p = pymunk.pygame_util.to_pygame(shape.body.position, screen)
                    pygame.draw.circle(screen, THECOLORS["orange"], p, int(r), 2)

            for event in pygame.event.get():
                # for box in input_boxes:
                #     box.handle_event(event)
                if event.type == QUIT or \
                    event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):  
                    running = False
                elif event.type == KEYDOWN and event.key in (K_s, K_p):
                    if event.key == K_s:
                        mode = 'spring'
                    elif event.key == K_p:
                        mode = 'pin'
                elif event.type == KEYDOWN and event.key is K_j:
                    print(self.save_state())
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and press_within_boundaries:
                    if near_query is not None:
                        shape = near_query.shape
                        shape.color = THECOLORS['cadetblue']
                        candidates.append(shape)
                        if len(candidates) == 2:
                            # TODO: check if a constraint already exists between the two bodies
                            self.add_pin_constraint(candidates[0].body, candidates[1].body)
                            for c in candidates:
                                c.color = THECOLORS['lightblue']
                            candidates = []
                    else:
                        self.add_ball(mouse_pos)

                    
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 3 and press_within_boundaries:
                    if near_query is not None:
                        shape = near_query.shape
                        self.space.remove(shape.body, shape)
                        # TODO: remove all constraints involving this object

            # for box in input_boxes:
            #     box.update()

            
            
            pygame.display.flip()
            clock.tick(TICK)
            self.space.step(1/50.0)




'''
class FuncConstraint(pymunk.SlideJoint):
    def __init__(self, b1, b2, params):
        super(FuncConstraint, self).__init__(b1, b2, (0,0), (0,0), 0, float('inf'))

        def default_func(bodies, params):
            towards_vec = bodies[1].position - bodies[0].position
            x = params[0] * (dist(bodies[0].position, bodies[1].position) - params[1])
            x2 = params[0] * (dist(bodies[0].position, bodies[1].position) - params[1])
            force_vec = vec_with_length(towards_vec, x)
            return (force_vec, vec_opposite(force_vec))

        if params == None:
            self.func = default_func
            self.params = [1.5, 100]
        else:
            self.func = params[0]
            self.params = params[1:]

        self.func_rep = self.get_func_rep()



    def get_func_rep(self):
        return ['*/', ['+', ['**', ['-', ['x0'], ['x4']]], ['**', ['-', ['x1'], ['x5']]]]]
        return ['']


    # def set_func_from_rep(rep):
    #     def func(params):
    #         return params[0] ** 2
    #     self.func = func
    #     self.func_rep = self.get_func_rep()


    def apply_func(self):
        bodies = [self.a, self.b]
        forces = self.func(bodies, self.params)
        print(forces)
        self.a.apply_impulse_at_local_point(forces[0], (0,0))
        self.b.apply_impulse_at_local_point(forces[1], (0,0))


def evaluate(node, case):
        """
        Evaluate a node recursively. The node's symbol string is evaluated.

        :param node: Evaluated node
        :type node: list
        :param case: Current fitness case
        :type case: list
        :returns: Value of the evaluation
        :rtype: float
        """
        symbol = node[0]
        symbol = symbol.strip()

        # Identify the node symbol
        if symbol == "+":
            # Add the values of the node's children
            return evaluate(node[1], case) + evaluate(node[2], case)

        elif symbol == "-":
            # Subtract the values of the node's children
            return evaluate(node[1], case) - evaluate(node[2], case)

        elif symbol == "*":
            # Multiply the values of the node's children
            return evaluate(node[1], case) * evaluate(node[2], case)

        elif symbol == "/":
            # Divide the value's of the nodes children. Too low values of the
            # denominator returns the numerator
            numerator = evaluate(node[1], case)
            denominator = evaluate(node[2], case)
            if abs(denominator) < 0.00001:
                denominator = 1

            return numerator / denominator

        elif symbol == "**2":
            return evaluate(node[1], case) ** 2
        elif symbol == "*/2":
            return evaluate(node[1], case) ** 0.5

        elif symbol.startswith("x"):
            # Get the variable value
            return case[int(symbol[1:])]
        else:
            # The symbol is a constant
            return float(symbol)
'''



# Euclidean distance function
def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** (1/2)

def norm(a):
    return (a[0] ** 2 + a[1] ** 2) ** (1/2)

def vec_with_length(a, new_len):
    return (a[0] / norm(a) * new_len, a[1] / norm(a) * new_len)

def vec_opposite(a):
    return (-a[0], -a[1])

def add_vec(a, b):
    return (a[0] + b[0], a[1] + b[1])
def sub_vec(a, b):
    return (a[0] - b[0], a[1] - b[1])

# angle between two vectors
def ang(a, b):
    return np.arccos(min((a[0] * b[0] + a[1] * b[1]) / (norm(a) * norm(b)),1))

# normal variable
def gauss(scale=1):
    return np.random.normal(scale=scale)

# DEPRECATED
# cost function given two location representations
# \Sigma_t (\gamma^t * distance)
# gamma is decay factor, t is timestep. sum over all objects for overall cost

# def get_cost(d1, d2, interval=1, obj_id=None):
#     assert len(d1) <= len(d2)
#     num_steps = len(d1) // interval
#     if interval * num_steps > len(d1):
#         raise ValueError('Need at least {} steps'.format(interval * (num_steps - 1) + 1))
#     num_objs = len(d1[0])
#     running_cost = 0
#     # gamma is the decay factor, which decreases in importance with every timestep
#     gamma = 1
#     for i in range(num_steps):
#         step = i * interval
#         if obj_id == None:
#             for j in range(num_objs):
#                 d = dist(d1[step][j][1], d2[step][j][1])
#                 running_cost += d * gamma
#         else:
#             d = dist(d1[step][obj_id][1], d2[step][obj_id][1])
#             running_cost += d * gamma
#         gamma *= 0.95
#     final_cost = running_cost / num_objs
#     return final_cost


# get cost based on given locations and velocities, using gaussian distributions
gauss_dist = scipy.stats.norm(0, 30)
gauss_vel = scipy.stats.norm(0, 100)
gauss_vel_norm = scipy.stats.norm(1, 0.3)
gauss_vel_ang = scipy.stats.norm(0, 0.4)
def get_cost(l1, l2, obj_id=None):
    # print(obj_id)
    running_cost = 0
    num_steps = len(l1)
    num_objs = len(l1[0])
    # default gaussian distributions of position and velocity differentials
    for i in range(num_steps):
        if obj_id == None:
            for j in range(num_objs):
                obj1 = l1[i][j]
                obj2 = l2[i][j]

                # ddif = dist(obj1[1], obj2[1])
                # # vdif = dist(obj1[2], obj2[2])
                # vdif_norm = norm(obj1[2]) / norm(obj2[2])
                # vdif_ang = ang(obj1[2], obj2[2])

                # dprob = max(gauss_dist.pdf(ddif) / gauss_dist.pdf(0), 1e-200)
                # # vprob = gauss_vel.pdf(vdif)
                # vprob_norm = max(gauss_vel_norm.pdf(vdif_norm) / gauss_vel_norm.pdf(1), 1e-200)
                # vprob_ang = max(gauss_vel_ang.pdf(vdif_ang) / gauss_vel_ang.pdf(0), 1e-200)

                # # tdif = np.log(dprob) + np.log(vprob)
                # # tdif = dprob * vprob
                # tdif = np.log(vprob_norm) + np.log(vprob_ang) + np.log(dprob)
                # # print('old', dprob, vprob)
                # #print('new', dprob, vprob_norm, vprob_ang)

                tdif = get_tdif(obj1, obj2)

                running_cost += tdif
        # elif type(obj_id) == tuple:
        #     obj1_1 = l1[i][obj_id[0]]
        #     obj2_1 = l1[i][obj_id[1]]
        #     obj1_2 = l2[i][obj_id[0]]
        #     obj2_2 = l2[i][obj_id[1]]

        #     ddif_1 = sub_vec(obj1_1[1], obj2_1[1])
        #     ddif_2 = sub_vec(obj1_2[1], obj2_2[1])
        #     ddif = dist(ddif_1, ddif_2)


        #     # ddif = dist(obj1_1[1], obj2[1])
        #     # vdif = dist(obj1[2], obj2[2])
        #     vdif_1 = sub_vec(obj1_1[2], obj2_1[2])
        #     vdif_2 = sub_vec(obj1_2[2], obj2_2[2])

        #     vdif_norm = norm(vdif_1) / norm(vdif_2)
        #     vdif_ang = ang(vdif_1, vdif_2)


        #     dprob = max(gauss_dist.pdf(ddif) / gauss_dist.pdf(0), 1e-200)
        #     vprob_norm = max(gauss_vel_norm.pdf(vdif_norm) / gauss_vel_norm.pdf(1), 1e-200)
        #     vprob_ang = max(gauss_vel_ang.pdf(vdif_ang) / gauss_vel_ang.pdf(0), 1e-200)
            

        #     # tdif = dprob * vprob * 1e5
        #     # print(vprob_norm, vprob_ang, dprob)
        #     tdif = np.log(vprob_norm) + np.log(vprob_ang) + np.log(dprob)
        #     running_cost += tdif
        else:
            obj1 = l1[i][obj_id]
            obj2 = l2[i][obj_id]

            # ddif = dist(obj1[1], obj2[1])
            # # vdif = dist(obj1[2], obj2[2])
            # vdif_norm = norm(obj1[2]) / norm(obj2[2])
            # vdif_ang = ang(obj1[2], obj2[2])


            # dprob = max(gauss_dist.pdf(ddif) / gauss_dist.pdf(0), 1e-200)
            # vprob_norm = max(gauss_vel_norm.pdf(vdif_norm) / gauss_vel_norm.pdf(1), 1e-200)
            # vprob_ang = max(gauss_vel_ang.pdf(vdif_ang) / gauss_vel_ang.pdf(0), 1e-200)
            

            # # tdif = dprob * vprob * 1e5
            # # print(vprob_norm, vprob_ang, dprob)
            # tdif = np.log(vprob_norm) + np.log(vprob_ang) + np.log(dprob)
            # # print(tdif)
            # #tdif = vprob_norm * vprob_ang * dprob * 1e5

            tdif = get_tdif(obj1, obj2)
            
            running_cost += tdif
    if obj_id == None:
        return -running_cost/num_objs/num_steps
    else:
        #pdb.set_trace()
        return -running_cost/num_steps



def get_tdif(obj1, obj2):
    ddif = dist(obj1[1], obj2[1])
    # vdif = dist(obj1[2], obj2[2])
    vdif_norm = norm(obj1[2]) / norm(obj2[2])
    vdif_ang = ang(obj1[2], obj2[2])

    dprob = max(gauss_dist.pdf(ddif) / gauss_dist.pdf(0), 1e-200)
    # vprob = gauss_vel.pdf(vdif)
    vprob_norm = max(gauss_vel_norm.pdf(vdif_norm) / gauss_vel_norm.pdf(1), 1e-200)
    vprob_ang = max(gauss_vel_ang.pdf(vdif_ang) / gauss_vel_ang.pdf(0), 1e-200)

    # tdif = np.log(dprob) + np.log(vprob)
    # tdif = dprob * vprob
    tdif = np.log(vprob_norm) + np.log(vprob_ang) + np.log(dprob)

    return tdif



def print_time(label, start):
    print('{}: {}'.format(label, time.time() - start))
