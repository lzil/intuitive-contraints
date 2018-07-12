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
# GRAVITY = 0
GRAVITY = (0,-10000.0)
ELASTICITY_WALL = 0.5
ELASTICITY_OBJ = 0.95
WALL_WIDTH = 30
FRICTION_WALL = 0.8
FRICTION_OBJ = 0.6

SPRING_PRIORS = [
    scipy.stats.norm(50, 50),  # rest length
    scipy.stats.norm(90, 50)    # stiffness
]
SPRING_LIMITS = ((0, 100), (40, 140))
SPRING_MULTS = (5, 5)
SPRING_START = (50, 90)



SPRINGS = {
    'short': (100, 500, 0.05),
    'medium': (175, 300, 0.1),
    'long': (250, 200, 0.3)
}

# SP_LENS = {
#     'short': 15,
#     'medium': 30,
#     'long': 60
# }

# SP_KS = {
#     'low': 100,
#     'medium': 200,
#     'high': 300
# }

# SP_DAMPS = {
#     'low': 0.2,
#     'medium': 0.5,
#     'high': 0.8
# }

COL_TYPES = {
    'wall': 1,
    'normal': 2,
    'differential': 3,
}


# manages the space, bodies, walls, and running of physics all in one class
class Scene:
    def __init__(self, space=None, noise=(0,0), walls=True, size=(1400, 1000)):
        self.size = size
        if noise[0] == 0 and noise[1] == 0:
            self.noise = None
        else:
            self.noise = {
                'collision': noise[0],
                'dynamic': noise[1]
            }
        
        if space:
            self.load_space(space)
        else:
            self.space = pymunk.Space()
            if walls:
                self.add_walls()

        # self.space.collision_slop = 0.01

        self.verbose = False
        self.update_collision_handler()

        # self.constraints = {}
        # self.func_constraints = []

    def load_space(self, space):
        self.space = space.copy()

    def get_space(self):
        return self.space.copy()

    # add walls on the four borders of the space
    def add_walls(self):
        bottom_wall = pymunk.Segment(self.space.static_body, (0,0), (self.size[0],0), WALL_WIDTH)
        left_wall = pymunk.Segment(self.space.static_body, (0,0), (0,self.size[1]), WALL_WIDTH)
        top_wall = pymunk.Segment(self.space.static_body, (0,self.size[1]), (self.size[0],self.size[1]), WALL_WIDTH)
        right_wall = pymunk.Segment(self.space.static_body, (self.size[0],self.size[1]), (self.size[0],0), WALL_WIDTH)
        walls = [bottom_wall, left_wall, top_wall, right_wall]
        for w in walls:
            w.friction = FRICTION_WALL
            w.elasticity = ELASTICITY_WALL
            w.color = THECOLORS['grey']
        self.space.add(walls)



    # def add_func_constraint(self, b1, b2, params=None):
    #     func_joint = FuncConstraint(b1, b2, params)
    #     self.func_constraints.append(func_joint)
    #     self.space.add(func_joint)
    #     return func_joint


    

    # add collision noise to the space
    def update_collision_handler(self):
        if self.noise is None:
            return
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
        if self.noise is None:
            return
        for obj in self.space.bodies:
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
    def run_and_visualize(self, steps=10000, tick=20):
        # start the drawing
        pygame.init()
        screen = pygame.display.set_mode(self.size)
        clock = pygame.time.Clock()
        draw_options = pymunk.pygame_util.DrawOptions(screen)

        self.update_collision_handler()
        for t in range(steps):
            for event in pygame.event.get():
                if event.type in [QUIT, K_ESCAPE]:
                    sys.exit(0)
                if event.type == KEYDOWN and event.key is K_r: # return to the earlier screen
                    return
            # draw everything
            screen.fill((255,255,255))
            self.space.debug_draw(draw_options)
            self.draw_body_borders(screen)
            pygame.display.flip()
            clock.tick(TICK)
            for i in range(4):
                self.space.step(1/400.0)

            # self.apply_dynamic_noise()
            # if self.verbose and not t % 10:
            #     print('step {}'.format(t))

    # make sure borders are drawn on shapes so they're pretty. DO NOT USE if not in visualization mode
    def draw_body_borders(self, screen):
        for b in self.space.bodies:
            s = list(b.shapes)[0]
            if type(s) is not pymunk.Circle:
                verts = []
                for v in s.get_vertices():
                    verts.append(pymunk.pygame_util.from_pygame(v.rotated(b.angle) + b.position, screen))
                verts.append(verts[0])
                pygame.draw.lines(screen, THECOLORS["gray"], False, verts, 4)
            else:
                pygame.draw.circle(screen, THECOLORS['gray'], pymunk.pygame_util.from_pygame(b.position, screen), int(s.radius) + 1, 4)


    # add constraint based on which mode is active in the builder
    def add_constraint_by_mode(self, b1, b2, mode, loc=(0,0)):
        # pin constraint
        if mode == 0:
            joint = self.add_pin_constraint(b1, b2, loc1=loc)
        # the varioua spring constraints
        elif mode == 1:
            joint = self.add_spring_constraint(b1, b2, loc1=loc, params=SPRINGS['short'])
        elif mode == 2:
            joint = self.add_spring_constraint(b1, b2, loc1=loc, params=SPRINGS['medium'])
        elif mode == 3:
            joint = self.add_spring_constraint(b1, b2, loc1=loc, params=SPRINGS['long'])
        return joint



    def add_background_constraint(self, b, mode, loc, params=None):
        joint = self.add_constraint_by_mode(self.space.static_body, b, mode, loc)
        if type(self) == SimulationScene:
            ind = self.bodies.index(b)
            if ind in self.constraints:
                self.constraints[ind].append(joint)
            else:
                self.constraints[ind] = [joint]
        return joint


    def add_pin_constraint(self, b1, b2, loc1=(0,0), loc2=(0,0), params=None):
        # if self.verbose:
        #     print('Adding pin to {}-{}'.format(self.bodies.index(b1), self.bodies.index(b2)))
        pin_joint = pymunk.PinJoint(b1, b2, loc1, loc2)
        self.space.add(pin_joint)
        if type(self) == SimulationScene:
            indices = (self.bodies.index(b1), self.bodies.index(b2))
            self.constraints[(min(indices), max(indices))] = pin_joint
        
        return pin_joint

    def add_spring_constraint(self, b1, b2, loc1=(0,0), loc2=(0,0), params=None):
        if params == None:
            rest_length = np.random.randint(SPRING_LIMITS[0][0], SPRING_LIMITS[0][1])
            stiffness = np.random.randint(SPRING_LIMITS[1][0], SPRING_LIMITS[1][1])
            # damping = np.random.gamma(1, 0.5)
            damping = 0
            params = [rest_length, stiffness, damping]
        # if self.verbose:
        #     print('Adding spring to {}-{}; {}, {}'.format(self.bodies.index(b1), self.bodies.index(b2), params[0], params[1]))

        spring_joint = pymunk.DampedSpring(b1, b2, loc1, loc2, params[0], params[1], params[2])
        self.space.add(spring_joint)

        if type(self) == SimulationScene:
            indices = (self.bodies.index(b1), self.bodies.index(b2))
            self.constraints[(min(indices), max(indices))] = spring_joint

        return spring_joint

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



class SimulationScene(Scene):
    def __init__(self, *args, **kwargs):
        super(SimulationScene, self).__init__(*args, **kwargs)
        self.bodies = [None] * len(self.space.bodies)
        for b in self.space.bodies:
            self.bodies[b.udata] = b
            b.activate()
        self.constraints = self.get_constraints()

    def get_constraints(self):
        constraints = {}
        for b in self.bodies:
            for c in b.constraints:
                if c.a in self.bodies and c.b in self.bodies:
                    cb = (self.bodies.index(c.a), self.bodies.index(c.b))
                    constraints[(min(cb), max(cb))] = c
                else:
                    if self.bodies.index(b) in constraints:
                        constraints[self.bodies.index(b)].append(c)
                    else:
                        constraints[self.bodies.index(b)] = [c]
                    
        return constraints

    # returns representation of the bodies in the scene
    def get_body_rep(self):
        rep = []
        for i, body in enumerate(self.bodies):
            rep.append((i, tuple(body.position), tuple(body.velocity)))
        return rep

    # returns representation of the bodies in the scene
    def get_constraint_rep(self):
        rep = []
        for pair, con in self.constraints.items():
            if type(con) == pymunk.constraint.PinJoint:
                rep.append(['pin', pair[0], pair[1], None])
            elif type(con) == pymunk.constraint.DampedSpring:
                rep.append(['spring', pair[0], pair[1], (con.rest_length, con.stiffness, con.damping)])
        return rep

    def update_body_locations(self, dt):
        for i, body in enumerate(self.bodies):
            body.position = dt[i][1]
            self.space.reindex_shapes_for_body(body)


    # def get_rep(self):
    #     dynamic_bodies = list(filter(lambda b: b.body_type is pymunk.Body.DYNAMIC, self.space.bodies))
    #     for con in self.space.constraints:
    #         if type(con) == pymunk.constraint.PinJoint:
    #             rep.append(['pin', bodies.index(con.a), bodies.index(con.b), None])
    #         # elif type(con) == FuncConstraint:
    #         #     rep.append(['func', self.bodies.index(con.a), self.bodies.index(con.b), (con.func_rep)])
    #         # elif type(con) == pymunk.constraint.SlideJoint:
    #         #     rep.append(['slide', self.bodies.index(con.a), self.bodies.index(con.b), (con.min, con.max)])
    #         elif type(con) == pymunk.constraint.DampedSpring:
    #             rep.append(['spring', bodies.index(con.a), bodies.index(con.b), (con.rest_length, con.stiffness, con.damping)])
    #     return dynamic_bodies, rep

    # def add_slide_constraint(self, b1, b2, params=None):
    #     if params == None:
    #         cur_dist = dist(b1.position, b2.position)
    #         params = [random.random() * cur_dist, (1 + random.random()) * cur_dist]

    #     if self.verbose:
    #         print('Adding slide to {}-{}; {}, {}'.format(self.bodies.index(b1), self.bodies.index(b2), params[0], params[1]))
    #     slide_joint = pymunk.SlideJoint(b1, b2, (0,0), (0,0), params[0], params[1])
    #     self.space.add(slide_joint)
    #     return slide_joint




    def remove_constraint(self, pair):
        con = self.constraints[pair]
        self.space.remove(con)
        del self.constraints[pair]



class BuilderScene(Scene):
    def __init__(self, *args, **kwargs):
        super(BuilderScene, self).__init__(*args, **kwargs)

    # add a small ball shape and body to the space
    def add_ball(self, pos, vel=(0,0), r=20, col_type='normal'):
        mass = math.pi * (r ** 2) / 1600
        moment = pymunk.moment_for_circle(mass, 0, r)
        body = pymunk.Body(mass, moment)
        body.position = pos
        body.velocity = vel
        shape = pymunk.Circle(body, r)
        shape.collision_type = COL_TYPES[col_type]
        shape.color = THECOLORS['lightblue']
        if col_type == 'ball_red':
            shape.color=(255,100,100,0)
        shape.elasticity = ELASTICITY_OBJ
        shape.friction = FRICTION_OBJ
        self.space.add(body, shape)
        return body

    # add a block shape and body to the space
    def add_block(self, pos1, pos2=None, vel=(0,0), col_type='normal', r=20):
        if pos2 is None:
            mass = 1
            rs = [(-r, -r), (-r, r), (r, r), (r, -r)]
            moment = pymunk.moment_for_box(mass, (r*2, r*2))
            body = pymunk.Body(mass, moment)
            body.position = pos1
        else:
            w, h = sub_vec(pos2, pos1)
            w, h = max(10, abs(w)), max(10, abs(h))
            mass = w * h / 1600
            rs = [(-w/2, -h/2), (-w/2, h/2), (w/2, h/2), (w/2, -h/2)]
            moment = pymunk.moment_for_box(mass, (w,h))
            body = pymunk.Body(mass, moment)
            body.position = avg_vec(pos1, pos2)
        body.velocity = vel
        shape = pymunk.Poly(body, rs)
        shape.collision_type = COL_TYPES[col_type]
        shape.color = THECOLORS['lightblue']
        shape.elasticity = ELASTICITY_OBJ
        shape.friction = FRICTION_OBJ
        self.space.add(body, shape)
        return body

    # get the constraint given two bodies; if no constraint exists return None
    def find_constraint(self, a, b):
        for c in a.constraints:
            if c.a is b or c.b is b:
                return c
        return None

    # remove a constraint if one exists between two bodies
    def remove_constraint(self, a, b):
        con = self.find_constraint(a, b)
        if con in self.space.constraints:
            self.space.remove(con)


    # get rid of everything in the space, as if nothing had happened
    def reset_space(self):
        self.space.remove(self.space.bodies, self.space.shapes, self.space.constraints)
        self.add_walls()


    # set the udata of a body so that it can be identified later
    def set_body_data(self):
        # bodies = list(filter(lambda b: b.body_type is pymunk.Body.DYNAMIC, self.space.bodies))
        for i, b in enumerate(self.space.bodies):
            b.__setattr__('udata', i)



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

def avg_vec(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

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



def load_file(file):
    # get file id which is determined by type of constraint and timestamp
    file_id = file.split('/')[-1].split('.')[0]

    # actually load the file
    with open(file, 'rb') as f:
        data = pickle.load(f)

    space_data = data['space']
    locs_data = data['locs']

    return file_id, space_data, locs_data


