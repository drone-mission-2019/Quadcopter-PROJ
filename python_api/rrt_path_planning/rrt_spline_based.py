from .rrt_base import RRTBase
from .bezier_util import get_d, generate_bezier, get_gamma
from .geometry import dist_between_points, steer
from .search_space import SearchSpace
from .heuristics import cost_to_go, segment_cost, path_cost
import random
import math
import uuid
import numpy as np


class RRTSpline(RRTBase):
    def __init__(self, X, x_init, x_goal, max_samples=1024, r=0.1, prc=1,
                 k_max=0.1, p_goal=0.1, gamma=0.3*math.pi, rewire_count=None):
        super().__init__(X, None, x_init, x_goal, max_samples, r, prc)
        self.k_max = k_max
        self.p_goal = p_goal
        self.gamma = gamma

        self.d = get_d(self.gamma, self.k_max)
        self.L_c = 6*self.d
        self.L_d = 1.5*self.d
        self.phi_d = math.pi

        for center in X.obs_centers:
            self.trees[0].V_obs.insert(0, center+center, center)

        self.min_energy = math.inf
        self.T = 0.1
        self.a = 1.5
        self.c = 0.01
        self.ita = 20000
        self.d_0 = 40
        self.fail = 0
        self.fail_max = 10
        self.K = (self.energy(x_goal) + self.energy(x_init)) * 0.5
        # self.K = 100
        # print(self.K)
        # print(self.attraction(x_init), self.resistance(x_init))
        # self.X.obs.insert(uuid.uuid4(), (20, 20, 60, 60), (20, 20, 60, 60))

        self.rewire_count = rewire_count
        self.c_best = math.inf

    def clear(self):
        self.__init__(self.X, self.x_init, self.x_goal)

    def set_init(self, x_init):
        self.x_init = x_init

    def set_goal(self, x_goal):
        self.x_goal = x_goal

    def add_obstacle(self, obs):
        self.X.obs.insert(uuid.uuid4(), obs, obs)
        center = self.X.obs_center(obs)
        self.trees[0].V_obs.insert(uuid.uuid4(), center + center, center)

    def current_rewire_count(self, tree):
        if self.rewire_count is None:
            return self.trees[tree].V_count

        return self.trees[tree].V_count

    def get_nearby_vertices(self, tree, x_new):
        X_near = list(self.nearby(tree, x_new, self.current_rewire_count(tree)))
        L_near = [(path_cost(self.trees[tree].E, self.x_init, x_near), x_near) for x_near in X_near]
        L_near.sort(key=lambda x: x[0])
        return L_near

    def choose_parent(self, tree, x_new, L_near):
        # L_near = self.get_nearby_vertices(tree, x_new)
        for c_near, x_near in L_near:
            cost = c_near + cost_to_go(x_near, self.x_goal)
            if cost < self.c_best \
                    and (self.can_connect_to_point(tree, x_new, x_near))[0] \
                    and segment_cost(x_near, x_new) >= self.d:
                self.connect_points(tree, x_near, x_new)
            self.c_best = cost

    def rewire(self, tree, x_new, L_near):
        for c_near, x_near in L_near:
            curr_cost = path_cost(self.trees[tree].E, self.x_init, x_near)
            tent_cost = path_cost(self.trees[tree].E, self.x_init, x_new) + segment_cost(x_new, x_near)
            if tent_cost < curr_cost \
                    and segment_cost(x_new, x_near) >= self.d \
                    and (self.can_connect_to_point(tree, x_near, x_new))[0]:
                self.trees[tree].E[x_near] = x_new

    def attraction(self, p_c):
        return self.c*dist_between_points(p_c, self.x_goal)**2

    def resistance(self, p_c):
        near_list = list(self.trees[0].V_obs.nearest(p_c, num_results=1, objects='raw'))
        if len(near_list) != 0:
            nearest_obs = near_list[0]
        else:
            return 0
        # print(nearest_obs)
        dist = dist_between_points(nearest_obs, p_c)
        if dist < self.d_0:
            return self.ita*(1/dist-1/self.d_0)**2
        return 0

    def energy(self, p_c):
        return self.attraction(p_c)+self.resistance(p_c)

    def improved_transition(self, x_rand, e_min):
        e_rand = self.energy(x_rand)
        dist = dist_between_points(x_rand, self.get_nearest(0, x_rand))
        if e_rand < e_min:
            return True
        prob = math.exp((e_min-e_rand)/(dist*self.K*self.T))
        # print("e_rand:", e_rand, "dist:", dist, "prob:", prob, "K:", self.K, "T:", self.T, "fail:", self.fail)
        # print(prob)
        if random.random() < prob:
            self.T /= self.a
            self.fail = 0
            return True
        if self.fail > self.fail_max:
            self.T *= self.a
            self.fail = 0
        else:
            self.fail += 1
        return False

    def state_sampling(self, p_c):
        p_c = np.array(p_c)
        L_h = dist_between_points(p_c, self.x_goal)

        if L_h < self.L_c:
            x_rand = self.sampling_for_concentration(p_c)
        else:
            x_rand = self.sampling_for_exploration()
        return x_rand

    def sampling_for_concentration(self, p_c):
        p = random.random()
        if p < 0.8:
            p_c = np.array(p_c)
            epsilon_h = random.uniform(-1, 1)
            epsilon_phi = random.uniform(-1, 1)

            L_h = dist_between_points(p_c[:2], self.x_goal[:2])
            if self.x_goal[0] - p_c[0] < 0.1:
                phi_0 = 0.5 * math.pi
            else:
                phi_0 = np.arctan((self.x_goal[1] - p_c[1]) / (self.x_goal[0] - p_c[0]))
            new_phi = phi_0 + self.phi_d * epsilon_phi
            arr = np.array([np.cos(new_phi), np.sin(new_phi)])
            arr *= (self.L_d + L_h) * abs(epsilon_h)

            if self.X.dimensions == 3:
                L_z = abs(self.x_goal[2]-p_c[2])
                arr += (L_z*epsilon_h, )

            x_rand = self.bound_point(self.x_goal+arr)
        else:
            x_rand = self.sampling_for_exploration()

        return x_rand

    def sampling_for_exploration(self):
        prob = random.random()
        if prob < self.p_goal:
            x_rand = self.X.sample_free()
            while not self.improved_transition(x_rand, self.min_energy):
                x_rand = self.X.sample_free()
        else:
            x_rand = self.x_goal
        return x_rand

    def node_selection(self, tree, x_rand):
        near_nodes = list(self.nearby(tree, x_rand, self.trees[tree].V_count))
        for node in near_nodes:
            parent_node = self.trees[tree].E[node]
            if parent_node is None:
                return node

            gamma = get_gamma(parent_node, node, x_rand)
            if gamma is None or gamma > self.gamma:
                continue

            return node
        return None

    # to be fixed
    def node_expansion(self, tree, x_rand, x_near):
        parent_node = self.trees[tree].E[x_near]
        if parent_node is None:
            x_new = self.bound_point(steer(x_near, x_rand, self.d))
        else:
            x_new = self.bound_point(steer(x_near, x_rand, 2*self.d))

        # collision_free = self.X.collision_free(x_near, x_new, self.r)
        # if parent_node is not None:
        #     bezier_coll_free, _, _, _ = self.bezier_collision_free(parent_node, x_near, x_rand)
        #     collision_free = (collision_free and bezier_coll_free)
        #
        # if collision_free:
        #     self.min_energy = min(self.min_energy, self.energy(x_new))
        #     return x_new
        # return None
        return x_new

    def connect_points(self, tree, x_a, x_b):
        if self.trees[tree].V.count(x_b) == 0:
            self.add_vertex(tree, x_b)
            self.add_edge(tree, x_b, x_a)
            self.min_energy = min(self.min_energy, self.energy(x_b))

    def can_connect_to_point(self, tree, x_new, x_near=None):
        # if self.trees[tree].V.count(x_new) != 0:
        #     return False, None
        if x_near is None:
            near_nodes = list(self.nearby(tree, x_new, self.trees[tree].V_count))
        else:
            near_nodes = [x_near]

        for node in near_nodes:
            parent_node = self.trees[tree].E[node]
            if parent_node is None:
                if self.X.collision_free(node, x_new, self.r):
                    return True, node
                return False, None

            col_free, gamma, s0, s1 = self.bezier_collision_free(parent_node, node, x_new)
            if not col_free:
                continue
            if gamma > self.gamma:
                continue
            if gamma == 0:
                if not self.X.collision_free(parent_node, x_new, self.r):
                    continue
                else:
                    return True, node
            if not self.X.collision_free(parent_node, s0, self.r):
                continue
            if not self.X.collision_free(s1, x_new, self.r):
                continue

            return True, node
        return False, None

    def rrt_search(self, d_lim=0.1, star=False):
        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)
        x_new = self.x_init
        i = 0

        while dist_between_points(x_new, self.x_goal) > d_lim:
            x_rand = self.state_sampling(x_new)
            x_near = self.node_selection(0, x_rand)
            x_old = x_new
            x_new = self.node_expansion(0, x_rand, x_near)

            while (x_near is None) or (not (self.can_connect_to_point(0, x_new, x_near))[0]):
                # print(self.can_connect_to_point(0, x_new, x_near))
                x_rand = self.state_sampling(x_old)
                x_near = self.node_selection(0, x_rand)
                x_new = self.node_expansion(0, x_rand, x_near)

            self.connect_points(0, x_near, x_new)
            # print(i, "rand:", x_rand, ", near:", x_near, ", new:", x_new)
            i += 1
            # print(i, "rand:", x_rand, ", near:", x_near, ", new:", x_new)
            if star:
                L_near = self.get_nearby_vertices(0, x_new)
                self.choose_parent(0, x_new, L_near)
                self.rewire(0, x_new, L_near)

            if self.prc > random.random() or dist_between_points(x_new, self.x_goal) < self.L_c:
                solution = self.can_connect_to_point(0, self.x_goal)
                # print(solution[1])
                if solution[0]:
                    self.connect_points(0, solution[1], self.x_goal)
                    x_new = self.x_goal
                    break

        old_path = self.reconstruct_path(0, self.x_init, x_new)
        print("original:", len(old_path))
        path, deleted_nodes = self.node_pruning(old_path)
        print("optimal:", len(path))

        self.trees[0].E[path[-1]] = path[-2]   # 保证两棵树之间的连续性的重构

        inter_path = [path[0]]
        for i in range(len(path)-2):
            _, _, B, E = generate_bezier(path[i], path[i+1], path[i+2])
            if isinstance(B, int):
                inter_path.append(path[i+1])
                continue

            points = [(B[0][j], B[1][j]) + ((B[2][j], ) if len(B) > 2 else tuple())
                      for j in range(len(B[0])-1)]
            E_points = [(E[0][j], E[1][j]) + ((E[2][j], ) if len(E) > 2 else tuple())
                        for j in reversed(list(range(len(E[0]))))]
            points.extend(E_points)
            inter_path.extend(points)

        inter_path.append(path[-1])
        return inter_path, old_path, deleted_nodes

    def node_pruning(self, path):
        deleted_node = []
        if len(path) < 3:
            return path
        if self.trees[0].E[path[0]] is None:
            temp = ['#']
        else:
            temp = [self.trees[0].E[path[0]]]
        temp.extend(path)
        temp.append('#')
        i = 2
        while i < len(temp)-2:
            # print(i)
            if self.five_node_prune(temp[i-2:i+3]):
                deleted_node.append(temp[i])
                del(temp[i])
            else:
                i += 1

        return temp[1:-1], deleted_node

    def five_node_prune(self, point):
        if point[0] == '#':
            if point[4] == '#':
                if self.X.collision_free(point[1], point[3], self.r):
                    return True
                return False
            col_free, gamma, s0, s1 = self.bezier_collision_free(point[1], point[3], point[4])
            if (not col_free) or (gamma > self.gamma):
                return False
            if (gamma == 0 and self.X.collision_free(point[1], point[3], self.r)) \
                    or self.X.collision_free(s0, point[1], self.r):
                return True
            return False

        if point[4] == '#':
            if point[0] == '#':
                if self.X.collision_free(point[1], point[3], self.r):
                    return True
                return False
            col_free, gamma, s0, s1 = self.bezier_collision_free(point[0], point[1], point[3])
            # print(point, col_free, gamma)
            if (not col_free) or (gamma > self.gamma):
                return False
            if (gamma == 0 and self.X.collision_free(point[1], point[3], self.r)) \
                    or self.X.collision_free(s1, point[3], self.r):
                return True
            return False

        # print(point)
        col_free, gamma, s0, s1 = self.bezier_collision_free(point[0], point[1], point[3])
        # print(col_free, gamma)
        if (not col_free) or (gamma > self.gamma):
            return False
        col_free, gamma, s2, s3 = self.bezier_collision_free(point[1], point[3], point[4])
        # print(col_free, gamma)

        if (not col_free) or (gamma > self.gamma):
            return False

        if self.X.collision_free(point[1], point[3], self.r):
            return True
        return False

    def bezier_collision_free(self, w1, w2, w3):
        gamma, d, B, E = generate_bezier(w1, w2, w3)
        if d is None:
            return False, 0, 0, 0
        if d == 0:
            return True, 0, 0, 0

        points = [(B[0][j], B[1][j]) + ((B[2][j],) if len(B) > 2 else tuple())
                  for j in range(len(B[0]))]
        E_points = [(E[0][j], E[1][j]) + ((E[2][j],) if len(E) > 2 else tuple())
                    for j in range(len(E[0]))]
        points.extend(E_points)
        coll_free = all(map(self.X.obstacle_free, points))

        return coll_free, gamma, points[0], E_points[0]


if __name__ == '__main__':
    path = [(0, 0), (0.058106828700898686, 3.8347946139141857),
            (4.019498227534062, 10.403164481928496), (4.983222586402014, 25.625036433493932),
            (8.178035055253435, 32.59850600227095), (14.45737816912282, 37.00372502021874),
            (20.736721282992203, 41.408944038166524), (27.016064396861587, 45.81416305611431),
            (33.29540751073097, 50.219382074062096), (39.57475062460035, 54.62460109200988),
            (45.85409373846973, 59.02982010995767), (49.943802379514054, 65.51907182099147),
            (50.39868683219468, 73.17604150047285), (52.01874787331546, 80.67347484534697),
            (57.699774663693056, 85.8273121676535), (65.3070296248421, 86.8100514483688),
            (72.91428458599115, 87.79279072908409), (90, 90)]
    X_dimensions = np.array([(0, 100), (0, 100)])  # dimensions of Search Space
    # obstacles
    Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80), (60, 20, 80, 40), (60, 60, 80, 80)])
    x_init = (0, 0)  # starting location
    x_goal = (100, 100)  # goal location

    Q = np.array([(8, 4)])  # length of tree edges
    r = 0.1  # length of smallest edge to check for intersection with obstacles
    max_samples = 1024  # max number of samples to take before timing out
    prc = 0.1  # probability of checking for a connection to goal
    print("start")
    # create search space
    X = SearchSpace(X_dimensions, Obstacles)
    # create rrt_search
    rrt = RRTSpline(X, x_init, x_goal)
    pruned_path, _ = rrt.node_pruning(path)
    print(pruned_path)
    # rrt.add_vertex(0, rrt.x_init)
    # rrt.add_edge(0, rrt.x_init, None)
    # rrt.improved_transition((1, 1), 0)
    # print((1, 2)[0:1])
    # print((1, 2)+tuple())
