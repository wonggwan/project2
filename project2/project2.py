import numpy as np
import math 
import random

def clip(n, minn, maxn):
    return max(min(maxn, n), minn)

def dynamics(xv, uv, ts=1):
    res = np.zeros_like(xv)
    tmp = uv[1] * ts / 2 + 1e-6
    sinc = np.divide(np.sin(tmp * np.pi), (tmp * np.pi))
    cal = np.multiply(uv[0], sinc)
    tmpx = np.multiply(cal, np.cos(xv[2] + tmp))
    tmpy = np.multiply(cal, np.sin(xv[2] + tmp))
    res[0] = xv[0] + tmpx
    res[1] = xv[1] + tmpy
    res[2] = xv[2] + ts * uv[1]
    return res

def get_obs_feature(element, x2, y2):
    
    # return the distance & angle of the obstalce in world coordinate system
    if len(element) == 3:
        # cylinder obstacle
        x1, y1, _ = element
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        phi = np.arctan(abs(y1-y2) / abs(x1-x2))
        # print(f"obs [{element}]: distance: {distance} | phi: {phi}")
        return distance, phi
    elif len(element) == 4:
        # print(f"current robot: {x2} | {y2}")
        x1, y1, l, h = element
        # distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        # calculate distance from a point to a line segment
        if x1 > 0:
            # right wall
            phi = 0
            # print(f"right: {element}")
            x1 -= l/2  # we assume the left/right wall is thick with l 
            xa = x1 
            ya = y1 + h/2 # upper end-point
            xb = x1 
            yb = y1 - h/2 # lower end-point
            # print(f"right: [{xa}, {ya}, {xb}, {yb}]")

        elif x1 < 0:
            # left wall
            phi = np.pi
            # print(f"left: {element}")
            x1 += l/2 
            xa = x1 
            ya = y1 + h/2 # upper end-point 
            xb = x1 
            yb = y1 - h/2 # lower end-point
            # print(f"left: [{xa}, {ya}, {xb}, {yb}]")

        elif y1 > 0:
            # up wall
            phi = np.pi/2
            # print(f"up: {element}")
            y1 -= h/2 # we assume the wall is thick with h in upper/lower wall
            xa = x1 - l/2 # left end-point 
            ya = y1 
            xb = x1 + l/2 # right end-point 
            yb = y1 
            # print(f"up: [{xa}, {ya}, {xb}, {yb}]")

        elif y1 < 0:
            # lower wall
            phi = -np.pi/2
            # print(f"low: {element}")
            y1 += h/2 
            xa = x1 - l/2 # left end-point 
            ya = y1 
            xb = x1 + l/2 # right end-point 
            yb = y1
            # print(f"low: [{xa}, {ya}, {xb}, {yb}]")
        distance = distance_point_to_segment(x2, y2, xa, ya, xb, yb)
        # print(f"Wall distance: {distance}\n")
        return distance, phi
    else:
        raise Exception("NotImplemented")
        
    
def distance_point_to_segment(x, y, x1, y1, x2, y2):
    # Calculate the distance from the point (x, y) to the line (x1, y1) - (x2, y2)
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1
    dot_product = A * C + B * D
    line_length_squared = C * C + D * D
    if line_length_squared == 0:
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    t = dot_product / line_length_squared
    if t < 0:
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    elif t > 1:
        return math.sqrt((x - x2) ** 2 + (y - y2) ** 2)
    else:
        closest_x = x1 + t * C
        closest_y = y1 + t * D
        return math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)



def update_feature_with_map(current_map, state, obstacles, sensor_range, is_collect=False):
    x2, y2, _ = state # robot xyt state

    """
    During data collection, we need to sample 0 ~ 2 obstacles in the environment 
    for different cases where the robot is in different stage of map exploration
    """
    for elm in obstacles: 
        # in the case when we use features to collect data (is_collect=True)
        # because current_map is initialized as [], we first collect the obstacles near the state
        distance, phi = get_obs_feature(elm, x2, y2)
        # collect (possible) obstacles that lie within the sensor_range 
        if distance <= sensor_range and elm not in current_map:
            current_map.append(elm)
    if is_collect:
        # we sample random number of obstacles in the environment
        # (we count the ones that were collected into this sampling as well)
        num_of_obs_to_show = np.random.randint(len(obstacles))
        obs = random.choices(obstacles, k=num_of_obs_to_show)
        if len(obs) > 0:
            for element in obs:
                if element not in current_map:
                    current_map.append(element)
    observable_list = []
    for elm in current_map:
        distance, phi = get_obs_feature(elm, x2, y2)
        observable_list.append((distance, phi))
    if len(current_map) >= 2:
        # print("\ncurrent map >= 2")
        # if there are more than 2 obstacles within the known map
        res = sorted(observable_list, key = lambda x: x[0])[:2]
        l1, phi_1 = res[0]
        l2, phi_2 = res[1]
    elif len(current_map) == 1:
        # print("current map == 1")
        l1, phi_1 = get_obs_feature(current_map[0], x2, y2)
        l2, phi_2 = -1, 0  # only 1 obstacle is near the robot
    elif len(current_map) == 0:
        l1, l2 = -1, -1
        phi_1, phi_2 = 0, 0
    else:
        raise Exception("Impossible case")
    feature = [l1, phi_1]
    # print(f"feature: {feature}")
    return current_map, feature

def move(state, action, cur_map, obstacles, sensor_range):
    l, rho, x, y, theta = state 
    xv = [x, y, theta]

    ws_size = 1.5 
    uv = action
    """
    For lab experiment, set high covariance for area near goal_1 and goal_2
    Goal 1: (0.4, 0.4) | radius: 8 cm
    Goal 2: (-0.4, -0.4) | radius: 8 cm
    For area [-0.4, -0.4] ~ [0.4, 0.4]
    N (0.002, 2)
    y1 = 1.125x1 + 0.03
    y2 = 1.125x2 + -0.13
    Others
    N (0.002, 0.001)
    """
    noise_v = np.random.normal(0.002, 0.001)
    if -0.3 < x < 0.3 and -0.3 < y < 0.3:
        noise_w = np.random.normal(0.002, 0.5)
    else:
        noise_w = np.random.normal(0.002, 0.001)
    uv0 = clip(uv[0] + noise_v, -0.26, 0.26)
    uv1 = clip(uv[1] + noise_w, -1.82, 1.82)
    uv = [uv0, uv1]

    state_next = dynamics(xv, uv)
    xp = clip(state_next[0], -ws_size, ws_size)
    yp = clip(state_next[1], -ws_size, ws_size)
    tp = clip(state_next[2], -np.pi, np.pi)
    
    next_x = [xp, yp, tp]

    new_map, s_ = update_feature_with_map(cur_map, next_x, obstacles, sensor_range)

    return list(s_) + list(next_x), new_map