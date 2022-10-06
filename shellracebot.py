''' 
Shellbot is a framework for controlling bots from other threads
Author: Nicholas Lorentzen
Date: 20220607
'''

import json
import math
import pickle
import sys
import threading
from datetime import datetime, timedelta
from random import randint, random
from time import sleep
from typing import List, Tuple, Union

import libpyAI as ai
import numpy as np
from neat import nn

from consoleutils import delete_last_lines, get_bar_graph
from xpracefitness import get_fitness


class ShellBot(threading.Thread):

    ## Initialization Settings
    username: str = "Kerbal"
    headless: bool = False
    gamemap: str = ""
    test_mode: bool = False
    show_info: bool = False
    just_printed_info: bool = False

    ## Neat Info
    nn = None
    exploration_rate = 0.02
    cum_bonus = 0.0
    max_thrust_val = -999
    last_observations = np.zeros(22).tolist()

    ## Racing Info
    completed_course: bool = False
    done: bool = False
    start_marker: int = 50
    finish_marker: int = 1670
    start_time: datetime = datetime.now()
    course_time: float = -1.0
    average_speed: float = 0.0
    cum_speed: float = 0.0
    average_completion_per_frame: float = 0.0
    cum_completion_per_frame: float = 0.0
    completion: float = 0.0
    last_completion: float = 0.0
    max_completion: float = 0.0
    max_completion_frame: int = 0
    
    ## Configuration Settings
    safety_margin: float = 1.1
    turnspeed: int = 20
    power_level: float = 28.0
    scan_distance: int = 1000

    ## Control Values
    desired_heading: float = 0
    thrust: bool = True
    thrust_val: float = 1.0
    turn_val: float = 1.0
    last_thrust: float = 0.0
    last_turn: float = 0.0
    cum_avg_thrust = 0.0

    ## Needed for episode reset to work
    reset_now: bool = False
    awaiting_reset: bool = False
    close_now: bool = False
    frame: int = 0
    started: bool = False
    died_but_no_reset: bool = False
    reset_frame: int = 0
    ask_for_perms = False

    ## State of the bot
    alive: float = 0.0
    last_alive: float = 0.0

    ## Properties For Processing
    ## Ship Info
    heading: int = 90
    tracking: int = 90
    heading_x: float = 0
    heading_y: float = 0
    x_vel: float = 0
    y_vel: float = 0
    speed: float = 0.0
    wall_front: float = 1000
    wall_left: float = 1000
    wall_right: float = 1000
    wall_back: float = 1000
    track_wall: float = 1000
    closest_wall: float = 0
    closest_wall_heading: int = 0
    tt_retro: float = 140.0
    tt_tracking: float = 140.0
    tt_retro_point: float = 140.0

    ## Track Info
    checkpoints: List[List[int]] = [[0, 0]]

    def __init__(self, username:str="InitNoName", mapname: str = 'testtrack') -> None:
        super(ShellBot, self).__init__()
        self.username = username
        self.max_turntime = math.ceil(180 / self.turnspeed)
        self.gamemap = mapname
        with open(f'{self.gamemap}.json', 'r') as f:
            map_data = json.load(f)
            self.checkpoints = map_data["checkpoints"]
            self.finish_marker = map_data["finish_marker"]
        
    
    ## For Interfacing with the Environment
    def run(self,) -> None:
        if not self.started:
            self.started = True
            ai.start(self.run_loop, ["-name", self.username, "-join", "localhost"])
        else:
            print("Bot already started")
    
    def reset(self,) -> None:
        self.reset_now = True

    def close_bot(self,) -> None:
        self.close_now = True

    def reset_values(self,) -> None:
        self.awaiting_reset = False
        self.start_time = datetime.now()
        self.died_but_no_reset = False
        self.cum_bonus = 0.0
        self.done = False
        self.completed_course = False
        self.course_time = -1.0
        self.frame = 0
        self.average_speed = 0.0
        self.cum_speed = 0.0
        self.completion = 0.0
        self.last_completion = 0.0
        self.max_completion = 0.0
        self.max_completion_frame = 0

    def run_loop(self,) -> None:
        ##print(f"Bot starting frame {self.frame}")
        self.reset_flags()
        self.frame += 1

        try:
            if self.ask_for_perms:
                ai.talk("/password test")
                self.ask_for_perms = False
            if self.alive == 1.0 and self.awaiting_reset and self.reset_frame + 28 <= self.frame:
                self.reset_values()
                ai.thrust(1)
            if self.reset_now:
                self.reset_now = False
                ai.talk("/reset all")
                self.alive = 0.0
                self.last_alive = 0.0
                self.awaiting_reset = True
                self.reset_frame = self.frame
                return

            ai.setTurnSpeedDeg(self.turnspeed)
            
            self.collect_info()
            self.check_done()
            self.set_action()
            self.perform_action()
            ai.setPowerLevel(self.power_level)
        except AttributeError:
           pass
        except Exception as e:
             print("Error: " + str(e))
        self.calculate_bonus()

        if self.test_mode or self.show_info:
            self.print_info()
        else:
            self.just_printed_info = False
        self.last_alive = self.alive

    def print_info(self,) -> None:
        if self.just_printed_info:
            delete_last_lines(14)
        feeler_view = []
        for row in range(0, 11):
            blank_row = []
            for column in range(0, 21):
                blank_row.append(" ")
            feeler_view.append(blank_row)
        feeler_view[10][10] = "^"
        feelers = [self.wall_front, self.wall_left, self.wall_right, self.wall_30_left, self.wall_30_right]
        feeler_chars = [u'\u2502', '─', u'\u2572', u'\u2571']
        feeler_percents = []
        for idx, feeler in enumerate(feelers):
            feeler = float(min(feeler, 500.0))
            feeler_percent = feeler / 500.0
            feeler_percent = int(round(feeler_percent * 10.0))
            feeler_percents.append(feeler_percent)
        
        for idx in range(0, feeler_percents[0]):
            feeler_view[9 - idx][10] = feeler_chars[0]
        
        feeler_percents[1] = int(round(feeler_percents[1] / 1.0))
        for idx in range(0, feeler_percents[1]):
            feeler_view[10][9 - idx] = feeler_chars[1]
        
        feeler_percents[2] = int(round(feeler_percents[2] / 1.0))
        for idx in range(0, feeler_percents[2]):
            feeler_view[10][11 + idx] = feeler_chars[1]
        
        feeler_percents[3] = int(round(feeler_percents[3] / 1.0))
        for idx in range(0, feeler_percents[3]):
            feeler_view[9 - idx][9 - idx] = feeler_chars[2]

        feeler_percents[4] = int(round(feeler_percents[4] / 1.0))
        for idx in range(0, feeler_percents[4]):
            feeler_view[9 - idx][11 + idx] = feeler_chars[3]

        # checkpoint_x_diff = min(self.checkpoints[self.current_checkpoint][0] - self.x, 500.0) / 500.0
        # checkpoint_y_diff = max(min(self.checkpoints[self.current_checkpoint][1] - self.y, 500.0), 0) / 500.0
        # checkpoint_x_diff = int(round(checkpoint_x_diff * 10.0))
        # checkpoint_y_diff = int(round(checkpoint_y_diff * 10.0))

        # feeler_view[10 - checkpoint_y_diff][10 + checkpoint_x_diff] = "*"

        current_coursetime = round((datetime.now() - self.start_time).total_seconds(), 3)

        time_readout = f' Current Lap Time: {current_coursetime:6}s'
        completion_readout = f"    {get_bar_graph(self.completion / 100.0)}    - Course Completion: {self.completion / 100.0:.2%}"
        steering_readout = f"  1 {get_bar_graph(-self.turn_val, center=True)} -1 - Steering: {round(self.turn_val * 20.0, 2):+4} degrees"
        speed_readout = f"  0 {get_bar_graph(self.last_observations[0])}  1 - Speed: {round(self.speed, 2):4} units/frame"
        time_to_wall_readout = f"  0 {get_bar_graph(self.last_observations[13])}  1 - Time to Wall: {self.tt_tracking:3} frames"

        fitness = get_fitness(self.completion, self.cum_bonus, current_coursetime, self.average_speed, self.average_completion_per_frame)
        fitness_readout = f" Fitness: {round(fitness, 2):6} - Bonus: {round(self.cum_bonus, 2):4} - Average Speed: {round(self.average_speed, 2):4} - Average %/Frame: {self.average_completion_per_frame:.2%}"

        waypoint_distance, waypoint_bearing = self.get_checkpoint_info(self.current_checkpoint)
        waypoint_readout = f" Waypoint: {self.current_checkpoint:2} Distance: {round(waypoint_distance, 1):4} units Bearing: {round(waypoint_bearing, 2):+4} degrees"

        if self.thrust_val > self.cum_avg_thrust:
            self.cum_avg_thrust = self.thrust_val
        else:
            self.cum_avg_thrust = (self.cum_avg_thrust * 3 + self.thrust_val) / 4.0
        thrust_readout = f"  0 {get_bar_graph(self.cum_avg_thrust)}  1 - Thrust: {round(self.thrust_val, 2):3} Smoothed: {round(self.cum_avg_thrust, 2):3}"
        
        dash_graphs = [time_readout, completion_readout, thrust_readout, steering_readout, speed_readout, time_to_wall_readout, waypoint_readout, fitness_readout]

        output = '┌'
        for idx in range(len(feeler_view[0])):
            output += '─'
        output += '┐\n'

        for feeler_row in feeler_view:
            output += '│'
            for char in feeler_row:
                output += char
            output += '│'
            if len(dash_graphs) != 0:
                output += dash_graphs.pop(0)
            output += '\n'
        output += '└'
        for idx in range(0, len(feeler_view[0])):
            output += '─'
        output += '┘\n'

        print(output)
        self.just_printed_info = True
        
    def get_observations(self,) -> List[float]:
        ## Normalize the values
        wall_track = 1.0 - (float(self.track_wall) / float(self.scan_distance))
        closest_wall = 1.0 - (float(self.closest_wall) / float(self.scan_distance))
        wall_front = 1.0 - (float(self.wall_front) / float(self.scan_distance))
        wall_left = 1.0 - (float(self.wall_left) / float(self.scan_distance))
        wall_right = 1.0 - (float(self.wall_right) / float(self.scan_distance))
        wall_back = 1.0 - (float(self.wall_back) / float(self.scan_distance))
        wall_30_right = 1.0 - (float(self.wall_30_right) / float(self.scan_distance))
        wall_30_left = 1.0 - (float(self.wall_30_left) / float(self.scan_distance))
        wall_15_right = 1.0 - (float(self.wall_15_right) / float(self.scan_distance))
        wall_15_left = 1.0 - (float(self.wall_15_left) / float(self.scan_distance))

        ship_speed = self.speed / 20.0

        tt_tracking = max(1.0 - (self.tt_tracking / 140.0), 0.0)
        tt_retro_point = max(1.0 - (self.tt_retro_point / 70.0), 0.0)
        angle_diff_tracking = self.angle_diff(self.heading, self.tracking) / 180.0
        angle_diff_closest = self.angle_diff(self.heading, self.closest_wall_heading) / 180.0

        self.current_checkpoint = self.get_current_checkpoint()

        checkpoint_info = []
        first_checkpoint = True
        for checkpoint_num in range(self.current_checkpoint, self.current_checkpoint + 3):
            dist, angle = self.get_checkpoint_info(checkpoint_num)
            dist = 1.0 - (float(dist) / float(self.scan_distance))
            if first_checkpoint:
                ##print(f"Checkpoint Bearing: {self.angle_diff(self.heading, angle)}")
                first_checkpoint = False
            angle = self.angle_diff(self.tracking, angle) / 180.0
            checkpoint_info.append(dist)
            checkpoint_info.append(angle)


        ## Organize the values
        oberservations = [ship_speed, wall_track, angle_diff_tracking, closest_wall, angle_diff_closest, wall_front, wall_back, wall_left, wall_right, wall_15_right, wall_15_left, wall_30_right, wall_30_left, tt_tracking, tt_retro_point, self.last_thrust, self.last_turn]
        oberservations.extend(checkpoint_info)

        ## Check Normalization
        for idx in range(0, len(oberservations)):
            num = oberservations[idx]
            if math.isnan(num):
                num = 1.0
            num = min(num, 1.0)
            num = max(num, -1.0)
            oberservations[idx] = num
        self.last_observations = oberservations
        return oberservations

    def sigmoid_activation(self, x: np.ndarray) -> np.ndarray:
        return (1/(1+np.exp(-x)))
    
    def forward_propogation(self, inputs: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
        activations = self.sigmoid_activation(np.dot(inputs, weights) + biases)
        return activations

    def set_action(self,) -> None:
        ## Get the observations
        observations = self.get_observations()

        ## Forward Propogation
        inputs = np.array(observations)
        if not self.test_mode:
            outputs = self.nn.activate(inputs)
        else:
            outputs = [0,0]

        self.thrust_val = outputs[0]
        self.turn_val = outputs[1]

        self.thrust_val = max(min(self.thrust_val, 1.0), 0.0)

        if self.thrust_val > 0.75:
            self.thrust = True
            self.power_level = 28.0
        elif self.thrust_val > 0.5:
            self.thrust = True
            self.power_level = 21.0
        elif self.thrust_val > 0.25:
            self.thrust = True
            self.power_level = 14.0
        else:
            self.thrust = False
        
        self.turn_val = max(min(self.turn_val, 1.0), -1.0)
        self.desired_heading = self.angle_add(float(self.heading), self.turn_val * float(self.turnspeed))

        self.last_thrust = self.thrust_val
        self.last_turn = self.turn_val
    
    def calculate_bonus(self,) -> None:
        self.cum_speed += self.speed
        self.cum_completion_per_frame += self.completion - self.last_completion
        if self.frame != 0:
            self.average_speed = self.cum_speed / self.frame
            self.average_completion_per_frame = self.completion / self.frame
        else:
            self.average_completion_per_frame = 0.0
            self.average_speed = self.speed

        step_bonus = 0.0
        speed_bonus = 0.0
        if self.speed > 0.5:
            speed_bonus = (self.speed ** 1.1) / 150.0

        if self.alive == 1.0 and not self.done:
            step_bonus += speed_bonus
        self.cum_bonus += step_bonus
    
    def get_scores(self,) -> Tuple:
        completion_percentage_bonus = self.get_completion_percent()
        return round(self.cum_bonus, 3), round(completion_percentage_bonus, 3), round(self.course_time, 3)
    
    def get_completion_percent(self,) -> float:
        if self.y >= self.finish_marker or self.completed_course:
            return 100.0
        
        percent_per_checkpt = 100.0 / float(len(self.checkpoints) - 1)
        checkpt_idx = self.get_current_checkpoint() - 1

        try:
            if self.checkpoints[checkpt_idx + 1][1] >= self.finish_marker:
                percentage = 100.0 - percent_per_checkpt
                diff_finish = self.finish_marker - self.checkpoints[checkpt_idx][1]
                percent_to_finish = max(min(diff_finish - (self.finish_marker - self.y) / diff_finish, 1.0), 0.0)
                percentage += percent_to_finish * percent_per_checkpt
                percentage = max(min(round(percentage, 3), 99.999), 0.1)
                return percentage
        except:
            print("Error in get_completion_percent")

        base_percentage = percent_per_checkpt * checkpt_idx
        distance_to_checkpt = self.get_checkpoint_info(checkpt_idx)[0]
        distance_btw_checkpt = self.get_distance(self.checkpoints[checkpt_idx][0], self.checkpoints[checkpt_idx][1], self.checkpoints[min(checkpt_idx + 1, len(self.checkpoints) - 1)][0], self.checkpoints[min(checkpt_idx + 1, len(self.checkpoints) - 1)][1])
        percent_to_next = (distance_to_checkpt / distance_btw_checkpt) * percent_per_checkpt
        return max(min(round(base_percentage + percent_to_next, 3), 99.999), 0.1)

    
    def perform_action(self,) -> None:
        if self.thrust == True:
            ai.thrust(1)

        self.turn_to_degree(self.desired_heading)
    
    def reset_flags(self,) -> None:
        '''reset_flags Sets all flags to false and resets control states
        '''
        self.turn, self.thrust, self.shoot = False, False, False
        ai.turnLeft(0)
        ai.turnRight(0)
        ai.thrust(0)
        self.last_action_failed = 0.0
    
    def check_done(self,) -> None:
        '''check_done Checks if the bot is done
        '''
        if self.frame < 28 or self.done and not self.awaiting_reset:
            return
        if self.completion > self.max_completion:
            self.max_completion = self.completion
            self.max_completion_frame = self.frame
        elif self.frame - self.max_completion_frame > 280:
            self.done = True
        if self.y >= self.finish_marker and self.alive == 1.0 and not self.awaiting_reset:
            self.completed_course = True
            self.done = True
            course_time = datetime.now() - self.start_time
            self.course_time = course_time.total_seconds()
            ##print(f"Bot completed course in {round(self.course_time, 3)} seconds")
        if self.alive != 1.0 and not self.awaiting_reset:
            self.done = True
            self.course_time = -1.0

    def get_current_checkpoint(self,) -> int:
        '''get_current_checkpoint Returns the current checkpoint
        '''
        check_dists = []
        for checkpoint in self.checkpoints:
           check_dists.append(self.get_distance(self.x, self.y, checkpoint[0], checkpoint[1]))
        
        current_idx = int(np.argmin(check_dists))
        dist_to_next = check_dists[min(current_idx + 1, len(check_dists) - 1)]
        current_wpt = self.checkpoints[current_idx]
        next_wpt = self.checkpoints[min(current_idx + 1, len(check_dists) - 1)]

        dist_btw_next = self.get_distance(current_wpt[0], current_wpt[1], next_wpt[0], next_wpt[1])

        if dist_to_next < dist_btw_next + 25 and current_idx < len(self.checkpoints) - 1:
            current_idx += 1

        return current_idx

        # for idx, checkpoint in enumerate(self.checkpoints):
        #     if self.y + 40 > checkpoint[1]:
        #         return idx + 1

    
    def get_checkpoint_info(self, checkpoint_idx: int) -> tuple:
        '''get_checkpoint_info gets information about the checkpoint

        Args:
            checkpoint_idx (int): The index of the checkpoint

        Returns:
            tuple: (The distance to the checkpoint, The angle to the checkpoint)
        '''

        checkpoint_idx = max(min(checkpoint_idx, len(self.checkpoints) - 1), 0)
        checkpoint = self.checkpoints[checkpoint_idx]
        checkpoint_x = checkpoint[0]
        checkpoint_y = checkpoint[1]
        checkpoint_dist = self.get_distance(self.x, self.y, checkpoint_x, checkpoint_y)
        checkpoint_angle = self.get_angle_from_to(self.x, self.y, checkpoint_x, checkpoint_y)
        return checkpoint_dist, checkpoint_angle
    
    def turn_to_degree(self, degree: float) -> None:
        '''turn_to_degree Turns the bot to the desired heading

        Args:
            degree (float): Heading to turn to
        '''
        delta = self.angle_diff(self.heading, degree)
        if abs(delta) > 20:
            if delta < 0:
                ai.turnRight(1)
            else:
                ai.turnLeft(1)
        else:
            ai.turnToDeg(int(degree))

    def collect_info(self,) -> None:
        ## Basics
        self.alive = float(ai.selfAlive())

        self.x = ai.selfX()
        self.y = ai.selfY()

        self.heading = int(ai.selfHeadingDeg())
        self.heading_x, self.heading_y = self.get_components(self.heading, 1)

        self.speed = ai.selfSpeed()
        self.tracking = int(ai.selfTrackingDeg())
        self.x_vel = ai.selfVelX()
        self.y_vel = ai.selfVelY()

        ## Walls
        self.track_wall = ai.wallFeeler(self.scan_distance, self.tracking)
        self.wall_front = float(ai.wallFeeler(self.scan_distance, int(self.heading)))
        self.wall_back = float(ai.wallFeeler(self.scan_distance, int(self.angle_add(self.heading, 180))))
        self.wall_left = float(ai.wallFeeler(self.scan_distance, int(self.angle_add(self.heading, 90))))
        self.wall_right = float(ai.wallFeeler(self.scan_distance, int(self.angle_add(self.heading, -90))))
        self.wall_30_left = float(ai.wallFeeler(self.scan_distance, int(self.angle_add(self.heading, 30))))
        self.wall_30_right = float(ai.wallFeeler(self.scan_distance, int(self.angle_add(self.heading, -30))))
        self.wall_15_left = float(ai.wallFeeler(self.scan_distance, int(self.angle_add(self.heading, 15))))
        self.wall_15_right = float(ai.wallFeeler(self.scan_distance, int(self.angle_add(self.heading, -15))))

        ## Timings
        self.tt_tracking = math.ceil(float(self.track_wall) / (self.speed + 0.0000001))
        self.tt_retro = math.ceil(self.speed / float(self.power_level))

        ## Timings to timing pts
        self.tt_retro_point = min(self.tt_tracking - ((self.max_turntime + self.tt_retro) * self.safety_margin + 1), 70.0)

        self.last_completion = self.completion
        self.completion = self.get_completion_percent()

    ##Utility Functions
    def update_closest_wall(self,) -> None:
        '''update_closest_wall Updates the closest wall distance and heading
        '''
        self.closest_wall = self.scan_distance
        self.closest_wall_heading = -1
        for degree in range(0, 360, 30):
            wall = ai.wallFeeler(self.scan_distance, degree)
            if wall < self.closest_wall:
                self.closest_wall = wall
                self.closest_wall_heading = degree

    def get_closer_angle(self, a1: float, a2: float) -> float:
        '''get_closer_angle Returns whichever heading is closer to the current heading of the ship

        Args:
            a1 (float): First heading
            a2 (float): Second heading

        Returns:
            float: The heading that is closer to the current heading of the ship
        '''        
        if abs(self.angle_diff(self.heading, a1)) < abs(self.angle_diff(self.heading, a2)):
            return a1
        return a2

    def get_cpa_self(self, x, y, heading, speed, t_step: float = 1) -> Tuple[float, float, float, float]:
        '''get_cpa_self Ever wondered how long it will be until some object either hits or passes the ship? No... Well this function calculates that for you.

        Args:
            x (int): X coordinate of the object
            y (int): Y coordinate of the object
            heading (float): heading of the object
            speed (float): speed of the object

        Returns:
            Tuple[float, float, float, float]: The time to CPA in frames and the distance to CPA in pixels and the coordinates of the CPA
        '''
        return self.get_cpa(self.x, self.y, self.tracking, self.speed, x, y, heading, speed, t_step=t_step)

    def get_cpa(self, x1: int, y1: int, h1: int, s1: float, x2: int, y2: int, h2: int, s2: float, max_t: int = 9999, t_step: float = 1) -> Tuple[float, float, float, float]:
        '''get_cpa Ever wondered how long it will be until some object either hits or passes some other object? No... Well this function calculates that for you.

        Args:
            x1 (int): X coordinate of the first object
            y1 (int): Y coordinate of the first object
            h1 (int): heading of the first object
            s1 (float): speed of the first object
            x2 (int): X coordinate of the second object
            y2 (int): Y coordinate of the second object
            h2 (int): heading of the second object
            s2 (float): speed of the second object
            max_t (int, optional): max number of frames to calculate to. Defaults to 9999.
            t_step (float, optional): how much to increment for each step. Defaults to 1.

        Returns:
            Tuple[float, float, float, float]: The time to CPA in frames and the distance to CPA in pixels and the coordinates of the CPA
        '''        
        y_vel_1, x_vel_1 = self.get_components(h1, s1)
        y_vel_2, x_vel_2 = self.get_components(h2, s2)
        found_min = False
        last_dist: float = self.get_distance(x1, x1, x2, y2)
        t: float = 1
        while not found_min and t < max_t:
            current_dist = self.get_distance(
                x1 + x_vel_1 * t, y1 + y_vel_1 * t, x2 + x_vel_2 * t, y2 + y_vel_2 * t)
            if current_dist < last_dist:
                t += t_step
                last_dist = current_dist
            else:
                found_min = True
        if t >= max_t or t == 1:
            return -1, -1, -1, -1
        return t, last_dist, x1 + x_vel_1 * t, y1 + y_vel_1 * t

    def get_components(self, heading: float, speed: float) -> Tuple[float, float]:
        '''get_components Returns the x and y components of a given heading and speed

        Args:
            heading (float): heading of the object
            speed (float): speed of the object

        Returns:
            Tuple[float, float]: x and y components of the given heading and speed
        '''

        heading_rad: float = math.radians(heading)
        x: float = speed * math.cos(heading_rad)
        y: float = speed * math.sin(heading_rad)

        return x, y

    def get_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        '''get_distance Returns the distance between two points

        Args:
            x1 (float): x coordinate of the first point
            y1 (float): y coordinate of the first point
            x2 (float): x coordinate of the second point
            y2 (float): y coordinate of the second point

        Returns:
            float: Distance between the two points
        '''        
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def get_angle_to(self, x: float, y: float) -> float:
        '''get_angle_to Finds the heading the ship would need to face to get to a point

        Args:
            x (float): x coordinate of the point
            y (float): y coordinate of the point

        Returns:
            float: angle to the point
        '''        
        x_diff = x - self.x
        y_diff = y - self.y

        angle_to = math.degrees(
            math.atan(abs(y_diff) / abs(x_diff + 0.0000001)))

        ## Too lazy for a better way to do this
        ## See libpyAI.c line 2042 for more information
        if x_diff <= 0 and y_diff > 0:
            angle_to = 180 - angle_to
        elif x_diff <= 0 and y_diff <= 0:
            angle_to = 180 + angle_to
        elif x_diff > 0 and y_diff <= 0:
            angle_to = 360 - angle_to
        return angle_to

    def get_angle_from_to(self, x1: float, y1: float, x2: float, y2: float) -> float:
        '''get_angle_to Finds the heading from 1 point to another

        Args:
            x1 (float): x coordinate of the first point
            y1 (float): y coordinate of the first point
            x2 (float): x coordinate of the second point
            y2 (float): y coordinate of the second point

        Returns:
            float: angle to the second point from the first point
        '''        
        x_diff = x2 - x1
        y_diff = y2 - y1

        angle_to = math.degrees(
            math.atan(abs(y_diff) / abs(x_diff + 0.0000001)))

        ## Too lazy for a better way to do this
        ## See libpyAI.c line 2042 for more information
        if x_diff <= 0 and y_diff > 0:
            angle_to = 180 - angle_to
        elif x_diff <= 0 and y_diff <= 0:
            angle_to = 180 + angle_to
        elif x_diff > 0 and y_diff <= 0:
            angle_to = 360 - angle_to
        return angle_to

    def add_vectors(self, h1: int, s1: float, h2: int, s2: float) -> Tuple[float, float]:
        x_vel1, y_vel1 = self.get_components(h1, s1)
        x_vel2, y_vel2 = self.get_components(h2, s2)
        x_vel = x_vel1 + x_vel2
        y_vel = y_vel1 + y_vel2
        return self.get_angle_to(x_vel, y_vel), pow(pow(x_vel, 2) + pow(y_vel, 2), 0.5)

    def angle_add(self, a1: float, a2: float) -> float:
        '''angle_add Adds two angles together

        Args:
            a1 (float): angle 1
            a2 (float): angle 2

        Returns:
            float: result of adding the two angles
        '''       
        while a1 <= 0:
            a1 += 360
        while a2 <= 0:
            a2 += 360
        return (a1+a2+360) % 360

    def angle_diff(self, a1: float, a2: float) -> float:
        '''angle_diff Finds the difference between two angles

        Args:
            a1 (float): angle 1
            a2 (float): angle 2

        Returns:
            float: result of the difference between the two angles
        '''        
        diff = a2 - a1
        comp_diff = a2 + 360 - a1
        if abs(diff) < abs(comp_diff):
            return diff
        return comp_diff
    
    def get_tt_feeler(self, angle: float, distance: float, x_vel: float, y_vel: float) -> float:
        x_dist, y_dist = self.get_components(angle, distance)
        tt_x = x_dist / (x_vel + 0.0000001)
        tt_y = y_dist / (y_vel + 0.0000001)
        tt_feel = min(tt_x, tt_y)
        return tt_feel

if __name__ == "__main__":
    test = ShellBot('Test', 'testtrack')
    test.test_mode = True
    sleep(3)
    test.start()
    test.ask_for_perms = True
    while not test.done:
        pass
    print('END RUN 1')
    print(f'{test.get_scores()}')
    print(f'{test.average_completion_per_frame}')
    test.reset()
    while test.done:
        pass
    while not test.done:
        pass
    print('END RUN 2')
    print(f'{test.get_scores()}')
    test.reset()
    while test.done:
        pass
    sleep(30)
    print('END RUN 3')
    print(f'{test.get_scores()}')
    test.reset()
    while test.done:
        pass
    sleep(30)
    print('END RUN 4')
    print(f'{test.get_scores()}')

