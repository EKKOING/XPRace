''' 
Shellbot is a framework for controlling bots from other threads
Author: Nicholas Lorentzen
Date: 20220607
'''

from datetime import datetime, timedelta
import json
import math
import pickle
import threading
from random import randint, random
from time import sleep
from typing import List, Tuple, Union

import libpyAI as ai
import numpy as np
from neat import nn


class ShellBot(threading.Thread):

    ## Initialization Settings
    username: str = "Kerbal"
    headless: bool = False

    ## Neat Info
    nn = None
    exploration_rate = 0.02
    cum_bonus = 0.0
    max_thrust_val = -999

    ## Racing Info
    gamemap: str = "testtrack"
    completed_course: bool = False
    done: bool = False
    start_marker: int = 50
    finish_marker: int = 3400
    start_time: datetime = datetime.now()
    course_time: float = -1.0
    average_speed: float = 0.0
    cum_speed: float = 0.0
    
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

    ## Needed for episode reset to work
    reset_now: bool = False
    awaiting_reset: bool = False
    close_now: bool = False
    frame: int = 0
    started: bool = False
    died_but_no_reset: bool = False
    reset_frame: int = 0

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

    def __init__(self, username:str="InitNoName") -> None:
        super(ShellBot, self).__init__()
        self.username = username
        self.max_turntime = math.ceil(180 / self.turnspeed)
        with open(f'{self.gamemap}.json', 'r') as f:
            self.checkpoints = json.load(f)
        
    
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

    def run_loop(self,) -> None:
        ##print(f"Bot starting frame {self.frame}")
        self.reset_flags()
        self.frame += 1
        if self.alive == 1.0 and self.awaiting_reset and self.reset_frame + 28 <= self.frame:
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
            ai.thrust(1)

        try:
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
        ##print(f"Cpt: {self.current_checkpoint} Done: {self.done} Course Comp.: {self.completed_course}")
        ##print(f"Scores (Bonus, Completion %, Time): {self.get_scores()}")
        ##print(f'Alive {self.alive} - Done {self.done} - Course Comp. {self.completed_course} - Awaiting Reset {self.awaiting_reset} - Frame {self.frame} - Reset Frame {self.reset_frame}')
        self.last_alive = self.alive

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
        for checkpoint_num in range(self.current_checkpoint + 1, self.current_checkpoint + 4):
            dist, angle = self.get_checkpoint_info(checkpoint_num)
            dist = 1.0 - (float(dist) / float(self.scan_distance))
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
        outputs = self.nn.activate(inputs)
        ##outputs = [0,0]

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
        if self.frame != 0:
            self.average_speed = self.cum_speed / self.frame
        else:
            self.average_speed = self.speed

        step_bonus = 0.0
        speed_bonus = 0.0
        if self.speed > 0.5:
            speed_bonus = (self.speed ** 1.1) / 150.0

        if self.alive == 1.0 and not self.done:
            step_bonus += speed_bonus
        self.cum_bonus += step_bonus
    
    def get_scores(self,) -> Tuple:
        completion_percentage_bonus = min((float(self.y - self.start_marker) / float(self.finish_marker - self.start_marker)) * 100.0, 100.0)
        return self.cum_bonus, completion_percentage_bonus, self.course_time
    
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
        if self.y >= self.finish_marker and self.alive == 1.0 and not self.awaiting_reset:
            self.completed_course = True
            self.done = True
            course_time = datetime.now() - self.start_time
            self.course_time = course_time.total_seconds()
            print(f"Bot completed course in {round(self.course_time, 3)} seconds")
        if self.alive != 1.0 and not self.awaiting_reset:
            self.done = True
            self.course_time = -1.0

    def get_current_checkpoint(self,) -> int:
        '''get_current_checkpoint Returns the current checkpoint
        '''
        check_dists = []
        for idx, checkpoint in enumerate(self.checkpoints):
            check_dists.append(self.get_distance(self.x, self.y, checkpoint[0], checkpoint[1]))
        
        current_idx = int(np.argmin(check_dists))
        return current_idx
    
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
        y_diff = y2 - x1

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
    test = ShellBot('Test')
    test.start()
    while not test.done:
        pass
    print('END RUN 1')
    print(f'{test.get_scores()}')
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

