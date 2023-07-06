import logging

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou


class TrajPlanning(object):
    def __init__(self, planning_time=1.0, interpolate_num=10, verbose=False) -> None:
        # create an SE3 state self.space
        self.space = ob.RealVectorStateSpace(6)

        # set lower and upper bounds
        bounds = ob.RealVectorBounds(6)
        for i in range(6):
            bounds.setLow(i, -2)
            bounds.setHigh(i, 2)
        self.space.setBounds(bounds)

        # create a simple setup object
        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))

        # Setup custom planner
        self.planner = None
        plan_range = 0.5
        self.planning_time = planning_time
        self.interpolate_num = interpolate_num
        self.set_planner("RRT", plan_range)  # RRT by default

        if verbose is False:
            ompl_logger = logging.getLogger(ou.__name__)
            ompl_logger.setLevel(logging.WARNING)

    def isStateValid(self, state):
        # collision detect
        return True

    def plan(self, cur_pos, goal_pos, cur_vel, goal_vel=np.zeros(3)):
        self.ss.clear()

        start_pos = cur_pos
        start_vel = cur_vel
        start = ob.State(self.space)
        goal = ob.State(self.space)
        for i in range(3):
            start[i] = start_pos[i]
            goal[i] = goal_pos[i]
            start[i+3] = start_vel[i]
            goal[i+3] = goal_vel[i]

        self.ss.setStartAndGoalStates(start, goal)

        # this will automatically choose a default planner with default parameters
        solved = self.ss.solve(self.planning_time)

        res = False
        sol_path_list = []
        if solved:
            # try to shorten the path
            self.ss.simplifySolution()
            # print the path to screen
            sol_path_geometric = self.ss.getSolutionPath()
            # interpolate the path points
            sol_path_geometric.interpolate(self.interpolate_num)

            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            # checkout again to ensure points is correct
            for sol_path in sol_path_list:
                self.isStateValid(sol_path)
            res = True
        else:
            print("No solution found")
        return res, sol_path_list

    def state_to_list(self, state):
        return [state[i] for i in range(6)]

    def set_planner(self, planner_name, r):
        """
            Setup planner.
        """
        try:
            exec(f'self.planner = og.{planner_name}(self.ss.getSpaceInformation())')
        except ValueError:
            raise ValueError("Invalid planner_name param.")
        finally:
            self.planner.setRange(r)
            self.ss.setPlanner(self.planner)
