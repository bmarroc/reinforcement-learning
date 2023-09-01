#!/usr/bin/env python

"""Class for plotting the value function.
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

class Manager:
    def __init__(self, env_info, agent_info, experiment_name=None):
        self.experiment_name = experiment_name
        
        self.grid_h, self.grid_w = env_info["grid_height"], env_info["grid_width"]
        self.cmap = matplotlib.cm.viridis
            
        self.values_table = None
        self.policy = agent_info["policy"]
        
    def compute_values_table(self, values):
        self.values_table = np.empty((self.grid_h, self.grid_w))
        self.values_table.fill(np.nan)
        for state in range(len(values)):
            self.values_table[np.unravel_index(state, (self.grid_h, self.grid_w))] = values[state]
                     
    def visualize(self, values, num_episodes):
        if not hasattr(self, "fig"):
            self.fig = plt.figure(figsize=(10, 20))
            plt.ion()

        self.fig.clear()

        self.compute_values_table(values)
        plt.xticks([])
        plt.yticks([])
        im = plt.imshow(self.values_table, cmap=self.cmap, interpolation='nearest', origin='upper')
        
        for state in range(self.policy.shape[0]):
            for action in range(self.policy.shape[1]):
                y, x = np.unravel_index(state, (self.grid_h, self.grid_w))
                pi = self.policy[state][action]
                if pi == 0:
                    continue
                if action == 0:
                    plt.arrow(x, y, 0,  -0.5 * pi, fill=False, length_includes_head=True, head_width=0.1, 
                              alpha=0.5)
                if action == 1: 
                    plt.arrow(x, y, -0.5 * pi, 0, fill=False, length_includes_head=True, head_width=0.1, 
                              alpha=0.5)
                if action == 2:
                    plt.arrow(x, y, 0, 0.5 * pi, fill=False, length_includes_head=True, head_width=0.1, 
                              alpha=0.5)
                if action == 3:
                    plt.arrow(x, y, 0.5 * pi, 0, fill=False, length_includes_head=True, head_width=0.1, 
                              alpha=0.5)
        
        plt.title((("" or self.experiment_name) + "\n") + "Predicted Values, Episodes: %d" % num_episodes)
        plt.colorbar(im, orientation='horizontal')
        
        self.fig.canvas.draw()