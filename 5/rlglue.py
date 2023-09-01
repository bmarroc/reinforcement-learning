#!/usr/bin/env python

"""Glues together an experiment, agent, and environment.
"""

class RLGlue:
    """RL class

    args:
        env_name (string): the name of the module where the Environment class can be found
        agent_name (string): the name of the module where the Agent class can be found
    """

    def __init__(self, env, agent):
        self.environment = env
        self.agent = agent

        self.last_action = None
        self.num_episodes = None
        self.num_steps = None
        self.max_steps_this_episode = None

    def rl_init(self, agent_init_info={}, env_init_info={}):
        """Initial method called when RLGlue experiment is created"""
        self.environment.env_init(env_init_info)
        self.agent.agent_init(agent_init_info)

        self.num_episodes = 0

    def rl_start(self, agent_start_info={}, env_start_info={}):
        """Starts RLGlue experiment

        Returns:
            tuple: (state, action)
        """

        state = self.environment.env_start()
        self.last_action = self.agent.agent_start(state)
        is_terminal = False
        obs_act = [state, self.last_action, is_terminal]
        self.num_steps = 0

        return obs_act

    def rl_step(self):
        """Step taken by RLGlue, takes environment step and either step or
            end by agent.

        Returns:
            (float, state, action, Boolean): reward, last state observation,
                last action, boolean indicating termination
        """

        [reward, state, is_terminal] = self.environment.env_step(self.last_action)

        
        if self.num_steps == self.max_steps_this_episode-1:
            is_terminal = True
            
        if is_terminal:
            self.agent.agent_end(reward)
            reward_obs_act_term = [reward, state, None, is_terminal]
            self.num_steps += 1
            self.num_episodes += 1
        else:
            self.last_action = self.agent.agent_step(reward, state)
            reward_obs_act_term = [reward, state, self.last_action, is_terminal]
            self.num_steps += 1

        return reward_obs_act_term

    def rl_episode(self, max_steps_this_episode):
        """Runs an RLGlue episode

        Args:
            max_steps_this_episode (Int): the maximum steps for the experiment to run in an episode

        Returns:
            Boolean: if the episode should terminate
        """
        self.max_steps_this_episode = max_steps_this_episode
        
        obs_act_term = self.rl_start()
        is_terminal = obs_act_term[2]

        while (not is_terminal):
            reward_obs_act_term = self.rl_step()
            is_terminal = reward_obs_act_term[3]
    
    def rl_cleanup(self):
        """Cleanup done at end of experiment."""
        self.environment.env_cleanup()
        self.agent.agent_cleanup()
        self.last_action = None
        self.num_episodes = None
        self.num_steps = None
        self.max_steps_this_episode = None
