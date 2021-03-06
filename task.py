import numpy as np
from physics_sim import PhysicsSim

class Land():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Task-specific parameters
        self.max_duration = 5.0  # secs
        self.start_z = 10.0
        self.target_z = 0.0  # target height (z position) to reach for successful landing


        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # reward = 0
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()


        #Compute reward / penalty and check if this episode is complete
        done = False
        reward = 0.0
        
        optimum_position = self.start_z + (self.target_z - self.start_z)*timestamp/self.max_duration
        
        if abs(optimum_position-pose.position.z)<1.0:
            reward += 1
        else:
            reward -= 1
        
        # increase reward in target region
        if pose.position.z <= 0.5:  # agent has crossed the target height
            reward += 2.0  # bonus reward
            done = True
            
        if timestamp > self.max_duration:  # agent has run out of time
            done = True

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state