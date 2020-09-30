# import numpy as np
# from gym import spaces
# from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
# import task

# class Land(Task):
#     """Simple task where the goal is to smoothly reach ground height again"""

#     def __init__(self):
#         # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
#         cube_size = 300.0  # env is cube_size x cube_size x cube_size
#         self.observation_space = spaces.Box(
#             np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
#             np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))

#         # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
#         max_force = 25.0
#         max_torque = 25.0
#         self.action_space = spaces.Box(
#             np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
#             np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))

#         # Task-specific parameters
#         self.max_duration = 5.0  # secs
#         self.start_z = 10.0
#         self.target_z = 0.0  # target height (z position) to reach for successful landing

#     def reset(self):
#         # Nothing to reset; just return initial condition
#         return Pose(
#                 position=Point(0.0, 0.0, self.start_z),  # drop off from a slight random height
#                 orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
#             ), Twist(
#                 linear=Vector3(0.0, 0.0, 0.0),
#                 angular=Vector3(0.0, 0.0, 0.0)
#             )

#     def update(self, timestamp, pose, angular_velocity, linear_acceleration):
#         # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
#         state = np.array([
#                 pose.position.x, pose.position.y, pose.position.z,
#                 pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])                

#         # Compute reward / penalty and check if this episode is complete
#         done = False
#         reward = 0.0
        
#         optimum_position = self.start_z + (self.target_z - self.start_z)*timestamp/self.max_duration
        
#         if abs(optimum_position-pose.position.z)<1.0:
#             reward += 1
#         else:
#             reward -= 1
        
#         # increase reward in target region
#         if pose.position.z <= 0.5:  # agent has crossed the target height
#             reward += 2.0  # bonus reward
#             done = True
            
#         if timestamp > self.max_duration:  # agent has run out of time
#             done = True

#         # Take one RL step, passing in current state and reward, and obtain action
#         # Note: The reward passed in here is the result of past action(s)
#         action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

#         # Convert to proper force command (a Wrench object) and return it
#         if action is not None:
#             action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
#             return Wrench(
#                     force=Vector3(action[0], action[1], action[2]),
#                     torque=Vector3(action[3], action[4], action[5])
#                 ), done
#         else:
#             return Wrench(), done



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

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 0
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
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