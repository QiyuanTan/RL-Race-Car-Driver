# Defines drivers for controlling a vehicle in a simulation environment.

class Driver:
    def __init__(self, env):
        self.__action_space = env.action_space

    def act(self, obs: list[dict]) -> dict:
        """
        Take an action in the environment. Supports vectorized environments, the obs and action_space are lists of
        observations and action spaces for each environment instance.

        :param obs: list of observation from the environment
        keys:
            - 'pose': vehicle pose Box(6,) with [x, y, z, roll, pitch, yaw] in order
            - 'velocity': vehicle velocity Box(6,) with [vx, vy, vz, roll_rate, pitch_rate, yaw_rate] in order
            - 'acceleration': vehicle acceleration Box(6,) with [ax, ay, az, roll_accel, pitch_accel, yaw_accel] in order
            - 'lidar': vehicle lidar Box(<scans>,) with 1080 scans by default
            - 'camera': vehicle camera Box(<height>, <width>, 3) with height 240 and width 320 by default
        :return: vectorized actions to take in the environment

        """
        raise NotImplementedError()


class DummyDriver(Driver):
    def act(self, obs):
        return self.__action_space.sample()

class HitTheGasDriver(Driver):
    def act(self, obs):
        return {"motor": 1.0, "steering": 0.0}