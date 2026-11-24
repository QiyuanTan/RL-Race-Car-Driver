"""Geometry-only Track Lookahead student"""

import gymnasium as gym


class GeometryOnlyObservationWrapper(gym.ObservationWrapper):
    """Expose existing track (5) and lookahead (10) features, no raw image/LiDAR."""

    KEYS = ('track', 'lookahead')

    def __init__(self, env):
        super().__init__(env)
        for key in self.KEYS:
            if key not in env.observation_space.spaces:
                raise KeyError(f'Geometry-only DQN requires {key} in observations')
        self.observation_space = gym.spaces.Dict({
            key: env.observation_space.spaces[key] for key in self.KEYS
        })

    def observation(self, observation):
        return {key: observation[key] for key in self.KEYS}
