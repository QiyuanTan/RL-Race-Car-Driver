from Agents.Driver import Driver
from gymnasium import ObservationWrapper

class SB3Driver(Driver):
    def __init__(self, env, model_class, policy="MultiInputPolicy", **kwargs):
        super().__init__(env)
        self.model_class = model_class
        self.model = model_class(policy, env, **kwargs)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = self.model_class.load(path)

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def act(self, obs):
        action, _states = self.model.predict(obs)
        return action

class SB3ObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        pass