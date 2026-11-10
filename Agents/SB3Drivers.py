from Driver import Driver

class SB3Driver(Driver):
    def __init__(self, env, model_class, policy="MlpPolicy", **kwargs):
        super().__init__(env)
        self.model_class = model_class
        self.model = model_class(policy, env, **kwargs)

    def load_model(self, path):
        self.model = self.model_class.load(path)

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def act(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return action
