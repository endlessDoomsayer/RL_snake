import abc

class BaseLogic(abc.ABC):
    @abc.abstractmethod
    def __init__(self, encoder):
        self.encoder = encoder
    
    @abc.abstractmethod
    def get_action(self, state, training=True):
        pass

    @abc.abstractmethod
    def train_step(self, state, action, reward, next_state, done):
        pass

    @abc.abstractmethod
    def save_models(self, folder_path):
        pass

    @abc.abstractmethod
    def load_models(self, folder_path, state_shape, train=True):
        pass