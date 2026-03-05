import abc

class BaseLogic(abc.ABC):
    @abc.abstractmethod
    def get_action(self, state, training=True):
        """Returns action indices and any metadata (logits/q-values)."""
        pass

    @abc.abstractmethod
    def train_step(self, state, action, reward, next_state, done):
        """Returns a scalar loss."""
        pass

    @abc.abstractmethod
    def save_models(self, folder_path):
        """Saves internal parameters."""
        pass

    @abc.abstractmethod
    def load_models(self, folder_path, state_shape, train=True):
        """Restores internal parameters."""
        pass