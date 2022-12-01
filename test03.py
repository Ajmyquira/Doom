import os
from stable_baselines3.common.callbacks import BaseCallback


class TrainAndLoggingCallBack(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallBack, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


CHECKPOINT_DIR = './train/train_basic'
LOG_DIR = './logs/log_basic'

callback = TrainAndLoggingCallBack(check_freq=10000, save_path=CHECKPOINT_DIR)
