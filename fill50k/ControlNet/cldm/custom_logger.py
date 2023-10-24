from pytorch_lightning.loggers.logger import WandbLogger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

class MyLogger(WandbLogger):
    @property
    def name(self) -> str:
        return "MyWandbLogger"
    
    @property
    def version(self) -> str:
        return "0.1"
    
    @rank_zero_only
    def log_hyperparams(self, params):
        print('log_hyperparams')
        print(params)
        print()

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        print('log_metrics')
        print(metrics)
        print()

    @rank_zero_only
    def save(self):
        print('save')

    @rank_zero_only
    def finalize(self, status):
        print('finalize')
        print(status)
        print()
    