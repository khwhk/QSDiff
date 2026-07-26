# Code from: https://github.com/Ladbaby/PyOmniTS
# distributed training

import os

# Accelerate disables unsupported NCCL P2P/IB paths on RTX 4000 GPUs when
# launched through its CLI. Apply the same defaults for direct `python` runs.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
projectConfiguration = ProjectConfiguration(
    automatic_checkpoint_naming=True,
    total_limit=0, # disable save state to minimize disk write time
)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb", project_config=projectConfiguration)

# logging

import logging
import logging.handlers as handlers

import coloredlogs


class MainProcessLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def _log(self, level, msg, *args, **kwargs):
        if accelerator.is_main_process:
            super()._log(level, msg, *args, **kwargs)

logging.setLoggerClass(MainProcessLogger)
logger = logging.getLogger('global')

os.makedirs('logs', exist_ok=True)

logHandler = handlers.RotatingFileHandler('logs/debug.log', maxBytes=5242880, backupCount=2, encoding='utf-8')
logger.addHandler(logHandler)
coloredlogs.install(level="DEBUG", logger=logger, fmt='%(asctime)s %(levelname)s %(message)s')
