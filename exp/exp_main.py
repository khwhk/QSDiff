# Code from: https://github.com/Ladbaby/PyOmniTS
import datetime
import gc
import importlib
import json
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from accelerate import load_checkpoint_in_model
from torch import Tensor, optim
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LRScheduler,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.ExpConfigs import ExpConfigs
from utils.globals import accelerator, logger
from utils.metrics import (
    metric,
    metric_classification,
)
from utils.tools import (
    EarlyStopping,
    test_gpu_memory,
    test_params_flop,
    test_train_time,
)

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, configs: ExpConfigs):
        '''
        Available variables:
        - self.configs
        - self.device
        '''
        super(Exp_Main, self).__init__(configs)

    def _build_model(self) -> Module:
        # dynamically import the desired model class
        model_module = importlib.import_module("models." + self.configs.model_name)
        model = model_module.Model(self.configs)
        return model

    def _get_data(self, flag: str) -> tuple[Dataset, DataLoader]:
        data_set, data_loader = data_provider(self.configs, flag)
        return data_set, data_loader

    def _select_optimizer(self, model: Module) -> optim.Optimizer:
        model_optim = optim.Adam(model.parameters(), lr=self.configs.learning_rate)
        return model_optim

    def _select_lr_scheduler(self, optimizer: optim.Optimizer) -> LRScheduler:
        # Initialize scheduler based on configs.lradj
        if self.configs.lr_scheduler == 'ExponentialDecayLR':
            '''
            Originally named as 'type1'
            '''
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** epoch)
        elif self.configs.lr_scheduler == 'ManualMilestonesLR':
            '''
            Originally named as 'type2'
            '''
            from lr_schedulers.ManualMilestonesLR import ManualMilestonesLR

            # Convert 1-based epochs to 0-based
            user_milestones = {2:5e-5, 4:1e-5, 6:5e-6, 8:1e-6, 10:5e-7, 15:1e-7, 20:5e-8}
            milestones = {k-1: v for k, v in user_milestones.items()}
            scheduler = ManualMilestonesLR(optimizer, milestones)
        elif self.configs.lr_scheduler == 'DelayedStepDecayLR':
            '''
            Originally named as 'type3'
            '''
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 if epoch < 2 else (0.8 ** (epoch - 2)))
        elif self.configs.lr_scheduler == 'CosineAnnealingLR':
            '''
            Originally named as 'cosine'
            '''
            scheduler = CosineAnnealingLR(optimizer, T_max=self.configs.train_epochs, eta_min=0.0)
        elif self.configs.lr_scheduler == "MultiStepLR":
            '''
            Configured following CSDI
            '''
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[0.75 * self.configs.train_epochs, 0.9 * self.configs.train_epochs], gamma=self.configs.lr_scheduler_gamma
            )
        else:
            logger.exception(f"Unknown lr scheduler '{self.configs.lr_scheduler}'", stack_info=True)
            exit(1)

        return scheduler

    def _select_criterion(self) -> Module:
        # dynamically import the desired loss function
        loss_module = importlib.import_module("loss_fns." + self.configs.loss)
        criterion = loss_module.Loss(self.configs)
        return criterion

    def _get_state_dict(self, path: Path) -> OrderedDict:
        '''
        Fix model state dict errors
        '''
        logger.info(f"Loading model checkpoint from {path}")
        state_dict = torch.load(path, map_location="cpu")
        new_state_dict = OrderedDict()
        if_fixed = False
        for key, value in state_dict.items():
            # you may insert modifications to the key and value here
            if 's4' in key and (('B' in key or 'P' in key or 'w' in key) and ('weight' not in key)):
                # S4 layer don't need to load these weights
                if_fixed = True
                continue
            new_state_dict[key] = value.contiguous()
        if if_fixed:
            logger.warning("Automatically fixed model state dict errors. It may cause unexpected behavior!")
        return new_state_dict

    def _check_model_outputs(self, batch:dict, outputs:dict) -> None:
        '''
        Perform necessary checks on model's outputs
        '''
        # check if the data type is dict
        if type(outputs) is not dict:
            logger.exception(f"Expect model's forward function to return dict. Current output's data type is {type(outputs)}.", stack_info=True)
            exit(1)

        if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
            # check if outputs' true is the the same as input dataset's y
            if "true" in outputs.keys() and not torch.equal(batch["y"], outputs["true"]):
                logger.warning(f"Model's outputs['true'] is not equal to input's batch['y']. Please confirm that you are not using input's batch['y'] as ground truth. This is expected in some models such as diffusion.")
        elif self.configs.task_name in ["classification"]:
            if "true_class" in outputs.keys():
                # check if outputs' true_class is the the same as input dataset's y_class
                if not torch.equal(batch["y_class"], outputs["true_class"]):
                    logger.warning(f"Model's outputs['true_class'] is not equal to input's batch['y_class']. Please confirm that you are not using input's batch['y_class'] as ground truth.")

    def _merge_gathered_dicts(self, dicts: list[dict]) -> dict:
        '''
        manually merge list of dictionary gathered when testing
        accelerate.gather_for_metrics may have unexpected behavior, thus merge manually instead
        '''
        merged_dict = {}
        keys_not_returned = []
        for d in dicts:
            for key, tensor in d.items():
                if type(tensor).__name__ != "Tensor":
                    # skip value that is not PyTorch Tensor
                    if key not in keys_not_returned:
                        keys_not_returned.append(key)
                        logger.warning(f"{key=} will not be gathered for metric calculation in test, since its value has data type '{type(tensor).__name__}', which is not 'Tensor'")
                    continue
                if key in merged_dict:
                    merged_dict[key] = torch.cat((merged_dict[key], tensor.detach().cpu()), dim=0)
                else:
                    merged_dict[key] = tensor.detach().cpu()
        return merged_dict

    def vali(
        self, 
        model_train: Module, 
        vali_loader: DataLoader, 
        criterion: Module, 
        current_epoch: int,
        train_stage: int
    ) -> np.ndarray:
        total_loss = []
        model_train.eval()
        with torch.no_grad():
            with tqdm(total=len(vali_loader), leave=False, desc="Validating") as it:
                batch: dict[str, Tensor] # type hints
                for i, batch in enumerate(vali_loader):
                    # warn if the size does not match
                    if batch[next(iter(batch))].shape[0] != self.configs.batch_size and current_epoch == 0:
                        logger.warning(f"Exp_Main.vali(): Batch No.{i} out of [0~{len(vali_loader) - 1}] has an actual batch_size={batch[next(iter(batch))].shape[0]}, which is not the same as --batch_size {self.configs.batch_size}")
                    if "y_mask" in batch.keys():
                        if torch.sum(batch["y_mask"]).item() == 0:
                            if current_epoch == 0:
                                logger.warning(f"Exp_Main.vali(): Batch No.{i} out of [0~{len(vali_loader) - 1}] has no evaluation point (inferred from y_mask), thus skipping")
                            continue
                    if not self.configs.use_multi_gpu:
                        batch = {k: v.to(self.device) for k, v in batch.items()}

                    # some model's forward function return different values in "train", "val", "test", they can use `exp_stage` as argument to distinguish
                    outputs: dict[str, Tensor] = model_train(
                        **batch,
                        exp_stage="val",
                        train_stage=train_stage,
                        current_epoch=current_epoch
                    )

                    loss: Tensor = criterion(
                        **outputs,
                        exp_stage="val",
                        train_stage=train_stage,
                        current_epoch=current_epoch,
                        model=model_train
                    )["loss"]
                    total_loss.append(loss.item())

                    if accelerator.is_main_process:
                        # update only in main process
                        it.update()
                        it.set_postfix(loss=f"{loss.item():.2e}")
        total_loss = np.average(total_loss)
        model_train.train()
        return total_loss

    def _qsdiff_sampled_vali(
        self,
        model_train: Module,
        vali_loader: DataLoader,
        current_epoch: int,
        train_stage: int,
    ) -> float:
        """Evaluate QSDiff with its actual sampled forecast mean.

        The regular validation loss is a single randomly noised x0 estimate.  It
        is useful for optimization diagnostics, but it is not the quantity used
        at test time.  This fixed-seed validation therefore selects checkpoints
        using a small DDIM ensemble and a deterministic validation order.
        """
        raw_model = accelerator.unwrap_model(model_train)
        old_num_samples = raw_model.num_samples
        old_sampling_steps = raw_model.sampling_steps
        raw_model.num_samples = max(1, int(self.configs.qsdiff_sample_val_num_samples))
        raw_model.sampling_steps = max(1, int(self.configs.qsdiff_sample_val_sampling_steps))
        max_batches = max(0, int(self.configs.qsdiff_sample_val_max_batches))
        seed = int(self.configs.qsdiff_sample_val_seed)
        was_training = model_train.training
        squared_error = 0.0
        observed_count = 0.0

        cuda_devices = [self.device.index] if self.device.type == "cuda" and self.device.index is not None else []
        try:
            model_train.eval()
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                with torch.no_grad():
                    total = min(len(vali_loader), max_batches) if max_batches else len(vali_loader)
                    with tqdm(total=total, leave=False, desc="Sampled validation") as it:
                        for i, batch in enumerate(vali_loader):
                            if max_batches and i >= max_batches:
                                break
                            if "y_mask" in batch and torch.sum(batch["y_mask"]).item() == 0:
                                continue
                            batch = {key: value.to(self.device) for key, value in batch.items()}
                            outputs: dict[str, Tensor] = model_train(
                                **batch,
                                exp_stage="test",
                                train_stage=train_stage,
                                current_epoch=current_epoch,
                            )
                            mask = outputs["mask"].float()
                            error = (outputs["pred"] - outputs["true"]).square() * mask
                            squared_error += error.sum().item()
                            observed_count += mask.sum().item()
                            if accelerator.is_main_process:
                                it.update()
                                it.set_postfix(mse=f"{squared_error / max(observed_count, 1.0):.2e}")
        finally:
            raw_model.num_samples = old_num_samples
            raw_model.sampling_steps = old_sampling_steps
            model_train.train(was_training)

        if observed_count == 0:
            raise RuntimeError("Sampled QSDiff validation found no observed target values.")
        return squared_error / observed_count

    def train(self) -> None:
        logger.info('>>>>>>> training start <<<<<<<')
        path = Path(self.configs.checkpoints) / self.configs.dataset_name / self.configs.dataset_id / self.configs.model_name / self.configs.model_id / f"{self.configs.seq_len}_{self.configs.pred_len}" / self.configs.subfolder_train / f"iter{self.configs.itr_i}"
        if (self.configs.wandb and accelerator.is_main_process) or self.configs.sweep:
            import wandb

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        sampled_val_interval = int(getattr(self.configs, "qsdiff_sample_val_interval", 0))
        sampled_val_enabled = (
            self.configs.model_name == "QSDiff"
            and sampled_val_interval > 0
            and not self.configs.use_multi_gpu
        )
        sampled_val_loader: DataLoader | None = None
        if sampled_val_enabled:
            # Do not inherit a shuffled validation sampler: each checkpoint must
            # see the same samples in the same order under the same DDIM noise.
            sampled_val_loader = DataLoader(
                vali_data,
                batch_size=self.configs.batch_size,
                shuffle=False,
                num_workers=self.configs.num_workers,
                drop_last=False,
                collate_fn=vali_loader.collate_fn,
                pin_memory=vali_loader.pin_memory,
            )
            logger.info(
                "QSDiff sampled validation enabled: interval=%d, samples=%d, steps=%d, max_batches=%d, seed=%d",
                sampled_val_interval,
                self.configs.qsdiff_sample_val_num_samples,
                self.configs.qsdiff_sample_val_sampling_steps,
                self.configs.qsdiff_sample_val_max_batches,
                self.configs.qsdiff_sample_val_seed,
            )
        elif self.configs.model_name == "QSDiff" and sampled_val_interval > 0:
            logger.warning("Sampled QSDiff validation is disabled under multi-GPU training.")

        # model initialized after dataset to obtain possible dynamic information from dataset (e.g., seq_len_max_irr)
        model_train = self._build_model()

        model_optim = self._select_optimizer(model_train)
        lr_scheduler = self._select_lr_scheduler(model_optim)
        criterion = self._select_criterion()

        if not self.configs.sweep:
            train_loader, vali_loader, model_train, model_optim = accelerator.prepare(
                train_loader, vali_loader, model_train, model_optim
            )
            # Optimizer state saving is disabled below; registering it here keeps
            # old GPU parameter references across repeated itr runs.
        else:
            model_train, model_optim = accelerator.prepare(
                model_train, model_optim
            )

        # Save initial states
        initial_optimizer_state = model_optim.state_dict()
        initial_scheduler_state = lr_scheduler.state_dict()

        if not self.configs.use_multi_gpu:
            model_train = model_train.to(self.device)

        # QSDiff uses its EMA weights for validation and saved checkpoints when enabled.
        # Keeping this opt-in avoids changing the behavior of existing models and runs.
        ema_decay = float(getattr(self.configs, "qsdiff_ema_decay", 0.0))
        ema_enabled = self.configs.model_name == "QSDiff" and 0.0 < ema_decay < 1.0
        ema_state: OrderedDict[str, Tensor] | None = None
        if ema_enabled:
            raw_model = accelerator.unwrap_model(model_train)
            ema_state = OrderedDict(
                (name, value.detach().clone()) for name, value in raw_model.state_dict().items()
            )
            logger.info(f"QSDiff EMA enabled with decay={ema_decay:.6f}")

        if_nan_loss = False # break nested loop without using for...else...
        for train_stage in range(1, self.configs.n_train_stages + 1):
            early_stopping = EarlyStopping(patience=self.configs.patience, verbose=True)
            logger.info(f"Train stage {train_stage}/{self.configs.n_train_stages} starts.")
            for epoch in tqdm(range(self.configs.train_epochs), desc="Epochs"):
                train_loss = []
                model_train.train()
                with tqdm(total=len(train_loader), leave=False, desc="Training") as it:
                    batch: dict[str, Tensor] # type hints
                    for i, batch in enumerate(train_loader):
                        # warn if the size does not match
                        if batch[next(iter(batch))].shape[0] != self.configs.batch_size and epoch == 0:
                            logger.warning(f"Exp_Main.train(): Batch No.{i} out of [0~{len(train_loader) - 1}] has an actual batch_size={batch[next(iter(batch))].shape[0]}, which is not the same as --batch_size {self.configs.batch_size}")
                        if "y_mask" in batch.keys():
                            if torch.sum(batch["y_mask"]).item() == 0:
                                if epoch == 0:
                                    logger.warning(f"Exp_Main.train(): Batch No.{i} out of [0~{len(train_loader) - 1}] has no evaluation point (inferred from y_mask), thus skipping")
                                continue
                        model_optim.zero_grad()
                        if not self.configs.use_multi_gpu:
                            batch = {k: v.to(self.device) for k, v in batch.items()}

                        outputs: dict[str, Tensor] = model_train(
                            **batch,
                            exp_stage="train",
                            train_stage=train_stage,
                            current_epoch=epoch
                        )

                        # check model's outputs only in the first iteration
                        if i == 0 and epoch == 0:
                            self._check_model_outputs(batch, outputs)
                        
                        loss: Tensor = criterion(
                            **outputs,
                            exp_stage="train",
                            train_stage=train_stage,
                            current_epoch=epoch,
                            model=model_train
                        )["loss"]

                        # check loss
                        if torch.any(torch.isnan(loss)):
                            logger.exception("Loss is nan! Training interruptted!")
                            for key, value in outputs.items():
                                if key == "loss":
                                    continue
                                elif type(value).__name__ == "Tensor" and torch.any(torch.isnan(value)):
                                    logger.error(f"Nan value found in model's output tensor '{key}' of shape {value.shape}: {value}")
                            logger.warning("Hint: possible cause for nan loss: 1. large learning rate; 2. sqrt(0); 3. ReLU->LeakyReLU")
                            if_nan_loss = True
                            break

                        train_loss.append(loss.item())

                        if accelerator.is_main_process:
                            # update progress bar only in main process
                            it.update()
                            it.set_postfix(loss=f"{loss.item():.2e}")

                        if self.configs.sweep:
                            loss.backward(retain_graph=self.configs.retain_graph)
                        else:
                            accelerator.backward(loss, retain_graph=self.configs.retain_graph)
                        grad_clip_norm = float(getattr(self.configs, "grad_clip_norm", 0.0))
                        if grad_clip_norm > 0:
                            accelerator.clip_grad_norm_(model_train.parameters(), max_norm=grad_clip_norm)
                        elif self.configs.task_name == "classification":
                            clip_grad_norm_(model_train.parameters(), max_norm=4.0)
                        model_optim.step()
                        if self.configs.model_name == "QSDiff":
                            raw_model = accelerator.unwrap_model(model_train)
                            update_teacher = getattr(raw_model, "update_consistency_teacher", None)
                            if callable(update_teacher):
                                update_teacher()
                        if ema_state is not None:
                            raw_model = accelerator.unwrap_model(model_train)
                            with torch.no_grad():
                                for name, value in raw_model.state_dict().items():
                                    ema_value = ema_state[name]
                                    if torch.is_floating_point(value):
                                        ema_value.mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
                                    else:
                                        ema_value.copy_(value.detach())

                if if_nan_loss:
                    accelerator.set_trigger()
                    if accelerator.check_trigger():
                        accelerator.wait_for_everyone()
                        break
                # DEBUG: state saving is disabled to minimize disk write time
                # save the state of optimizer
                # if not self.configs.sweep:
                #     accelerator.save_state(safe_serialization=False)

                # validation
                if epoch % self.configs.val_interval == 0:
                    sampled_val_loss: float | None = None
                    select_with_sampled_val = (
                        sampled_val_enabled
                        and epoch % sampled_val_interval == 0
                        and sampled_val_loader is not None
                    )
                    if ema_state is None:
                        vali_loss = self.vali(
                            model_train=model_train,
                            vali_loader=vali_loader,
                            criterion=criterion,
                            current_epoch=epoch,
                            train_stage=train_stage
                        )
                        if select_with_sampled_val:
                            sampled_val_loss = self._qsdiff_sampled_vali(
                                model_train=model_train,
                                vali_loader=sampled_val_loader,
                                current_epoch=epoch,
                                train_stage=train_stage,
                            )
                            logger.info("QSDiff sampled validation MSE: %.8f", sampled_val_loss)
                            early_stopping(sampled_val_loss, model_train, path)
                        elif not sampled_val_enabled:
                            early_stopping(vali_loss, model_train, path)
                    else:
                        raw_model = accelerator.unwrap_model(model_train)
                        raw_state = OrderedDict(
                            (name, value.detach().clone()) for name, value in raw_model.state_dict().items()
                        )
                        raw_model.load_state_dict(ema_state, strict=True)
                        try:
                            vali_loss = self.vali(
                                model_train=model_train,
                                vali_loader=vali_loader,
                                criterion=criterion,
                                current_epoch=epoch,
                                train_stage=train_stage
                            )
                            if select_with_sampled_val:
                                sampled_val_loss = self._qsdiff_sampled_vali(
                                    model_train=model_train,
                                    vali_loader=sampled_val_loader,
                                    current_epoch=epoch,
                                    train_stage=train_stage,
                                )
                                logger.info("QSDiff sampled validation MSE: %.8f", sampled_val_loss)
                                early_stopping(sampled_val_loss, model_train, path)
                            elif not sampled_val_enabled:
                                early_stopping(vali_loss, model_train, path)
                        finally:
                            raw_model.load_state_dict(raw_state, strict=True)
                    if (self.configs.wandb and accelerator.is_main_process) or self.configs.sweep:
                        validation_log = {
                            "loss_train": np.mean(train_loss),
                            "loss_val": vali_loss,
                        }
                        if sampled_val_loss is not None:
                            validation_log["loss_val_sampled_mse"] = sampled_val_loss
                        if early_stopping.best_score is not None:
                            validation_log["loss_val_best"] = -early_stopping.best_score
                        wandb.log(validation_log)
                    if early_stopping.early_stop:
                        logger.info("Early stopping")
                        accelerator.set_trigger()
                elif (self.configs.wandb and accelerator.is_main_process) or self.configs.sweep:
                    wandb.log({
                        "loss_train": np.mean(train_loss),
                    })

                lr_scheduler.step()
                logger.debug(f'Updating learning rate to {lr_scheduler.get_last_lr()[0]:.6e}')
                if accelerator.check_trigger():
                    accelerator.wait_for_everyone()
                    break

            # Reset optimizer, scheduler
            model_optim.load_state_dict(initial_optimizer_state)
            lr_scheduler.load_state_dict(initial_scheduler_state)


    def test(self) -> None:
        logger.info('>>>>>>> testing start <<<<<<<')

        # convert task_name to task_key for storage folder naming
        task_key_mapping = {
            "short_term_forecast": "forecasting",
            "long_term_forecast": "forecasting",
        }
        if self.configs.test_flop:
            self.configs.batch_size = 1
            logger.debug("batch_size automatically overwritten to 1.")
            test_params_flop(
                model=self._build_model().to(self.device), 
                x_shape=(self.configs.seq_len,self.configs.enc_in),
                model_id=self.configs.model_id,
                task_key=task_key_mapping[self.configs.task_name] if self.configs.task_name in task_key_mapping.keys() else self.configs.task_name
            )
            exit(0)

        if self.configs.test_train_time:
            self.configs.batch_size = 32
            logger.debug("batch_size automatically overwritten to 32.")
            train_data, train_loader = self._get_data(flag='train')
            test_train_time(
                model=self._build_model().to(self.device), 
                dataloader=train_loader,
                criterion=self._select_criterion(),
                model_id=self.configs.model_id,
                dataset_name=self.configs.dataset_name,
                gpu=self.configs.gpu_id,
                seq_len=self.configs.seq_len,
                pred_len=self.configs.pred_len,
                task_key=task_key_mapping[self.configs.task_name] if self.configs.task_name in task_key_mapping.keys() else self.configs.task_name,
                retain_graph=self.configs.retain_graph
            )
            exit(0)

        if self.configs.test_gpu_memory:
            self.configs.batch_size = 32
            logger.debug("batch_size automatically overwritten to 32.")
            train_data, train_loader = self._get_data(flag='train')
            batch = next(iter(train_loader))
            batch = {k: v.to(self.device) for k, v in batch.items()}
            model = self._build_model().to(self.device).train()
            test_gpu_memory(
                model=model,
                batch=batch,
                model_id=self.configs.model_id,
                dataset_name=self.configs.dataset_name,
                gpu=self.configs.gpu_id,
                seq_len=self.configs.seq_len,
                pred_len=self.configs.pred_len,
                task_key=task_key_mapping[self.configs.task_name] if self.configs.task_name in task_key_mapping.keys() else self.configs.task_name
            )
            exit(0)

        if self.configs.test_dataset_statistics:
            _, data_loader = self._get_data(flag='test_all')
            n_observations_raw = 0
            n_observations_all = 0
            logger.info(f"""Testing Dataset '{self.configs.dataset_name}':
            - seq_len={self.configs.seq_len}
            - pred_len={self.configs.pred_len}
            - batch_size={self.configs.batch_size}
            - collate_fn='{self.configs.collate_fn}'""")
            logger.warning("Make sure seq_len and pred_len are correctly set.")
            for batch in tqdm(data_loader):
                n_observations_raw += np.sum(batch["x_mask"].detach().cpu().numpy())
                n_observations_raw += np.sum(batch["y_mask"].detach().cpu().numpy())
                n_observations_all += np.sum(np.ones_like(batch["x_mask"].detach().cpu().numpy()))
                n_observations_all += np.sum(np.ones_like(batch["y_mask"].detach().cpu().numpy()))

            logger.info(f"No. observations (raw): {n_observations_raw}")
            logger.info(f"No. observations (all): {n_observations_all}")
            exit(0)

        # test_all will test the model on all available sets (train, val, test). Needs to be supported by the dataset
        flag = "test_all" if self.configs.test_all else self.configs.test_split
        if flag == "val" and getattr(self.configs, "seq_len_max_irr", None) is None:
            # Irregular validation loaders reuse padding lengths discovered by the
            # training split. Test-only validation must initialize them first.
            self._get_data(flag="train")
        test_data, test_loader = self._get_data(flag=flag)

        # find model checkpoint path
        checkpoint_location: Path = None
        actual_itrs = 1
        if self.configs.checkpoints_test is None:
            # by default, if checkpoints_test is not given, it tries to load the latest corresponding checkpoint
            checkpoint_location = Path(self.configs.checkpoints) / self.configs.dataset_name / self.configs.dataset_id / self.configs.model_name / self.configs.model_id / f"{self.configs.seq_len}_{self.configs.pred_len}"
            try:
                # first, find the latest one based on timestamp in name
                child_folders = [(entry.name, entry) for entry in checkpoint_location.iterdir() if entry.is_dir()]
                if len(child_folders) == 0:
                    logger.exception(f"No folder under '{checkpoint_location}' matches the model_id '{self.configs.model_id}'.", stack_info=True)
                    logger.exception(f"Tips: Failed to infer the latest checkpoint folder. Please manually provide the checkpoints_test argument pointing to the folder of checkpoint file")
                    exit(1)
                latest_folder: str = sorted(child_folders, key=lambda item: datetime.datetime.strptime(item[0], "%Y_%m%d_%H%M"))[-1][1].name
                checkpoint_location = checkpoint_location / latest_folder
                self.configs.subfolder_train = latest_folder
                # then find the latest iter
                actual_itrs = len([entry.name for entry in checkpoint_location.iterdir() if entry.is_dir()])
            except Exception as e:
                logger.exception(f"{e}", stack_info=True)
                logger.exception(f"Tips: Failed to infer the latest checkpoint folder. Please manually provide the checkpoints_test argument pointing to the folder of checkpoint file")
                exit(1)

        # test on all iters' checkpoints
        for itr_i in range(actual_itrs):
            if self.configs.checkpoints_test is None:
                checkpoint_location_itr = checkpoint_location / f"iter{itr_i}"
            else:
                checkpoint_location_itr = Path(self.configs.checkpoints_test)

            model_test = self._build_model().eval()
            # load model checkpoint if load_checkpoints_test
            if self.configs.load_checkpoints_test:
                checkpoint_file = checkpoint_location_itr / "pytorch_model.bin"
                if checkpoint_file.exists():
                    try: 
                        # model state dict cannot be modified after accelerator.prepare
                        original_state_dict = self._get_state_dict(checkpoint_file)
                        load_result = model_test.load_state_dict(original_state_dict, strict=False)
                        del original_state_dict
                        gc.collect()
                        if load_result.missing_keys or load_result.unexpected_keys:
                            logger.warning(f"""The following keys in checkpoint are not correctly loaded:
                            {load_result.missing_keys=}
                            {load_result.unexpected_keys=}

                            Results may be incorrect!
                            """)
                    except Exception as e:
                        logger.exception(f"{e}", stack_info=True)
                        logger.exception(f"Failed to load checkpoint file at {checkpoint_file}. Skipping it...")
                        del model_test
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                else:
                    try:
                        # when weights are large (>10GB), they will be saved in several files
                        load_checkpoint_in_model(model_test, checkpoint_location_itr)
                    except Exception as e:
                        logger.exception(f"{e}", stack_info=True)
                        logger.exception(f"Failed to load checkpoint file at {checkpoint_file}. Skipping it...")
                        del model_test
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

            test_loader_iter = test_loader
            if self.configs.use_multi_gpu:
                model_test, test_loader_iter = accelerator.prepare(model_test, test_loader_iter)
            else:
                model_test = model_test.to(self.device)

            # create folder for test results
            # Include split and microseconds so validation/test results are
            # auditable and concurrent calibration jobs cannot overwrite one
            # another under the same checkpoint.
            subfolder_eval = (
                f'eval_{flag}_'
                f'{datetime.datetime.now().strftime("%Y_%m%d_%H%M%S_%f")}'
            )
            folder_path = checkpoint_location_itr / subfolder_eval
            folder_path.mkdir(exist_ok=True)
            logger.info(f"Testing results will be saved under {folder_path}")

            # dictionary holding input and output data
            array_dict: dict[str, list[np.ndarray] | np.ndarray] = {}
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
                input_tensor_names = ["x", "y", "x_mark", "y_mark", "x_mask", "y_mask", "sample_ID"]
                probabilistic_models = {
                    "QSDiff", "CSDI", "TMDM", "LLapDiff", "DiffusionTS", "TimeGrad",
                    "Latent_ODE", "NeuralFlows", "GNeuralFlow", "CRU", "ProFITi",
                    "SeparableFlows",
                }
                output_tensor_names = ["pred", "pred_samples"] if self.configs.model_name in probabilistic_models else ["pred"]
            elif self.configs.task_name in ["classification"]:
                input_tensor_names = ["x", "y_class", "x_mask", "y_mask", "sample_ID"]
                output_tensor_names = ["pred_class"]
            else:
                raise NotImplementedError()

            for tensor_name in input_tensor_names + output_tensor_names:
                array_dict[tensor_name] = []

            query_keep_ratio = min(max(float(self.configs.test_query_keep_ratio), 0.0), 1.0)
            query_subset_generator = torch.Generator(device="cpu")
            query_subset_generator.manual_seed(int(self.configs.test_query_subset_seed))
            if query_keep_ratio < 1.0:
                logger.info(
                    "Query-set robustness evaluation: retaining %.0f%% of valid future queries (seed=%d)",
                    100.0 * query_keep_ratio,
                    self.configs.test_query_subset_seed,
                )

            # try to recover from cache saved by save_cache_arrays, if any
            cache_folder = checkpoint_location_itr / "cache"
            n_cache_batches = 0
            if cache_folder.exists():
                logger.warning(f"Trying to recover the testing process using cache files in {cache_folder}")
                for tensor_name in output_tensor_names:
                    cache_file_path = cache_folder / f"output_{tensor_name}.npy"
                    if cache_file_path.exists():
                        cache_array = np.load(cache_file_path)
                        n_cache_samples = cache_array.shape[0]
                        # overwrite init content with cache
                        array_dict[tensor_name] = [cache_array[i:i + self.configs.batch_size] for i in range(0, n_cache_samples, self.configs.batch_size)] # ndarray -> list[ndarray]
                    else:
                        logger.error(f"Cache file for {tensor_name} not found. You may encounter unexpected error if proceed!")
                n_cache_batches = len(array_dict[tensor_name])

            
            with torch.no_grad():
                batch: dict[str, Tensor] # type hints
                for i, batch in tqdm(enumerate(test_loader_iter), total=len(test_loader_iter), leave=False, desc="Testing"):
                    if query_keep_ratio < 1.0 and "y_mask" in batch:
                        subset_y_mask = torch.zeros_like(batch["y_mask"])
                        for batch_idx in range(batch["y_mask"].shape[0]):
                            valid_idx = torch.nonzero(
                                batch["y_mask"][batch_idx].reshape(-1) > 0,
                                as_tuple=False,
                            ).flatten()
                            if valid_idx.numel() == 0:
                                continue
                            keep_count = max(1, int(round(valid_idx.numel() * query_keep_ratio)))
                            keep_idx = valid_idx[
                                torch.randperm(valid_idx.numel(), generator=query_subset_generator)[:keep_count]
                            ]
                            subset_y_mask[batch_idx].reshape(-1)[keep_idx] = batch["y_mask"][batch_idx].reshape(-1)[keep_idx]
                        batch["y_mask"] = subset_y_mask
                    if n_cache_batches > 0:
                        # recovering from cache. append input batch and skip it, such that model don't have to inference again.
                        batch_all: list[dict] = accelerator.gather_for_metrics([batch])
                        batch_all: dict = self._merge_gathered_dicts(batch_all)
                        for tensor_name in input_tensor_names:
                            if tensor_name in batch_all.keys():
                                array_dict[tensor_name].append(batch_all[tensor_name].detach().cpu().numpy())
                        n_cache_batches -= 1
                        continue
                    # warn if the size does not match
                    if batch[next(iter(batch))].shape[0] != self.configs.batch_size:
                        logger.warning(f"Exp_Main.test(): Batch No.{i} out of [0~{len(test_loader_iter) - 1}] has an actual batch_size={batch[next(iter(batch))].shape[0]}, which is not the same as --batch_size {self.configs.batch_size}")
                        # continue
                    if not self.configs.use_multi_gpu:
                        batch = {k: v.to(self.device) for k, v in batch.items()}

                    outputs: dict[str, Tensor] = model_test(
                        **batch,
                        exp_stage="test"
                    )

                    # check model's outputs only in the first iteration
                    if i == 0 and itr_i == 0:
                        self._check_model_outputs(batch, outputs)

                    batch_all: list[dict] = accelerator.gather_for_metrics([batch])
                    batch_all: dict = self._merge_gathered_dicts(batch_all)
                    outputs_all: list[dict] = accelerator.gather_for_metrics([outputs])
                    outputs_all: dict = self._merge_gathered_dicts(outputs_all)

                    for tensor_name in input_tensor_names:
                        if tensor_name in batch_all.keys():
                            array_dict[tensor_name].append(batch_all[tensor_name].detach().cpu().numpy())
                    for tensor_name in output_tensor_names:
                        if tensor_name in outputs_all.keys():
                            array_dict[tensor_name].append(outputs_all[tensor_name].detach().cpu().numpy())

                    if self.configs.save_cache_arrays:
                        # save intermediate model outputs, to enable recovery from interruption
                        cache_folder.mkdir(exist_ok=True)
                        for tensor_name in output_tensor_names:
                            if len(array_dict[tensor_name]) > 0:
                                np.save(
                                    cache_folder / f"output_{tensor_name}.npy",
                                    np.concatenate(array_dict[tensor_name], axis=0)
                                )
                        logger.debug(f"Model outputs saved into cache folder {cache_folder}")

            for tensor_name in input_tensor_names + output_tensor_names:
                if len(array_dict[tensor_name]) > 0:
                    array_dict[tensor_name] = np.concatenate(array_dict[tensor_name], axis=0)
                else:
                    array_dict[tensor_name] = None # reset to default value for metric calculation

            metrics = None
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
                metric_inputs = dict(array_dict)
                if (
                    self.configs.model_name
                    in {
                        "QSDiff", "CSDI", "TMDM", "LLapDiff", "DiffusionTS", "TimeGrad",
                        "Latent_ODE", "NeuralFlows", "GNeuralFlow", "CRU", "ProFITi",
                        "SeparableFlows",
                    }
                    and self.configs.qsdiff_crps_scale == "original"
                    and array_dict["pred_samples"] is not None
                ):
                    inverse_transform = getattr(test_data, "inverse_transform_values", None)
                    if inverse_transform is not None:
                        metric_inputs["crps_pred_samples"] = inverse_transform(array_dict["pred_samples"])
                        metric_inputs["crps_y"] = inverse_transform(array_dict["y"])
                metrics = metric(**metric_inputs)
                eval_temperatures = str(getattr(self.configs, "qsdiff_eval_temperatures", "")).strip()
                if self.configs.model_name == "QSDiff" and eval_temperatures and array_dict["pred_samples"] is not None:
                    base_samples = array_dict["pred_samples"]
                    sample_mean = base_samples.mean(axis=1, keepdims=True)
                    temperature_metrics = {}
                    for value in eval_temperatures.split(","):
                        temperature = max(float(value.strip()), 0.0)
                        calibrated_samples = sample_mean + temperature * (base_samples - sample_mean)
                        calibrated_inputs = dict(array_dict)
                        # Temperature calibration changes only ensemble spread.
                        # Keep the original point forecast array byte-for-byte
                        # instead of recomputing a NumPy ensemble mean.
                        calibrated_inputs["pred"] = array_dict["pred"]
                        calibrated_inputs["pred_samples"] = calibrated_samples
                        if self.configs.qsdiff_crps_scale == "original":
                            inverse_transform = getattr(test_data, "inverse_transform_values", None)
                            if inverse_transform is not None:
                                calibrated_inputs["crps_pred_samples"] = inverse_transform(calibrated_samples)
                                calibrated_inputs["crps_y"] = inverse_transform(array_dict["y"])
                        calibrated_result = metric(**calibrated_inputs)
                        temperature_key = format(temperature, ".12g")
                        if temperature_key in temperature_metrics:
                            raise ValueError(
                                "Duplicate qsdiff_eval_temperatures value after "
                                f"normalization: {temperature_key}"
                            )
                        temperature_metrics[temperature_key] = {
                            key: float(metric_value)
                            for key, metric_value in calibrated_result.items()
                            if np.isscalar(metric_value)
                        }
                    metrics["temperature_metrics"] = temperature_metrics
                if (self.configs.wandb and accelerator.is_main_process and self.configs.is_training) or self.configs.sweep:
                    import wandb
                    wandb.log({
                        "loss_test": np.mean(metrics["MSE"]),
                    })
            elif self.configs.task_name in ["classification"]:
                metrics = metric_classification(
                    **array_dict,
                    n_classes=self.configs.n_classes
                )
                if (self.configs.wandb and accelerator.is_main_process and self.configs.is_training) or self.configs.sweep:
                    import wandb
                    wandb.log({
                        "loss_test": np.mean(metrics["Accuracy"]),
                    })

            
            if metrics is not None:
                metrics["evaluation_split"] = flag
                if self.configs.model_name == "QSDiff":
                    metrics["qsdiff_sample_temperature"] = float(
                        getattr(self.configs, "qsdiff_sample_temperature", 1.0)
                    )
                    metrics["qsdiff_num_samples"] = int(
                        getattr(self.configs, "qsdiff_num_samples", 25)
                    )
                    metrics["qsdiff_sampling_steps"] = int(
                        getattr(self.configs, "qsdiff_sampling_steps", 20)
                    )
                    metrics["qsdiff_prior_sample_residual_scale"] = float(
                        getattr(
                            self.configs,
                            "qsdiff_prior_sample_residual_scale",
                            1.0,
                        )
                    )
                    metrics["evaluation_seed"] = int(
                        getattr(self.configs, "seed_base", 2024)
                    )
                # convert to float before saving to json
                for key, value in metrics.items():
                    if isinstance(value, np.float32):
                        metrics[key] = float(value)
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, np.float32):
                                metrics[key] = [float(v) for v in value]
                                break
                logger.info("Metrics:\n%s", json.dumps(metrics, indent=4)) # log result in a readable way
                with open(folder_path / "metric.json", "w") as f:
                    json.dump(metrics, f, indent=2)

            if self.configs.save_arrays:
                for tensor_name in input_tensor_names:
                    if array_dict[tensor_name] is not None:
                        np.save(folder_path / f"input_{tensor_name}.npy", array_dict[tensor_name])
                for tensor_name in output_tensor_names:
                    if array_dict[tensor_name] is not None:
                        np.save(folder_path / f"output_{tensor_name}.npy", array_dict[tensor_name])

            del model_test, array_dict
            if "batch" in locals():
                del batch
            if "outputs" in locals():
                del outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception as cleanup_error:
                    logger.debug(f"torch.cuda.ipc_collect() skipped: {cleanup_error}")
