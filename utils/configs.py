# Code from: https://github.com/Ladbaby/PyOmniTS
import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from utils.ExpConfigs import ExpConfigs

'''
variable `configs` are ONLY determined by argparse, not .yaml files.

.yaml files are automatically saved and maintained as references:

- configs.yaml in training directory:

    Exact configs used in that training.

- *.yaml in "configs" folder

    Reference default configs for each model and dataset combination. Only automatically updated if new argument are added in argparse.

Note: Models and datasets only access the arguments they need.
'''

def get_configs(args=None) -> ExpConfigs:
    parser = argparse.ArgumentParser(description='PyOmniTS')

    # basic config
    parser.add_argument('--checkpoints', type=str, default='storage/results/', help='where to save model checkpoints in training. By default, results are organized under storage/results/DATASET_NAME/MODEL_NAME/MODEL_ID/SEQ_LEN_PRED_LEN/Y_md_HM/iter0')
    parser.add_argument('--is_training', type=int, default=1, help='training or testing')
    parser.add_argument('--model_id', type=str, default='GRU_D', help='By default, model_id is the same as model_name. However, if the model has multiple variants, then model_id will additionally contain more detailed information (backbone name, model size, etc). Unlike model_name, model_id only affects the storage location of experiment results (refer to --checkpoints) and yaml configs, so you can change it to whatever you like (e.g., GRU_D_d_model_512).')
    parser.add_argument('--model_name', type=str, default='GRU_D', help='model name. PyOmniTS will try to construct the model from class Model in models/MODEL_NAME.py')
    parser.add_argument('--task_name', type=str, choices=["long_term_forecast", "short_term_forecast", "imputation", "classification", "anomaly_detection", "representation_learning", "causal_discovery"], default='short_term_forecast', help='task name')

    # dataset & data loader
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment in regular time series forecasting datasets.")
    parser.add_argument('--collate_fn', type=str, default=None, help='Name of the function as a custom collate_fn for dataloader. By default, datasets without collate_fn defined in data/data_provider/datasets/DATASET_NAME will use the default collate_fn of PyTorch. Refer to data/data_provider/data_factory.py for implementation detail.')
    parser.add_argument('--dataset_file_name', type=str, default=None, help='Some datasets contains more than one files, then this argument may be ignored.')
    parser.add_argument('--dataset_id', type=str, default='ETTm1', help='By default, dataset_id is the same as dataset_name. However, if the dataset has multiple subsets, then dataset_id will additionally contain dataset_subset_name. Unlike dataset_name, dataset_id only affects the storage location of experiment results (refer to --checkpoints), so you can change it to whatever you like (e.g., ETTm1_batch_size_32).')
    parser.add_argument('--dataset_name', type=str, default='ETTm1', help='dataset name. PyOmniTS will try to construct the dataset from class Data in data/data_provider/datasets/DATASET_NAME.py')
    parser.add_argument('--dataset_root_path', type=str, default='storage/datasets/ETT-small/', help='root path of the data file')
    parser.add_argument('--dataset_subset_name', type=str, default=None, help='Some datasets may be composed of multiple subsets (e.g., UCR128).')
    parser.add_argument('--embed', type=str, choices=["timeF", "fixed", "learned"], default='timeF', help='time features encoding. Only used in some regular time series datasets.')
    parser.add_argument('--features', type=str, choices=['M', 'S', "MS"], default='M', help='M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate. WARNING: not thoroughly tested yet.')
    parser.add_argument('--freq', type=str, choices=['s', 't', 'h', 'd', 'b', 'w', 'm', 'others'], default='h', help='freq for time features encoding in regular time series forecasting datasets, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--missing_rate', type=float, default=0., help="Manually mask out some observations. WARNING: not thoroughly tested yet.")
    parser.add_argument('--target_variable_index', type=int, default=0, help='target variable index in datasets. Should not be used together with target_variable_name. WARNING: not thoroughly tested yet.')
    parser.add_argument('--target_variable_name', type=str, default="OT", help='target variable name in regular time series forecasting datasets. Originally named as --target.')
    parser.add_argument('--train_val_loader_drop_last', type=int, default=0, help="By default, train and val loader will not drop the last batch if the number of samples is not sufficient.")
    parser.add_argument('--train_val_loader_shuffle', type=int, default=1, help="By default, train and val loader are shuffled.")

    # forecasting task
    parser.add_argument('--label_len', type=int, default=0, help='start token length. In other words, the length prepanded to the input of decoder. Only used in some regular time series forecasting models.')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length. Should be zero for tasks other than forecasting')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')

    # classification task
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes')

    # GPU
    parser.add_argument('--allow_tf32', type=int, default=1, help='Whether PyTorch is allowed to use the TensorFloat32 (TF32) tensor cores, available on NVIDIA GPUs since Ampere. If set to 1, model accuracy will be lower, but speed can be faster. Refer to https://docs.pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere')
    parser.add_argument('--gpu_id', type=int, default=0, help='primary gpu id, will be overwritten when use_multi_gpu is 1. Originally named as --gpu.')
    parser.add_argument('--gpu_ids', type=str, default=None, help='string of device ids for multile gpus. Originally named as --devices. WARNING: not thoroughly tested yet. Currently, setting --use_multi_gpu 1 is enough.')
    parser.add_argument('--use_gpu', type=int, default=1, help='Whether to use gpu.')
    parser.add_argument('--use_multi_gpu', type=int, default=0, help='use multiple gpus, via huggingface accelerate library')

    # training
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--itr', type=int, default=5, help='Number of runs for each experiment setting. By default, we run every experiment setting for 5 times using 5 different random seeds (2024~2028), then calculate the average and std of these 5 metrics.')
    parser.add_argument('--seed_base', type=int, default=2024, help='Base random seed for repeated runs; iteration i uses seed_base + i.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--grad_clip_norm', type=float, default=0.0, help='Global gradient-norm clipping threshold; zero disables clipping.')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function. PyOmniTS will try to construct the loss function from class Loss in loss_fns/LOSS.py')
    parser.add_argument('--lr_scheduler', type=str, choices=["ExponentialDecayLR", "ManualMilestonesLR", "DelayedStepDecayLR", "CosineAnnealingLR", "MultiStepLR", "StepLR"], default='DelayedStepDecayLR', help='learning rate scheduler. Originally named as --lradj')
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.1, help='gamma for StepLR and MultiStepLR.')
    parser.add_argument('--n_train_stages', type=int, default=1, help="Some models have multiple training stages, like pretraining + finetuning. e.g., --n_train_stages 2 will pass train_stage=1 and train_stage=2 to model during training.")
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--pretrained_checkpoint_file_name', type=str, default=None, help="file name of pretrained model's checkpoints, including file type extension")
    parser.add_argument('--pretrained_checkpoint_root_path', type=str, default=None, help="Path to folder containing pretrained model's checkpoints")
    parser.add_argument('--retain_graph', type=int, default=0, help='whether to retain compute graph in back propagation. Used in special models like HD_TTS.')
    parser.add_argument('--sweep', type=int, default=0, help='whether to use weight & bias for hyperparameter searching')
    parser.add_argument('--skip_test', type=int, default=0, help='Skip test evaluation after training. Intended for validation-only hyperparameter pilots.')
    parser.add_argument('--train_epochs', type=int, default=300, help='train epochs')
    parser.add_argument('--val_interval', type=int, default=1, help='validation interval relative to training epochs')
    parser.add_argument('--wandb', type=int, default=0, help='whether to use weight & bias for monitoring')

    # testing
    parser.add_argument('--checkpoints_test', type=str, default=None, help='folder where model checkpoint file is saved, for testing. If not given, Exp_Main.test() in exp/exp_main.py will try to find the latest checkpoint.')
    parser.add_argument('--load_checkpoints_test', type=int, default=1, help='whether to load checkpoint during testing')
    parser.add_argument('--save_arrays', type=int, default=0, help='whether to save model input and output as .npy files during testing, for later visualization')
    parser.add_argument('--save_cache_arrays', type=int, default=0, help='whether to save model output (not input) as cache .npy files during every test iteration, such that the testing can recover from interruption. Designed for extremely slow models like diffusion models.')
    parser.add_argument('--test_all', type=int, default=0, help='By default, we test the model on test set only. Setting this option to 1 will test the model on all samples from train, val, and test sets. Some datasets may not support this option yet.')
    parser.add_argument('--test_query_keep_ratio', type=float, default=1.0, help='Fraction of valid future queries retained during testing. Used for query-set robustness evaluation.')
    parser.add_argument('--test_query_subset_seed', type=int, default=2024, help='Random seed used to construct test query subsets.')
    parser.add_argument('--test_split', type=str, choices=['val', 'test'], default='test', help='Dataset split used by test-only evaluation.')
    parser.add_argument('--test_dataset_statistics', type=int, default=0, help="Test dataset's statistics. See function Exp_Main.test() in exp/exp_main.py for implementation details.")
    parser.add_argument('--test_flop', type=int, default=0, help='Test model flops. See utils/tools for implementation details.')
    parser.add_argument('--test_gpu_memory', type=int, default=0, help="Test model's gpu memory usage. See utils/tools for implementation details.")
    parser.add_argument('--test_train_time', type=int, default=0, help="Test model's training time. See utils/tools for implementation details.")

    # model configs
    # common
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--c_out', type=int, default=7, help='output size. Usually it is the same as --enc_in')
    parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS/TimeMixer model')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers. Used to replace n_layers in some model decoders.')
    parser.add_argument('--d_model', type=int, default=512, help='size of latent embeddings')
    parser.add_argument('--d_timesteps', type=int, default=1, help='UNUSED. Size of last dimension of `x_mark`/`y_mark`. Many Regular/Spatiotemporal datasets stack time in day, day in week, etc. along the last dimension. Others default to size 1.')
    parser.add_argument('--dec_in', type=int, default=7, help='Number of variables in input time series (of decoder). Usually it is the same as --enc_in')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers. Used to replace n_layers in some model encoders.')
    parser.add_argument('--embed_type', type=int, choices=[0, 1, 2, 3, 4], default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding; Only used in some regular time series forecasting models.')
    parser.add_argument('--enc_in', type=int, default=7, help='Number of variables in input time series (of encoder). In most cases, it should be adjusted per dataset.') 
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--output_attention', type=int, default=0, help='DEPRECATED. output attention weight')
    parser.add_argument('--hidden_layers', type=int, default=1, help='Number of hidden layers. WARNING: Will be replaced by --n_layers in the future, due to duplicated meaning.')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0. Only used in some regular time series forecasting models.')
    parser.add_argument('--kernel_size', type=int, default=25, help='kernel size')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads (in attention)')
    parser.add_argument('--n_layers', type=int, default=1, help='num of layers')
    parser.add_argument('--n_patches_list', type=int, nargs="+", default=[2], help='List of number of patches. --patch_len is more commonly used, and in most cases, you wont have to set both --n_patches_list and --patch_len.')
    parser.add_argument('--node_dim', type=int, default=10, help='hidden dimension of nodes used in a few GNNs, like tPatchGNN')
    parser.add_argument('--patch_len', type=int, default=12, help='patch length. Also used as period_len in some models (SparseTSF).')
    parser.add_argument('--patch_len_list', type=int, nargs="+", default=[12, 6], help="list of patch length. Note: you don't have to set --patch_len at the same time.")
    parser.add_argument('--patch_stride', type=int, default=12, help='stride when splitting patches. Originally named as --stride.')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--revin_affine', type=int, default=0, help='RevIN-affine; True 1 False 0. Originally named as --affine.')
    parser.add_argument('--scale_factor', type=int, default=2, help='scale factor for upsample. Only used in some regular time series forecasting models.')
    parser.add_argument('--top_k', type=int, default=5, help='top k selection')
    # Adaptor class
    parser.add_argument('--ts_backbone_name', type=str, default="", help='For adaptor class, which time series backbone to use. e.g., LSTM')
    parser.add_argument('--ts_backbone_overwrite_config_list', type=str, nargs='+', default=[], help='List of argument names that we want to overwrite in time series backbones, using argument values in adaptor class. E.g., [d_model, n_layers]')
    # CRU
    parser.add_argument('--cru_bandwidth', type=int, default=3, help="Bandwidth for basis matrices A_k. b in paper")
    parser.add_argument('--cru_num_basis', type=int, default=15, help="Number of basis matrices to use in transition model for locally-linear transitions. K in paper")
    parser.add_argument('--cru_ts', type=float, default=1.0, help="Scaling factor of timestamps for numerical stability.")
    # Informer
    parser.add_argument('--informer_distil', type=int, default=1, help='whether to use distilling in encoder, using this argument means not using distilling')
    # Latent ODE
    parser.add_argument('--latent_ode_classif', type=int, default=0, help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")
    parser.add_argument('--latent_ode_gen_layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")
    parser.add_argument('--latent_ode_gru_units', type=int, default=100, help="Number of units per layer in each of GRU update networks")
    parser.add_argument('--latent_ode_linear_classif', type=int, default=0, help="If using a classifier, use a linear classifier instead of 1-layer NN")
    parser.add_argument('--latent_ode_rec_dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")
    parser.add_argument('--latent_ode_rec_layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
    parser.add_argument('--latent_ode_units', type=int, default=100, help="Number of units per layer in ODE func")
    parser.add_argument('--latent_ode_z0_encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")
    # Mamba
    parser.add_argument('--mamba_d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--mamba_expand', type=int, default=2, help='expansion factor for Mamba')
    # mTAN
    parser.add_argument('--mtan_alpha', type=float, default=100., help='In classification task, loss is calculated as recon_loss + self.alpha * ce_loss')
    parser.add_argument('--mtan_num_ref_points', type=int, default=8, help='number of reference points, originally chosen in [8, 16, 32, 64, 128]')
    # NeuralFlows
    parser.add_argument('--neuralflows_flow_layers', type=int, default=1, help='Number of flow layers')
    parser.add_argument('--neuralflows_flow_model', type=str, default='coupling', help='Type of NeuralFlows model', choices=['coupling', 'resnet', 'gru'])
    parser.add_argument('--neuralflows_latents', type=int, default=20, help='Size of the latent state')
    parser.add_argument('--neuralflows_time_hidden_dim', type=int, default=1, help='Number of time features (only for Fourier)')
    parser.add_argument('--neuralflows_time_net', type=str, default='TimeLinear', help='Name of time net', choices=['TimeFourier', 'TimeFourierBounded', 'TimeLinear', 'TimeTanh'])
    # Nonstationary Transformer
    parser.add_argument('--nonstationarytransformer_p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
    parser.add_argument('--nonstationarytransformer_p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    # PatchTST
    parser.add_argument('--patchtst_decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--patchtst_fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--patchtst_head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patchtst_padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--patchtst_subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    # PrimeNet
    parser.add_argument('--primenet_pooling', type=str, default='ave', help='[ave, att, bert]: What pooling to use to aggregate the model output sequence representation for different tasks.')
    # ReIMTS
    parser.add_argument('--reimts_pad_time_emb', type=int, default=1, help='Whether to pad temporal embedding')
    # ScaleFormer
    parser.add_argument('--scaleformer_scales', default=[2, 1], help='scales in mult-scale')
    # TimeMixer
    parser.add_argument('--timemixer_decomp_method', type=str, default='moving_avg', help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--timemixer_down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--timemixer_down_sampling_method', type=str, default="avg", help='down sampling method, only support avg, max, conv')
    parser.add_argument('--timemixer_use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    # tPatchGNN
    parser.add_argument('--tpatchgnn_te_dim', type=int, default=10, help="Number of units for time encoding")
    # CSDI
    parser.add_argument('--csdi_diffusion_steps', type=int, default=50, help='Number of CSDI diffusion steps.')
    parser.add_argument('--csdi_beta_schedule', type=str, choices=['linear', 'quad'], default='quad', help='CSDI beta schedule.')
    parser.add_argument('--csdi_num_samples', type=int, default=25, help='Number of CSDI forecast samples used at test time.')
    parser.add_argument('--csdi_channels', type=int, default=64, help='Hidden channels in the CSDI denoiser.')
    parser.add_argument('--csdi_layers', type=int, default=4, help='Number of CSDI residual attention blocks.')
    parser.add_argument('--csdi_nheads', type=int, default=4, help='Number of attention heads in CSDI.')
    parser.add_argument('--csdi_time_embedding_dim', type=int, default=128, help='Continuous-time embedding size in CSDI.')
    parser.add_argument('--csdi_feature_embedding_dim', type=int, default=16, help='Variable embedding size in CSDI.')
    parser.add_argument('--csdi_diffusion_embedding_dim', type=int, default=128, help='Diffusion-step embedding size in CSDI.')
    parser.add_argument('--csdi_beta_start', type=float, default=1e-4, help='First CSDI diffusion beta.')
    parser.add_argument('--csdi_beta_end', type=float, default=0.02, help='Last CSDI diffusion beta.')
    parser.add_argument('--csdi_sample_batch_size', type=int, default=5, help='Maximum CSDI samples generated together to limit memory use.')
    parser.add_argument('--csdi_history_bins', type=int, default=0, help='History grid size. Zero selects min(seq_len, csdi_max_history_bins).')
    parser.add_argument('--csdi_future_bins', type=int, default=0, help='Future grid size. Zero selects pred_len.')
    parser.add_argument('--csdi_max_history_bins', type=int, default=256, help='Maximum automatic CSDI history grid size.')
    # TMDM
    parser.add_argument('--tmdm_diffusion_steps', type=int, default=100, help='Number of TMDM diffusion and reverse sampling steps.')
    parser.add_argument('--tmdm_num_samples', type=int, default=25, help='Number of TMDM forecast samples used at test time.')
    parser.add_argument('--tmdm_history_bins', type=int, default=0, help='History grid size. Zero selects min(seq_len, tmdm_max_history_bins).')
    parser.add_argument('--tmdm_future_bins', type=int, default=0, help='Future grid size. Zero selects pred_len.')
    parser.add_argument('--tmdm_max_history_bins', type=int, default=256, help='Maximum automatic TMDM history grid size.')
    parser.add_argument('--tmdm_prior_loss_weight', type=float, default=1.0, help='Weight of the deterministic TMDM prior loss.')
    parser.add_argument('--tmdm_beta_start', type=float, default=1e-4, help='First TMDM diffusion beta.')
    parser.add_argument('--tmdm_beta_end', type=float, default=0.02, help='Last TMDM diffusion beta, matching the effective original YAML configuration.')
    # Diffusion-TS
    parser.add_argument('--diffusionts_source_root', type=str, default='../NsDiff-main', help='Path to the NsDiff source tree containing the official Diffusion-TS implementation.')
    parser.add_argument('--diffusionts_history_bins', type=int, default=0, help='History grid size. Zero selects min(seq_len, diffusionts_max_history_bins).')
    parser.add_argument('--diffusionts_future_bins', type=int, default=0, help='Future grid size. Zero selects pred_len.')
    parser.add_argument('--diffusionts_max_history_bins', type=int, default=256, help='Maximum automatic Diffusion-TS history grid size.')
    parser.add_argument('--diffusionts_diffusion_steps', type=int, default=50, help='Number of Diffusion-TS forward diffusion steps.')
    parser.add_argument('--diffusionts_sampling_steps', type=int, default=20, help='Number of Diffusion-TS DDIM reverse sampling steps.')
    parser.add_argument('--diffusionts_num_samples', type=int, default=25, help='Number of Diffusion-TS forecast samples used at test time.')
    parser.add_argument('--diffusionts_sample_batch_size', type=int, default=5, help='Maximum Diffusion-TS samples generated together to limit memory use.')
    parser.add_argument('--diffusionts_n_layer_enc', type=int, default=3, help='Diffusion-TS encoder layer count.')
    parser.add_argument('--diffusionts_n_layer_dec', type=int, default=6, help='Diffusion-TS decoder layer count.')
    parser.add_argument('--diffusionts_d_model', type=int, default=64, help='Diffusion-TS hidden width.')
    parser.add_argument('--diffusionts_n_heads', type=int, default=4, help='Diffusion-TS attention head count.')
    parser.add_argument('--diffusionts_mlp_hidden_times', type=int, default=4, help='Diffusion-TS feed-forward expansion ratio.')
    parser.add_argument('--diffusionts_eta', type=float, default=0.0, help='Diffusion-TS DDIM stochasticity.')
    parser.add_argument('--diffusionts_attn_dropout', type=float, default=0.0, help='Diffusion-TS attention dropout.')
    parser.add_argument('--diffusionts_kernel_size', type=int, default=1, help='Diffusion-TS convolution kernel size.')
    parser.add_argument('--diffusionts_padding_size', type=int, default=0, help='Diffusion-TS convolution padding size.')
    parser.add_argument('--diffusionts_beta_schedule', type=str, choices=['linear', 'cosine'], default='cosine', help='Diffusion-TS noise schedule.')
    parser.add_argument('--diffusionts_loss_type', type=str, choices=['l1', 'l2'], default='l2', help='Diffusion-TS reconstruction loss.')
    parser.add_argument('--diffusionts_use_langevin', type=int, default=1, help='Use the official Diffusion-TS Langevin infill correction during sampling.')
    parser.add_argument('--diffusionts_langevin_coef', type=float, default=0.1, help='Diffusion-TS Langevin history consistency coefficient.')
    parser.add_argument('--diffusionts_langevin_lr', type=float, default=0.05, help='Diffusion-TS Langevin correction learning rate.')
    # TimeGrad
    parser.add_argument('--timegrad_history_bins', type=int, default=0, help='History grid size. Zero selects min(seq_len, timegrad_max_history_bins).')
    parser.add_argument('--timegrad_future_bins', type=int, default=0, help='Future grid size. Zero selects pred_len.')
    parser.add_argument('--timegrad_max_history_bins', type=int, default=256, help='Maximum automatic TimeGrad history grid size.')
    parser.add_argument('--timegrad_diffusion_steps', type=int, default=50, help='Number of TimeGrad diffusion and reverse sampling steps.')
    parser.add_argument('--timegrad_num_samples', type=int, default=25, help='Number of TimeGrad forecast samples used at test time.')
    parser.add_argument('--timegrad_sample_batch_size', type=int, default=5, help='Maximum TimeGrad samples generated together to limit memory use.')
    parser.add_argument('--timegrad_rnn_hidden_size', type=int, default=64, help='TimeGrad recurrent hidden width.')
    parser.add_argument('--timegrad_rnn_type', type=str, choices=['LSTM', 'GRU'], default='LSTM', help='TimeGrad recurrent cell type.')
    parser.add_argument('--timegrad_num_layers', type=int, default=2, help='TimeGrad recurrent layer count.')
    parser.add_argument('--timegrad_static_dim', type=int, default=16, help='TimeGrad per-variable static embedding width.')
    parser.add_argument('--timegrad_conditioning_length', type=int, default=100, help='TimeGrad diffusion conditioning width.')
    parser.add_argument('--timegrad_residual_layers', type=int, default=8, help='TimeGrad denoiser residual layer count.')
    parser.add_argument('--timegrad_residual_channels', type=int, default=32, help='TimeGrad denoiser residual channel count.')
    parser.add_argument('--timegrad_dilation_cycle', type=int, default=2, help='TimeGrad denoiser dilation cycle length.')
    parser.add_argument('--timegrad_beta_end', type=float, default=0.1, help='Last TimeGrad diffusion beta.')
    parser.add_argument('--timegrad_beta_schedule', type=str, choices=['linear', 'quad', 'cosine'], default='linear', help='TimeGrad noise schedule.')
    parser.add_argument('--timegrad_include_history_loss', type=int, default=1, help='Train TimeGrad diffusion on observed history as well as forecast targets.')
    # LLapDiff
    parser.add_argument('--llapdiff_source_root', type=str, default='../LLapDiffusion', help='Path to the official LLapDiffusion source tree.')
    parser.add_argument('--llapdiff_max_history_len', type=int, default=336, help='Maximum history length passed to LaplaceAE; longer irregular histories are time-binned with masks preserved.')
    parser.add_argument('--llapdiff_latent_channels', type=int, default=16, help='Target set-VAE latent width.')
    parser.add_argument('--llapdiff_vae_latent_dim', type=int, default=128, help='Set-VAE token hidden width.')
    parser.add_argument('--llapdiff_vae_layers', type=int, default=3, help='Set-VAE encoder and decoder layer count.')
    parser.add_argument('--llapdiff_vae_heads', type=int, default=4, help='Set-VAE attention head count.')
    parser.add_argument('--llapdiff_vae_ff', type=int, default=256, help='Set-VAE feed-forward width.')
    parser.add_argument('--llapdiff_vae_beta', type=float, default=1e-3, help='Set-VAE KL weight.')
    parser.add_argument('--llapdiff_vae_anneal_epochs', type=int, default=25, help='Epochs used to anneal the VAE KL weight.')
    parser.add_argument('--llapdiff_vae_input_dropout', type=float, default=0.35, help='Additional observed-target dropout during VAE pretraining.')
    parser.add_argument('--llapdiff_vae_noise_std', type=float, default=0.02, help='Observed-target noise during VAE pretraining.')
    parser.add_argument('--llapdiff_vae_entity_conditioned', type=int, default=1, help='Use learned variable embeddings in the target VAE decoder.')
    parser.add_argument('--llapdiff_summary_mix_dim', type=int, default=64, help='LaplaceAE history mixer width.')
    parser.add_argument('--llapdiff_summary_tv_hidden', type=int, default=32, help='LaplaceAE dynamics-head width.')
    parser.add_argument('--llapdiff_summary_tokens', type=int, default=16, help='Number of history summary tokens.')
    parser.add_argument('--llapdiff_summary_layers', type=int, default=2, help='LaplaceAE temporal encoder layers.')
    parser.add_argument('--llapdiff_summary_time2vec_dim', type=int, default=9, help='LaplaceAE Time2Vec width.')
    parser.add_argument('--llapdiff_summary_irreg_pooling', type=str, default='repair', help='LaplaceAE irregular pooling mode.')
    parser.add_argument('--llapdiff_summary_irreg_hidden', type=int, default=32, help='LaplaceAE irregular adapter width.')
    parser.add_argument('--llapdiff_summary_irreg_scale', type=float, default=0.1, help='LaplaceAE irregular residual scale.')
    parser.add_argument('--llapdiff_summary_t_token_mode', type=str, choices=['none', 'residual', 'bias', 'both'], default='both', help='How first-difference tokens enter LaplaceAE.')
    parser.add_argument('--llapdiff_summary_t_token_scale', type=float, default=0.1, help='First-difference token residual scale.')
    parser.add_argument('--llapdiff_summary_pos_encoding', type=str, choices=['learned_abs', 'continuous_rope', 'learned_plus_continuous_rope'], default='continuous_rope', help='LaplaceAE temporal position encoding.')
    parser.add_argument('--llapdiff_summary_channel_balanced', type=int, default=1, help='Balance LaplaceAE reconstruction loss across variables.')
    parser.add_argument('--llapdiff_summary_loss_x', type=float, default=1.0, help='LaplaceAE value reconstruction weight.')
    parser.add_argument('--llapdiff_summary_loss_v', type=float, default=0.1, help='LaplaceAE level-dynamics reconstruction weight.')
    parser.add_argument('--llapdiff_summary_loss_t', type=float, default=0.1, help='LaplaceAE difference-dynamics reconstruction weight.')
    parser.add_argument('--llapdiff_summary_loss_dt', type=float, default=0.05, help='LaplaceAE timestamp reconstruction weight.')
    parser.add_argument('--llapdiff_summary_loss_obs', type=float, default=0.05, help='LaplaceAE missingness reconstruction weight.')
    parser.add_argument('--llapdiff_hidden_dim', type=int, default=128, help='LLapDiff backbone and conditioning width.')
    parser.add_argument('--llapdiff_num_layers', type=int, default=3, help='LLapDiff LapFormer block count.')
    parser.add_argument('--llapdiff_num_heads', type=int, default=4, help='LLapDiff attention head count.')
    parser.add_argument('--llapdiff_laplace_k', type=int, default=32, help='Number of learned Laplace poles.')
    parser.add_argument('--llapdiff_diffusion_steps', type=int, default=1000, help='LLapDiff forward diffusion steps.')
    parser.add_argument('--llapdiff_sampling_steps', type=int, default=36, help='LLapDiff DDIM sampling steps.')
    parser.add_argument('--llapdiff_num_samples', type=int, default=25, help='Forecast samples used for mean prediction and CRPS.')
    parser.add_argument('--llapdiff_schedule', type=str, choices=['linear', 'cosine'], default='cosine', help='LLapDiff noise schedule.')
    parser.add_argument('--llapdiff_predict_type', type=str, choices=['v', 'eps', 'x0'], default='v', help='LLapDiff denoising target.')
    parser.add_argument('--llapdiff_attn_dropout', type=float, default=0.0, help='LLapDiff attention dropout.')
    parser.add_argument('--llapdiff_drop_cond_p', type=float, default=0.18, help='Classifier-free history-conditioning dropout.')
    parser.add_argument('--llapdiff_guidance_strength', type=float, default=2.0, help='Classifier-free guidance strength at sampling.')
    parser.add_argument('--llapdiff_guidance_power', type=float, default=2.0, help='Classifier-free guidance schedule power.')
    parser.add_argument('--llapdiff_sample_eta', type=float, default=0.0, help='DDIM stochasticity coefficient.')
    parser.add_argument('--llapdiff_loss_weight_scheme', type=str, choices=['none', 'weighted_min_snr'], default='weighted_min_snr', help='LLapDiff diffusion loss weighting.')
    parser.add_argument('--llapdiff_minsnr_gamma', type=float, default=5.0, help='MinSNR gamma for diffusion training.')
    # ProFITi
    parser.add_argument('--profiti_source_root', type=str, default='../ProFITi', help='Path to the official ProFITi source tree.')
    parser.add_argument('--profiti_latent_dim', type=int, default=64, help='ProFITi conditioning width.')
    parser.add_argument('--profiti_conditioning_layers', type=int, default=4, help='Number of GraFITi conditioning layers in ProFITi.')
    parser.add_argument('--profiti_flow_layers', type=int, default=3, help='Number of invertible ProFITi flow blocks.')
    parser.add_argument('--profiti_attention_heads', type=int, default=1, help='Number of ProFITi conditioning attention heads.')
    parser.add_argument('--profiti_marginal_training', type=int, default=0, help='Train marginal NLL instead of normalized joint NLL.')
    parser.add_argument('--profiti_num_samples', type=int, default=25, help='Forecast samples retained for probabilistic metrics.')
    parser.add_argument('--profiti_point_num_samples', type=int, default=0, help='Monte Carlo samples used for the ProFITi point mean; zero reuses profiti_num_samples.')
    # Marginalization-consistent separable flows
    parser.add_argument('--separable_flows_source_root', type=str, default='../separable_flows', help='Path to the official separable-flows source tree.')
    parser.add_argument('--separable_flows_latent_dim', type=int, default=64, help='Separable-flows encoder and flow width.')
    parser.add_argument('--separable_flows_encoder_layers', type=int, default=1, help='Number of separable-flows encoder layers.')
    parser.add_argument('--separable_flows_flow_layers', type=int, default=3, help='Number of rational-linear spline flow layers.')
    parser.add_argument('--separable_flows_attention_heads', type=int, default=1, help='Number of separable-flows encoder attention heads.')
    parser.add_argument('--separable_flows_num_components', type=int, default=3, help='Number of Gaussian mixture components.')
    parser.add_argument('--separable_flows_num_bins', type=int, default=16, help='Number of bins in each spline transform.')
    parser.add_argument('--separable_flows_bounds', type=float, default=20.0, help='Tail bound of the spline transforms.')
    parser.add_argument('--separable_flows_encoder', type=str, choices=['transformer', 'grafiti'], default='transformer', help='Conditioning encoder used by separable flows.')
    parser.add_argument('--separable_flows_marginal_training', type=int, default=0, help='Train marginal NLL instead of normalized joint NLL.')
    parser.add_argument('--separable_flows_num_samples', type=int, default=25, help='Forecast samples used for point and probabilistic metrics.')
    # QSDiff
    parser.add_argument('--qsdiff_diffusion_steps', type=int, default=100, help='Number of DDPM forward diffusion steps.')
    parser.add_argument('--qsdiff_sampling_steps', type=int, default=20, help='Number of DDIM reverse sampling steps.')
    parser.add_argument('--qsdiff_num_samples', type=int, default=25, help='Number of test-time samples used for the mean prediction and CRPS.')
    parser.add_argument('--qsdiff_sample_temperature', type=float, default=1.0, help='Scale ensemble deviations around their mean at inference; preserves the point forecast mean while calibrating spread.')
    parser.add_argument('--qsdiff_eval_temperatures', type=str, default='', help='Comma-separated ensemble temperatures evaluated from one shared set of forecast samples.')
    parser.add_argument('--qsdiff_target_mode', type=str, choices=['query', 'grid_mask'], default='query', help='Diffuse raw query values or a fixed-grid masked future target for the matched formulation control.')
    parser.add_argument('--qsdiff_grid_future_bins', type=int, default=0, help='Future bins used by qsdiff_target_mode=grid_mask; zero uses pred_len.')
    parser.add_argument('--qsdiff_noise_type', type=str, choices=[
        'iid',
        'ou',
        'channel_ou',
        'dynamic_lowrank',
        'dqspn',
        'multi_scale_ou',
        'ou_multiscale',
        'annealed_ou',
        'ou_annealed',
        'gap_adaptive_ou',
        'ou_adaptive',
        'ou_dynamic_lowrank',
        'ou_dynamic_diag',
        'scale_aware_ou',
    ], default='iid', help='QSDiff query noise process.')
    parser.add_argument('--qsdiff_ou_gamma', type=float, default=1.0, help='OU kernel decay.')
    parser.add_argument('--qsdiff_ou_sigma0', type=float, default=1e-2, help='OU diagonal noise scale.')
    parser.add_argument('--qsdiff_ou_jitter', type=float, default=1e-4, help='Cholesky jitter for correlated noise.')
    parser.add_argument('--qsdiff_noise_hidden_dim', type=int, default=32, help='Hidden size of the dynamic noise process.')
    parser.add_argument('--qsdiff_noise_rank', type=int, default=8, help='Low-rank kernel dimension.')
    parser.add_argument('--qsdiff_noise_dyn_weight', type=float, default=0.5, help='Dynamic-kernel weight used by dqspn.')
    parser.add_argument('--qsdiff_noise_residual_weight', type=float, default=0.1, help='Residual variance weight for OU plus dynamic residual noise.')
    parser.add_argument('--qsdiff_noise_iid_residual_weight', type=float, default=0.0, help='Optional iid residual variance weight for composite QSDiff noise.')
    parser.add_argument('--qsdiff_noise_gap_weight', type=float, default=0.5, help='Gap-adaptive OU amplitude multiplier.')
    parser.add_argument('--qsdiff_noise_multiscale_lengths', type=str, default='0.03,0.10,0.30', help='Comma-separated OU length scales for multi-scale QSDiff noise.')
    parser.add_argument('--qsdiff_noise_multiscale_weights', type=str, default='', help='Optional comma-separated initial weights for multi-scale QSDiff noise; empty means uniform.')
    parser.add_argument('--qsdiff_noise_scale_mixture_mode', type=str, choices=['fixed', 'learned_global', 'query_adaptive'], default='fixed', help='Fixed, globally learned, or query-adaptive PSD mixing of multi-scale OU kernels.')
    parser.add_argument('--qsdiff_noise_anneal_min_length', type=float, default=0.03, help='Shortest OU length scale for annealed QSDiff noise.')
    parser.add_argument('--qsdiff_noise_anneal_max_length', type=float, default=0.30, help='Longest OU length scale for annealed QSDiff noise.')
    parser.add_argument('--qsdiff_noise_anneal_direction', type=str, choices=['forward', 'reverse'], default='forward', help='Increase or decrease the OU correlation length as the diffusion noise level grows.')
    parser.add_argument('--qsdiff_noise_min_length', type=float, default=1e-2, help='Minimum learned time length scale.')
    parser.add_argument('--qsdiff_noise_min_scale', type=float, default=5e-2, help='Minimum learned query noise scale.')
    parser.add_argument('--qsdiff_noise_cholesky_tries', type=int, default=4, help='Maximum Cholesky jitter escalation attempts.')
    parser.add_argument('--qsdiff_use_prior', type=int, default=0, help='Use a pretrained deterministic model as prior mean.')
    parser.add_argument('--qsdiff_prior_model', type=str, default='APN', help='Prior model: APN, HyperIMTS, GraFITi, or Warpformer (case-insensitive).')
    parser.add_argument('--qsdiff_prior_checkpoint', type=str, default='', help='Prior checkpoint file, iteration directory, or run directory.')
    parser.add_argument('--qsdiff_prior_strict', type=int, default=1, help='Strictly load the prior checkpoint.')
    parser.add_argument('--qsdiff_freeze_prior', type=int, default=1, help='Freeze the prior model parameters.')
    parser.add_argument('--qsdiff_prior_center', type=int, default=1, help='Center diffusion transitions on the prior mean.')
    parser.add_argument('--qsdiff_prior_condition', type=int, default=1, help='Condition the denoiser on the prior mean.')
    parser.add_argument('--qsdiff_prior_residual_mode', type=str, choices=['absolute', 'residual', 'gated_residual'], default='absolute', help='Predict x0 directly, as a prior residual, or as a gated prior residual.')
    parser.add_argument('--qsdiff_prior_gate_init', type=float, default=-1.0, help='Initial logit of the gated prior-residual correction.')
    parser.add_argument('--qsdiff_prior_loss_weight', type=float, default=0.0, help='Weight of the optional prior MSE loss.')
    parser.add_argument('--qsdiff_prior_anchor_weight', type=float, default=0.0, help='Weight of the support-aware denoiser-to-prior anchor; zero disables it.')
    parser.add_argument('--qsdiff_prior_anchor_gap_scale', type=float, default=0.1, help='Time-gap scale used to increase prior anchoring for unsupported queries.')
    parser.add_argument('--qsdiff_prior_sample_residual_scale', type=float, default=1.0, help='Scale sampled residuals around the prior at inference; one keeps current behavior.')
    parser.add_argument('--qsdiff_query_context_mode', type=str, choices=['none', 'same_var', 'temporal_variable'], default='none', help='Optional local same-variable or temporal-variable query conditioner.')
    parser.add_argument('--qsdiff_query_context_lengths', type=str, default='0.03,0.10,0.30', help='Comma-separated kernel lengths for the QSDiff query conditioner.')
    parser.add_argument('--qsdiff_query_context_group', type=int, default=0, help='Add a target-time query-group variable token to the conditioner.')
    parser.add_argument('--qsdiff_query_context_dropout', type=float, default=0.0, help='Dropout applied to QSDiff query-context features.')
    parser.add_argument('--qsdiff_consistency_weight', type=float, default=0.0, help='Weight of query-subset consistency; zero disables it.')
    parser.add_argument('--qsdiff_consistency_keep_prob', type=float, default=0.5, help='Query retention probability for consistency training.')
    parser.add_argument('--qsdiff_consistency_mode', type=str, choices=['x0', 'moments', 'ema_trajectory', 'covariance_trajectory', 'ema_same_step', 'kernel_pairwise'], default='x0', help='Pointwise x0, multi-view moments, EMA DDIM-path, covariance-transported trajectory, same-step EMA, or multi-scale query-pair consistency.')
    parser.add_argument('--qsdiff_consistency_views', type=int, default=2, help='Number of shared-noise views used by moment consistency.')
    parser.add_argument('--qsdiff_consistency_variance_weight', type=float, default=0.1, help='Variance-matching weight used by moment consistency.')
    parser.add_argument('--qsdiff_consistency_stopgrad', type=int, default=1, help='Use the full-query prediction as a detached consistency teacher.')
    parser.add_argument('--qsdiff_consistency_stage_weighting', type=str, choices=['none', 'alpha_bar'], default='none', help='Optional diffusion-stage confidence weighting for QSDiff consistency.')
    parser.add_argument('--qsdiff_consistency_stage_power', type=float, default=1.0, help='Exponent applied to alpha_bar when stage-weighted consistency is enabled.')
    parser.add_argument('--qsdiff_consistency_stage_max_step', type=int, default=0, help='Optional maximum diffusion step for consistency; zero keeps every step.')
    parser.add_argument('--qsdiff_consistency_teacher_decay', type=float, default=0.995, help='EMA decay of the teacher used by QSDiff EMA consistency modes.')
    parser.add_argument('--qsdiff_consistency_teacher_warmup_epochs', type=int, default=0, help='Epochs used to warm up the EMA teacher before an EMA consistency mode starts.')
    parser.add_argument('--qsdiff_consistency_trajectory_ratio', type=float, default=0.5, help='Teacher diffusion-step ratio r/s used by QSDiff cross-noise consistency modes.')
    parser.add_argument('--qsdiff_consistency_trajectory_min_step', type=int, default=1, help='Minimum student diffusion step used by QSDiff cross-noise consistency modes.')
    parser.add_argument('--qsdiff_consistency_teacher_confidence', type=str, choices=['none', 'advantage'], default='none', help='Optional detached reliability weight for a QSDiff EMA teacher.')
    parser.add_argument('--qsdiff_consistency_teacher_confidence_temperature', type=float, default=0.05, help='Softness of the qsdiff teacher-advantage reliability gate.')
    parser.add_argument('--qsdiff_consistency_graph_weight', type=float, default=0.0, help='Relative weight of optional multi-scale query-graph residual consistency for same-step EMA; zero disables it.')
    parser.add_argument('--qsdiff_clean_loss_weight', type=float, default=0.0, help='Weight of an auxiliary low-noise x0 prediction loss; zero disables it.')
    parser.add_argument('--qsdiff_clean_loss_step', type=int, default=1, help='Diffusion step used for the auxiliary x0 loss; zero is nearly clean.')
    parser.add_argument('--qsdiff_mean_loss_weight', type=float, default=0.0, help='Weight of an auxiliary deterministic mean prediction loss; zero disables it.')
    parser.add_argument('--qsdiff_ema_decay', type=float, default=0.0, help='EMA decay for QSDiff weights used for validation and checkpointing; zero disables EMA.')
    parser.add_argument('--qsdiff_low_noise_step_prob', type=float, default=0.0, help='Probability of sampling training diffusion steps from the low-noise range.')
    parser.add_argument('--qsdiff_low_noise_step_max', type=int, default=30, help='Largest diffusion step for low-noise timestep sampling.')
    parser.add_argument('--qsdiff_sample_val_interval', type=int, default=0, help='Epoch interval for sampled QSDiff validation; zero uses standard one-step validation only.')
    parser.add_argument('--qsdiff_sample_val_num_samples', type=int, default=5, help='Forecast samples used by sampled QSDiff validation.')
    parser.add_argument('--qsdiff_sample_val_sampling_steps', type=int, default=10, help='DDIM steps used by sampled QSDiff validation.')
    parser.add_argument('--qsdiff_sample_val_max_batches', type=int, default=0, help='Maximum deterministic validation batches for sampled QSDiff validation; zero uses all.')
    parser.add_argument('--qsdiff_sample_val_seed', type=int, default=2024, help='Fixed RNG seed for sampled QSDiff validation.')
    parser.add_argument('--qsdiff_crps_scale', type=str, choices=['model', 'original'], default='original', help='Calculate CRPS on model scale or after dataset inverse transformation, matching NsDiff evaluation.')

    # Used to be compatible with ipython. Never used
    parser.add_argument("--f", help="a dummy argument to fool ipython", default="1")

    configs = ExpConfigs(**vars(parser.parse_args(args))) # enable type hints

    # .yaml reference file maintainance
    yaml_configs_path_deprecated = Path(f"configs/{configs.model_name}/{configs.dataset_name}.yaml") # backward compatibility
    yaml_configs_path = Path(f"configs/{configs.model_name}/{configs.model_id}/{configs.dataset_name}.yaml")
    if yaml_configs_path.exists():
        with open(yaml_configs_path, 'r', encoding="utf-8") as stream:
            try:
                yaml_configs: dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(f"utils/configs.py: Exception when parsing {yaml_configs_path}: {exc}")
                exit(1)
        if yaml_configs is not None:
            # update yaml only if new args are added in argparse
            if_update = False
            for key, value in configs.__dict__.items():
                if key not in yaml_configs.keys():
                    if_update = True
                    yaml_configs[key] = value

            if if_update:
                with open(yaml_configs_path, 'w', encoding='utf-8') as f:
                    yaml.dump(yaml_configs, f, default_flow_style=False)
    else:
        Path(f"configs/{configs.model_name}/{configs.model_id}").mkdir(parents=True, exist_ok=True)
        if yaml_configs_path_deprecated.exists():
            # migrate from deprecated folder structure to new one
            yaml_configs_path_deprecated.replace(yaml_configs_path) # will overwrite if destination exists
        else:
            # save yaml if not exist
            with open(yaml_configs_path, 'w', encoding='utf-8') as f:
                yaml.dump(asdict(configs), f, default_flow_style=False)
                
    return configs
