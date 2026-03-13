import os
import argparse
import torch
from torch.optim import lr_scheduler
from optimizer.muon import Muon_AdamW
from logger import utils
from reflow.data_loaders import get_data_loaders
from reflow.vocoder import Vocoder, Unit2Wav
from accelerate import Accelerator

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)

    amp_type = args.train.amp_dtype if args.train.amp_dtype != 'fp32' else "no"
    accelerator = Accelerator(mixed_precision=amp_type, split_batches=True)

    args.device = accelerator.device

    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)
    
    # load model
    if args.model.type == 'RectifiedFlow':
        from reflow.solver import train
        model = Unit2Wav(
                    args.data.sampling_rate,
                    args.data.block_size,
                    args.model.win_length,
                    args.data.encoder_out_channels, 
                    args.model.n_spk,
                    args.model.use_norm,
                    args.model.use_attention,
                    args.model.use_pitch_aug,
                    vocoder.dimension,
                    args.model.n_aux_layers,
                    args.model.n_aux_chans,
                    args.model.n_layers,
                    args.model.n_chans) 
                    
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    # device
    model.to(accelerator.device)
    
    # load parameters
    optimizer = Muon_AdamW(model, 
                    muon_args={'weight_decay': args.train.weight_decay}, 
                    adamw_args={'weight_decay': 0})
    initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)
    last_step = initial_global_step - 1
    
    # Read scheduler type from config, default to 'step' if not found
    scheduler_type = getattr(args.train, 'lr_scheduler', 'step')
    
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        if scheduler_type == 'step':
            # Manual LR calc for StepLR resumption
            param_group['lr'] = args.train.lr * args.train.gamma ** max((last_step) // getattr(args.train, 'decay_step', 5000), 0)
        elif scheduler_type == 'cosine':
            # CosineAnnealingLR calculates LR internally based on last_epoch
            param_group['lr'] = args.train.lr

    if scheduler_type == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, 
            step_size=getattr(args.train, 'decay_step', 5000), 
            gamma=getattr(args.train, 'gamma', 0.95), 
            last_epoch=last_step
        )
    elif scheduler_type == 'cosine':
        t_max = getattr(args.train, 't_max', 200000) # Total training steps
        eta_min = getattr(args.train, 'eta_min', 1e-6) # Minimum learning rate
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=t_max, 
            eta_min=eta_min, 
            last_epoch=last_step
        )
    else:
        raise ValueError(f" [x] Unknown scheduler: {scheduler_type}")
                        
    # datas

    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False)
    
    # run
    model, optimizer, loader_train = accelerator.prepare(
        model, optimizer, loader_train
    )
    train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_valid, accelerator)
    
