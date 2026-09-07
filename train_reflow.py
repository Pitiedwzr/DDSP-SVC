import os
import argparse
import math
import torch
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from optimizer.muon import Muon_AdamW
from optimizer.aurora import Aurora_AdamW
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
                    args.model.n_chans,
                    args.model.get('spec_min', -12),
                    args.model.get('spec_max', 2),
                    args.model.get('use_aux_f0', True),
                    args.model.get('use_f0_conditioning', False),
                    args.model.get('use_self_flow', False),
                    args.model.get('self_flow_student_layer', 2),
                    args.model.get('self_flow_teacher_layer', 4),
                    args.model.get('self_flow_projector_dim', 1024),
                    args.model.get('self_flow_mask_ratio', 0.5),
                    args.model.get('self_flow_condition_mask_ratio', 0.0),
                    args.model.get('detach_ddsp_cond', True),
                    args.model.get('self_flow_span_length', 1),
                    args.model.get('self_flow_loss_on_masked_only', False))

    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    # device
    model.to(accelerator.device)

    # Read optimizer type from config, default to 'muon' for backward compatibility
    optim_type = args.train.get('optimizer', 'muon').lower()

    if optim_type == 'aurora':
        print(" [*] Initializing Aurora_AdamW Optimizer...")
        optimizer = Aurora_AdamW(model,
                                 aurora_args={'weight_decay': args.train.weight_decay},
                                 adamw_args={'weight_decay': 0.01})
    elif optim_type == 'muon':
        print(" [*] Initializing Muon_AdamW Optimizer...")
        optimizer = Muon_AdamW(model,
                               muon_args={'weight_decay': args.train.weight_decay},
                               adamw_args={'weight_decay': 0.01})
    else:
        raise ValueError(f" [x] Unknown optimizer: {optim_type}")

    # Create the EMA teacher before restoring so its state resumes with the model.
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9999))

    # load parameters
    initial_global_step, model, optimizer = utils.load_model(
        args.env.expdir, model, optimizer, ema_model=ema_model, device=args.device)
    last_step = initial_global_step - 1
    
    # Read scheduler type from config, default to 'step' if not found
    scheduler_type = args.train.get('lr_scheduler', 'step')
    div_factor = args.train.get('div_factor', 25.0)
    final_div_factor = args.train.get('final_div_factor', 10000.0)
    
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        if scheduler_type == 'step':
            # Manual LR calc for StepLR resumption
            decay_step = args.train.get('decay_step', 5000)
            gamma = args.train.get('gamma', 0.95)
            param_group['lr'] = args.train.lr * gamma ** max(last_step // decay_step, 0)
        elif scheduler_type == 'cosine':
            param_group['lr'] = args.train.lr

    if scheduler_type == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, 
            step_size=args.train.get('decay_step', 5000),
            gamma=args.train.get('gamma', 0.95),
            last_epoch=last_step
        )
    elif scheduler_type == 'cosine':
        t_max = args.train.get('t_max', 300000)
        default_eta_min = args.train.lr / div_factor / final_div_factor
        eta_min = args.train.get('eta_min', default_eta_min)
        if t_max <= 0:
            raise ValueError("train.t_max must be positive")
        if not 0.0 <= eta_min <= args.train.lr:
            raise ValueError("train.eta_min must be between 0 and train.lr")

        warmup_steps = max(1, int(t_max * 0.05))
        start_ratio = 1.0 / div_factor
        min_ratio = eta_min / args.train.lr

        def warmup_cosine(step):
            if step < warmup_steps:
                progress = step / warmup_steps
                return start_ratio + (1.0 - start_ratio) * progress
            if step < t_max:
                progress = (step - warmup_steps) / max(t_max - warmup_steps, 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_ratio + (1.0 - min_ratio) * cosine
            return min_ratio

        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_cosine,
            last_epoch=last_step)
    else:
        raise ValueError(f" [x] Unknown scheduler: {scheduler_type}")
                        
    # datas

    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False)

    # run
    model, optimizer, loader_train = accelerator.prepare(
        model, optimizer, loader_train
    )

    train(args, initial_global_step, model, ema_model, optimizer, scheduler, vocoder, loader_train, loader_valid, accelerator)
    
