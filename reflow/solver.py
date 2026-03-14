import os
import time
import random
import numpy as np
import torch
import librosa
from logger.saver import Saver
from logger import utils
from nsf_hifigan.nvSTFT import STFT

def calculate_mel_snr(gt_mel, pred_mel):
    # 计算误差图像
    error_image = gt_mel - pred_mel
    # 计算参考图像的平方均值
    mean_square_reference = torch.mean(gt_mel ** 2)
    # 计算误差图像的方差
    variance_error = torch.var(error_image)
    # 计算并返回SNR
    snr = 10 * torch.log10((mean_square_reference + 1e-9) / (variance_error + 1e-9))
    return snr


def calculate_mel_si_snr(gt_mel, pred_mel):
    # 将测试图像按比例调整以最小化误差
    scale = torch.sum(gt_mel * pred_mel) / torch.sum(gt_mel ** 2)
    test_image_scaled = scale * pred_mel
    # 计算误差图像
    error_image = gt_mel - test_image_scaled
    # 计算参考图像的平方均值
    mean_square_reference = torch.mean(gt_mel ** 2)
    # 计算误差图像的方差
    variance_error = torch.var(error_image)
    # 计算并返回SI-SNR
    si_snr = 10 * torch.log10(mean_square_reference / variance_error)
    return si_snr


def calculate_mel_psnr(gt_mel, pred_mel):
    # 计算误差图像
    error_image = gt_mel - pred_mel
    # 计算误差图像的均方误差
    mse = torch.mean(error_image ** 2)
    # 计算参考图像的最大可能功率
    max_power = torch.max(gt_mel) ** 2
    # 计算并返回PSNR
    psnr = 10 * torch.log10(max_power / mse)
    return psnr

def test(args, model, vocoder, loader_test, saver):
    print(' [*] testing...')
    model.eval()

    # losses
    test_ddsp_loss = 0.
    test_reflow_loss = 0.
    test_f0_loss = 0.

    # mel mse val
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0

    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    spec_min = -2
    spec_max = 10
    spec_range = 12
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            mel = model(
                    data['units'], 
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    infer=True,
                    return_wav=False,
                    infer_step=args.infer.infer_step, 
                    method=args.infer.method,
                    t_start=args.model.t_start)
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()
                        
            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            ddsp_loss, reflow_loss, f0_loss = model(
                data['units'], 
                data['f0'], 
                data['volume'], 
                data['spk_id'],
                vocoder=vocoder,
                gt_spec=data['mel'],
                infer=False,
                t_start=args.model.t_start)
            test_ddsp_loss += ddsp_loss.item()
            test_reflow_loss += reflow_loss.item()
            test_f0_loss += f0_loss.item()
            
            # log mel
            saver.log_spec(data['name'][0], data['mel'], mel)
            
            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': signal})

            # ==========================================
            # FIX: Removed the dead WAV2MEL STFT code here.
            # It was calculating pre_mel and gt_mel but never using them.
            # ==========================================

            # FIX: Match the actual bounds defined in reflow.py
            spec_min = -12.0
            spec_max = 2.0
            spec_range = spec_max - spec_min # 14.0

            # FIX: Correct Min-Max Normalization to [0, 1] range for accurate power math
            gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
            gt_mel_norm = (gt_mel_norm - spec_min) / spec_range
            pre_mel_norm = torch.clip(mel, spec_min, spec_max)
            pre_mel_norm = (pre_mel_norm - spec_min) / spec_range
            # 计算指标
            mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
            mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
            
    # report
    test_ddsp_loss /= num_batches
    test_reflow_loss /= num_batches 
    test_f0_loss /= num_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    print(' [test_ddsp_loss] test_ddsp_loss:', test_ddsp_loss)
    print(' [test_reflow_loss] test_reflow_loss:', test_reflow_loss)
    print(' [test_f0_loss] test_f0_loss:', test_f0_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    saver.log_value({
        'validation/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        'validation/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        'validation/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        'validation/mel_val_sisnr': mel_val_sisnr_all
    })
    return test_ddsp_loss, test_reflow_loss


def train(args, initial_global_step, model, ema_model, optimizer, scheduler, vocoder, loader_train, loader_test, accelerator):
    # saver
    saver = None

    if accelerator.is_main_process:
        saver = Saver(args, initial_global_step=initial_global_step)
        params_count = utils.get_network_paras_amount({'model': accelerator.unwrap_model(model)})
        saver.log_info('--- model size ---')
        saver.log_info(params_count)
        saver.log_info('======= start training =======')

    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()

    global_step = initial_global_step

    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            global_step += 1
            if accelerator.is_main_process:
                saver.global_step = global_step

            optimizer.zero_grad()
            
            # CFG
            drop_spk = random.random() < 0.15
            
            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(accelerator.device)
            
            # forward
            ddsp_loss, reflow_loss, f0_loss = model(
                data['units'].float(), data['f0'], data['volume'], data['spk_id'], 
                aug_shift=data['aug_shift'], vocoder=vocoder, 
                gt_spec=data['mel'].float(), infer=False, 
                t_start=args.model.t_start, drop_spk=drop_spk
            )
            
            loss = args.train.lambda_ddsp * ddsp_loss + reflow_loss + 0.1 * f0_loss

            # Removed the "if isnan: continue" block completely
            # When using accelerate, DO NOT zero_grad and DO NOT continue
            if torch.isnan(loss) and accelerator.is_main_process:
                print(' [!] NaN loss detected. Letting Accelerate auto-fix the scale...')

            # Backward Pass
            # If loss is NaN, this safely generates NaN gradients for the Scaler to catch.
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            # Optimizer Step
            # The GradScaler intercepts the step here! If gradients are NaN, it safely 
            # aborts the weight update and dynamically lowers the FP16 scale factor.
            optimizer.step()

            # Step scheduler ONLY if the optimizer didn't skip due to NaNs
            if not accelerator.optimizer_step_was_skipped:
                scheduler.step()
                ema_model.update_parameters(model)

            # log loss
            if accelerator.is_main_process and global_step % args.train.interval_log == 0:
                current_lr = optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch, batch_idx, num_batches, args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr, loss.item(), saver.get_total_time(), global_step
                    )
                )
                saver.log_value({
                    'train/loss': loss.item(), 'train/ddsp_loss': ddsp_loss.item(),
                    'train/reflow_loss': reflow_loss.item(), 'train/f0_loss': f0_loss.item(),
                    'train/lr': current_lr
                })
            accelerator.wait_for_everyone()

            # validation
            if global_step % args.train.interval_val == 0:
                if accelerator.is_main_process:
                    optimizer_save = optimizer if args.train.save_opt else None
                    unwrapped_model = accelerator.unwrap_model(model)
                    
                    saver.save_model(unwrapped_model, optimizer_save, postfix=f'{global_step}', ema_model=ema_model)
                    last_val_step = global_step - args.train.interval_val
                    if last_val_step % args.train.interval_force_save != 0:
                        saver.delete_model(postfix=f'{last_val_step}')
                    
                    test_ddsp_loss, test_reflow_loss = test(args, unwrapped_model, vocoder, loader_test, saver)
                    test_loss = args.train.lambda_ddsp * test_ddsp_loss + test_reflow_loss
                    
                    saver.log_info(' --- <validation> --- \nloss: {:.3f}. '.format(test_loss))
                    saver.log_value({'validation/loss': test_loss, 'validation/ddsp_loss': test_ddsp_loss, 'validation/reflow_loss': test_reflow_loss})
                
                accelerator.wait_for_everyone()
                model.train()
