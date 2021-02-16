import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
# from torch.distributed import init_process_group
from stft_loss import MultiResolutionSTFTLoss
from torch.nn.parallel import DistributedDataParallel
from dataset import MelDataset, mel_spectrogram, get_dataset_filelist
from generator import Generator, generator_loss
from discriminator import SpecDiscriminator, MultiScaleDiscriminator, feature_loss,   discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint, HParam

torch.backends.cudnn.benchmark = True


def train(rank, args, hp, hp_str):
    # if hp.train.num_gpus > 1:
    #     init_process_group(backend=hp.dist.dist_backend, init_method=hp.dist.dist_url,
    #                        world_size=hp.dist.world_size * hp.train.num_gpus, rank=rank)

    torch.cuda.manual_seed(hp.train.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(hp.model.in_channels, hp.model.out_channels).to(device)
    specd = SpecDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    stft_loss = MultiResolutionSTFTLoss()

    if rank == 0:
        print(generator)
        os.makedirs(hp.logs.chkpt_dir, exist_ok=True)
        print("checkpoints directory : ", hp.logs.chkpt_dir)

    if os.path.isdir(hp.logs.chkpt_dir):
        cp_g = scan_checkpoint(hp.logs.chkpt_dir, 'g_')
        cp_do = scan_checkpoint(hp.logs.chkpt_dir, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        specd.load_state_dict(state_dict_do['specd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if hp.train.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        specd = DistributedDataParallel(specd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), hp.train.adamG.lr, betas=[hp.train.adamG.beta1, hp.train.adamG.beta2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), specd.parameters()),
                                hp.train.adamD.lr, betas=[hp.train.adamD.beta1, hp.train.adamD.beta2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    # scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hp.train.adam.lr_decay, last_epoch=last_epoch)
    # scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hp.train.adam.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(args)

    trainset = MelDataset(training_filelist, hp.data.input_wavs, hp.data.output_wavs, hp.audio.segment_length,
                          hp.audio.filter_length, hp.audio.n_mel_channels, hp.audio.hop_length, hp.audio.win_length,
                          hp.audio.sampling_rate, hp.audio.mel_fmin, hp.audio.mel_fmax, n_cache_reuse=0,
                          shuffle=False if hp.train.num_gpus > 1 else True, fmax_loss=None, device=device)

    train_sampler = DistributedSampler(trainset) if hp.train.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=hp.train.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=hp.train.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, hp.data.input_wavs, hp.data.output_wavs, hp.audio.segment_length,
                          hp.audio.filter_length, hp.audio.n_mel_channels, hp.audio.hop_length, hp.audio.win_length,
                          hp.audio.sampling_rate, hp.audio.mel_fmin, hp.audio.mel_fmax, split=False, shuffle=False,
                        n_cache_reuse=0, fmax_loss=None, device=device)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(hp.logs.chkpt_dir, 'logs'))

    generator.train()
    specd.train()
    msd.train()
    with_postnet = False
    for epoch in range(max(0, last_epoch), args.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if hp.train.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            if steps > hp.train.postnet_start_steps:
                with_postnet = True
            x, y, file, _, y_mel_loss = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel_loss = torch.autograd.Variable(y_mel_loss.to(device, non_blocking=True))
            # y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
            before_y_g_hat, y_g_hat = generator(x, with_postnet)

            if y_g_hat is not None:
                y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), hp.audio.filter_length, hp.audio.n_mel_channels,
                                              hp.audio.sampling_rate, hp.audio.hop_length, hp.audio.win_length,
                                              hp.audio.mel_fmin, None)

            if steps > hp.train.discriminator_train_start_steps:
                for _ in range(hp.train.rep_discriminator):
                    optim_d.zero_grad()

                    # SpecD
                    y_df_hat_r, y_df_hat_g, _, _ = specd(y_mel_loss, y_g_hat_mel.detach())
                    loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

                    # MSD
                    y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                    loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                    loss_disc_all = loss_disc_s + loss_disc_f

                    loss_disc_all.backward()
                    optim_d.step()

            before_y_g_hat_mel = mel_spectrogram(before_y_g_hat.squeeze(1), hp.audio.filter_length, hp.audio.n_mel_channels,
                                              hp.audio.sampling_rate, hp.audio.hop_length, hp.audio.win_length,
                                              hp.audio.mel_fmin, None)
            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            # before_loss_mel = F.l1_loss(y_mel_loss, before_y_g_hat_mel)
            sc_loss, mag_loss = stft_loss(before_y_g_hat[:, :, :y.size(2)].squeeze(1), y.squeeze(1))
            before_loss_mel = sc_loss + mag_loss

            # L1 Sample Loss
            before_loss_sample = F.l1_loss(y, before_y_g_hat)
            loss_gen_all = before_loss_mel + before_loss_sample

            if y_g_hat is not None:
                # L1 Mel-Spectrogram Loss
                # loss_mel = F.l1_loss(y_mel_loss, y_g_hat_mel)
                sc_loss_, mag_loss_ = stft_loss(y_g_hat[:, :, :y.size(2)].squeeze(1), y.squeeze(1))
                loss_mel = sc_loss_ + mag_loss_
                # L1 Sample Loss
                loss_sample = F.l1_loss(y, y_g_hat)
                loss_gen_all += loss_mel + loss_sample


            if steps > hp.train.discriminator_train_start_steps:
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = specd(y_mel_loss, y_g_hat_mel)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all += hp.model.lambda_adv * (loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f)

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % args.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel_loss, before_y_g_hat_mel).item()
                        sample_error =  F.l1_loss(y, before_y_g_hat)

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Sample Error: {:4.3f}, '
                          'Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, sample_error, mel_error, time.time() - start_b))

                # checkpointing
                if steps % hp.logs.save_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(hp.logs.chkpt_dir, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if hp.train.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(hp.logs.chkpt_dir, steps)
                    save_checkpoint(checkpoint_path,
                                    {'specd': (specd.module if hp.train.num_gpus > 1
                                             else specd).state_dict(),
                                     'msd': (msd.module if hp.train.num_gpus > 1
                                             else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch, 'hp_str': hp_str})

                # Tensorboard summary logging
                if steps % hp.logs.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % hp.logs.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, file, y_mel, y_mel_loss = batch
                            x = x.unsqueeze(1)
                            y = y.unsqueeze(1).to(device)
                            before_y_g_hat, y_g_hat = generator(x.to(device))
                            y_mel_loss = torch.autograd.Variable(y_mel_loss.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(before_y_g_hat.squeeze(1), hp.audio.filter_length, hp.audio.n_mel_channels,
                                          hp.audio.sampling_rate, hp.audio.hop_length, hp.audio.win_length,
                                          hp.audio.mel_fmin, None)
                            val_err_tot += F.l1_loss(y_mel_loss, y_g_hat_mel).item()
                            val_err_tot += F.l1_loss(y, before_y_g_hat).item()
                            if y_g_hat is not None:
                                val_err_tot += F.l1_loss(y, y_g_hat).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt_noise/y_{}'.format(j), x[0], steps, hp.audio.sampling_rate)
                                    sw.add_audio('gt_clean/y_{}'.format(j), y[0], steps, hp.audio.sampling_rate)
                                    sw.add_figure('gt/y_spec_clean_{}'.format(j), plot_spectrogram(y_mel[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), before_y_g_hat[0], steps, hp.audio.sampling_rate)
                                if y_g_hat is not None:
                                    sw.add_audio('generated/y_hat_after_{}'.format(j), y_g_hat[0], steps,
                                                 hp.audio.sampling_rate)
                                y_hat_spec = mel_spectrogram(before_y_g_hat.squeeze(1), hp.audio.filter_length, hp.audio.n_mel_channels,
                                          hp.audio.sampling_rate, hp.audio.hop_length, hp.audio.win_length,
                                          hp.audio.mel_fmin, None)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        # scheduler_g.step()
        # scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--train_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--valid_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('-c', '--config', default='config.yaml')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    torch.manual_seed(hp.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.train.seed)
        hp.train.num_gpus = torch.cuda.device_count()
        hp.train.batch_size = int(hp.train.batch_size / hp.train.num_gpus)
        print('Batch size per GPU :', hp.train.batch_size)
    else:
        pass

    if hp.train.num_gpus > 1:
        mp.spawn(train, nprocs=hp.train.num_gpus, args=(args, hp, hp_str,))
    else:
        train(0, args, hp, hp_str)


if __name__ == '__main__':
    main()