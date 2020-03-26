from model import Generator
from model import Discriminator
import torch
import torch.nn.functional as F
from os.path import join, basename, dirname, split
import time
import datetime
from data_loader import to_categorical
from utils import *
from tqdm import tqdm


def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, train_loader, test_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sampling_rate = config.sampling_rate

        # Model configurations.
        self.num_speakers = config.num_speakers
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model.
        self.generator = Generator(num_speakers=self.num_speakers)
        self.discriminator = Discriminator(num_speakers=self.num_speakers)

        self.g_optimizer = torch.optim.Adam(params=self.generator.parameters(),
                                            lr=self.g_lr,
                                            weight_decay=1e-5,
                                            betas=[self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(params=self.discriminator.parameters(),
                                            lr=self.d_lr,
                                            weight_decay=1e-5,
                                            betas=[self.beta1, self.beta2])

        print_network(self.generator, 'Generator')
        print_network(self.discriminator, 'Discriminator')

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Build tensorboard.
        if self.use_tensorboard:
            self.build_tensorboard()

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.generator.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.discriminator.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def sample_spk_c(self, size):
        spk_c = np.random.randint(0, self.num_speakers, size=size)
        spk_c_cat = to_categorical(spk_c, self.num_speakers)
        return torch.LongTensor(spk_c), torch.FloatTensor(spk_c_cat)

    def classification_loss(self, logit, target):
        """Compute softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def load_wav(self, wavfile, sr=16000):
        wav, _ = librosa.load(wavfile, sr=sr, mono=True)
        return wav_padding(wav, sr=16000, frame_period=5, multiple=4)

    def save_optim_checkpoints(self, g_name, d_name, type_saving):
        G_path = os.path.join(self.model_save_dir, g_name)
        D_path = os.path.join(self.model_save_dir, d_name)
        torch.save(self.generator.state_dict(), G_path)
        torch.save(self.discriminator.state_dict(), D_path)
        print('Saved {} optimal model checkpoints into {}...'.format(type_saving, self.model_save_dir))

    def train(self):
        """Train StarGAN."""

        # Set data loader.
        train_loader = self.train_loader
        data_iter = iter(train_loader)

        # Read a batch of testdata
        test_wavfiles = self.test_loader.get_batch_test_data(batch_size=4)
        test_wavs = [self.load_wav(wavfile) for wavfile in test_wavfiles]

        # Determine whether do copysynthesize when first do training-time conversion test.
        cpsyn_flag = [True, False][0]
        # f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print("resuming step %d ..." % self.resume_iters)
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Keep a track of loss for checkpoint saving
        g_adv_optim = 0            # init optimum g_adv
        g_adv_converge_low = True  # check which direction g_adv is converging (init as low)
        g_rec_optim = 0            # init optimum g_rec
        g_rec_converge_low = True  # check which direction g_rec is converging (init as low)
        g_tot_optim = 0            # converges around 37
        g_tot_converge_low = True  # check which direction g_tot is converging (init as low)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch labels.
            try:
                mc_real, spk_label_org, spk_c_org = next(data_iter)
            except:
                data_iter = iter(train_loader)
                mc_real, spk_label_org, spk_c_org = next(data_iter)

            mc_real.unsqueeze_(1)  # (B, D, T) -> (B, 1, D, T) for conv2d

            # Generate target domain labels randomly.
            # spk_label_trg: int,   spk_c_trg:one-hot representation
            spk_label_trg, spk_c_trg = self.sample_spk_c(mc_real.size(0))

            mc_real = mc_real.to(self.device)  # Input mc.
            spk_label_org = spk_label_org.to(self.device)  # Original spk labels.
            spk_c_org = spk_c_org.to(self.device)  # Original spk acc conditioning.
            spk_label_trg = spk_label_trg.to(self.device)  # Target spk labels for classification loss for G.
            spk_c_trg = spk_c_trg.to(self.device)  # Target spk conditioning.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real mc feats.
            d_out_src = self.discriminator(mc_real, spk_c_org, spk_c_trg)
            d_loss_real = - torch.mean(torch.log(d_out_src))

            # Compute loss with fake mc feats.
            mc_fake = self.generator(mc_real, spk_c_trg)
            d_out_fake = self.discriminator(mc_fake.detach(), spk_c_org, spk_c_trg)
            d_loss_fake = - torch.mean(torch.log(d_out_fake))

            d_loss = d_loss_real + d_loss_fake

            # TODO: look to include Wassertein GP later - original paper does not include this
            # Compute loss for gradient penalty.
            alpha = torch.rand(mc_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * mc_real.data + (1 - alpha) * mc_fake.data).requires_grad_(True)
            out_src = self.discriminator(x_hat, spk_c_org, spk_c_trg)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)
            # d_loss = d_loss + self.lambda_gp * d_loss_gp

            # Backward and optimize.
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_gp'] = d_loss_gp.item()
            loss['D/loss'] = d_loss.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                # Original-to-target domain.
                mc_fake = self.generator(mc_real, spk_c_trg)
                out_fake = self.discriminator(mc_fake, spk_c_org, spk_c_trg)
                g_loss_fake = torch.mean(torch.log(out_fake))

                # Target-to-original domain.
                mc_reconst = self.generator(mc_fake, spk_c_org)
                g_loss_rec = torch.mean(torch.abs(mc_real - mc_reconst))

                # Original-to-original, Id mapping loss.
                mc_fake_id = self.generator(mc_real, spk_c_trg)
                g_loss_id = torch.mean(torch.abs(mc_real - mc_fake_id))

                # Backward and optimize.
                if (i + 1) < 10 ** 4:  # only calc. id mapping loss on first 10^4 iters.
                    g_loss = g_loss_fake \
                             + self.lambda_rec * g_loss_rec \
                             + self.lambda_id * g_loss_id
                else:
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec

                # Check convergence direction of losses
                if (i + 1) == 20 * (10 ** 3):  # update optims at 20000 iters
                    g_adv_optim = g_loss_fake
                    g_rec_optim = g_loss_rec
                    g_tot_optim = g_loss
                if (i + 1) == 70 * (10 ** 3):  # check which direction optims have gone over 50000 iters
                    if g_loss_fake > g_adv_optim:
                        g_adv_converge_low = False
                    if g_loss_rec > g_rec_optim:
                        g_rec_converge_low = False
                    if g_loss > g_tot_optim:
                        g_tot_converge_low = False
                    print('CONVERGE DIRECTION')
                    print(f'adv_loss low: {g_adv_converge_low}')
                    print(f'g_rec_loss los: {g_rec_converge_low}')
                    print(f'g_loss loq: {g_tot_converge_low}')

                # Update loss for checkpoint saving
                if (i + 1) > 75 * (10 ** 3):  # only start saving at high enough epochs
                    # adv and reconstruction together
                    if g_tot_converge_low:
                        if (g_loss_fake < g_adv_optim and abs(g_loss_fake - g_adv_optim) > 0.1) and g_loss_rec < g_rec_optim:
                            self.save_optim_checkpoints('g_adv_rec_optim-G.ckpt', 'g_adv_rec_optim-D.ckpt', 'adv+rec')
                    elif not g_tot_converge_low:
                        if (g_loss_fake > g_adv_optim and abs(g_loss_fake - g_adv_optim) > 0.1) and g_loss_rec < g_rec_optim:
                            self.save_optim_checkpoints('g_adv_rec_optim-G.ckpt', 'g_adv_rec_optim-D.ckpt', 'adv+rec')

                    # adv optimal model point
                    if g_adv_converge_low:
                        if g_loss_fake < g_adv_optim:
                            g_adv_optim = g_loss_fake
                            self.save_optim_checkpoints('g_adv_optim-G.ckpt', 'g_adv_optim-D.ckpt', 'adv')
                    elif not g_adv_converge_low:
                        if g_loss_fake < g_adv_optim:
                            g_adv_optim = g_loss_fake
                            self.save_optim_checkpoints('g_adv_optim-G.ckpt', 'g_adv_optim-D.ckpt', 'adv')

                    # reconstruction optimal model point
                    if g_rec_converge_low:
                        if g_loss_rec < g_rec_optim:
                            g_rec_optim = g_loss_rec
                            self.save_optim_checkpoints('g_rec_optim-G.ckpt', 'g_rec_optim-D.ckpt', 'rec')
                    elif not g_rec_converge_low:
                        if g_loss_rec > g_rec_optim:
                            g_rec_optim = g_loss_rec
                            self.save_optim_checkpoints('g_rec_optim-G.ckpt', 'g_rec_optim-D.ckpt', 'rec')

                    # total loss optimal model point
                    if g_tot_converge_low:
                        if g_loss < g_tot_optim:
                            g_tot_optim = g_loss
                            self.save_optim_checkpoints('g_tot_optim-G.ckpt', 'g_tot_optim-D.ckpt', 'tot')
                    elif not g_tot_converge_low:
                        if g_loss > g_tot_optim:
                            g_tot_optim = g_loss
                            self.save_optim_checkpoints('g_tot_optim-G.ckpt', 'g_tot_optim-D.ckpt', 'tot')

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/g_loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss'] = g_loss.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)

            if (i + 1) % self.sample_step == 0:
                # TODO: change to parameters
                sampling_rate = 16000
                num_mcep = 36
                frame_period = 5
                with torch.no_grad():
                    for idx, wav in tqdm(enumerate(test_wavs)):
                        wav_name = basename(test_wavfiles[idx])
                        # print(wav_name)
                        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
                        f0_converted = pitch_conversion(f0=f0,
                                                        mean_log_src=self.test_loader.logf0s_mean_src,
                                                        std_log_src=self.test_loader.logf0s_std_src,
                                                        mean_log_target=self.test_loader.logf0s_mean_trg,
                                                        std_log_target=self.test_loader.logf0s_std_trg)
                        coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)

                        coded_sp_norm = (coded_sp - self.test_loader.mcep_mean_src) / self.test_loader.mcep_std_src
                        coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(
                            self.device)
                        conds = torch.FloatTensor(self.test_loader.spk_c_trg).to(self.device)
                        # print(conds.size())
                        coded_sp_converted_norm = self.generator(coded_sp_norm_tensor, conds).data.cpu().numpy()
                        coded_sp_converted = np.squeeze(
                            coded_sp_converted_norm).T * self.test_loader.mcep_std_trg + self.test_loader.mcep_mean_trg
                        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                        # decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                        wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted,
                                                                 ap=ap, fs=sampling_rate, frame_period=frame_period)

                        librosa.output.write_wav(
                            join(self.sample_dir, str(i + 1) + '-' + wav_name.split('.')[0] + '-vcto-{}'.format(
                                self.test_loader.trg_spk) + '.wav'), wav_transformed, sampling_rate)
                        if cpsyn_flag:
                            wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp,
                                                               ap=ap, fs=sampling_rate, frame_period=frame_period)
                            librosa.output.write_wav(join(self.sample_dir, 'cpsyn-' + wav_name), wav_cpsyn,
                                                     sampling_rate)
                    cpsyn_flag = False

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.generator.state_dict(), G_path)
                torch.save(self.discriminator.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
