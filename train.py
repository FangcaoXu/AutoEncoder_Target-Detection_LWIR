import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.utils.data import DataLoader, Dataset
import random
# from msssim import msssim, ssim

LOSSES = {
    'mse': {'func': F.mse_loss, 'negative': False},
    # 'ssim': {'func': ssim, 'negative': True},
    # 'msssim': {'func': msssim, 'negative': True},
}


class NpzDataset(Dataset):
    def __init__(self, npz_file, normalize=True, keys=None, reshape=None):
        """
        Args:
            npz_file (string): Path to the npz file. The file can have multiple
                tensors with the same shape.
            keys (list): name of the keys in the npz to read. If None reads
                all the keys
            reshape (tuple): reshape all tensors to the  specified dimensions.
            normalize (bool, optional): Optional normalization.
        """
        self.npz = np.load(npz_file)
        self.keys = keys if keys else list(self.npz.keys())
        self.max = []
        self.data = []
        for key in self.keys:
            d = self.npz[key]
            if normalize:
                self.max.append(d.max())
                d = d / d.max()              # d.max() is the maximum value of each component across all days, time, and different reflectances
            if reshape:
                d = d.reshape(reshape)
            self.data.append(d)
        self.data = np.array(self.data)      # numpy.ndarray
        assert all(arr.shape == self.data[0].shape for arr in self.data)   # data[0] is the first key's all matrix tables
        # all() function returns True iff all items in an iterable are true
        print(self.data.shape)     # shape: (2, 7854, 256, 8)

    def __len__(self):
        return self.data[0].shape[0]

    # accessing list items, dictionary entries, array elements etc.
    # NpzDataset[i] is a list of (256,8) matrix for all keys. The returned data shape is (7854, 2, 256, 8)
    def __getitem__(self, idx):
        # i is the key index, data[i]: (7854, 256, 8), idx is the index of the matrix of the data[i]
        return [torch.unsqueeze(torch.from_numpy(self.data[i][idx]).float(), 0)
                for i, key in enumerate(self.keys)]

# dataset = NpzDataset("/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/data1_256x8.npz", keys=["total", "grnd", "up1_emission"], reshape=(7854, 1, 256, 8))
# dataset[77], it will return [torch.unsqueeze(torch.from_numpy(self.data[i][77]).float(), 0) for i, key in enumerate(self.keys)]
# two tensors, one is the first matrix (256,8) from (7854, 256, 8) np.array of the first key, second one is the first matrix (256,8) from (7854, 256, 8) np.array of the second key


class Encoder(nn.Module):
    # instantiate all modules
    def __init__(self, nc, ndf, nz):
        super(Encoder, self).__init__()
        self.nz = nz
        self.main = [
            # input is (nc) x 256 x 8
            # Number of channels in the input image, nc: 1
            # Number of channels produced by the convolution, ndf: 64
            # kernel_size: Size of the convolving kernel, 4
            # stride: Stride of the convolution, 2
            # padding: padding added to both sides of the input, 1
            # nz: vector size in the latent space, 16

            # # state size. 1 x 256 x 8
            # nn.Conv2d(nc,  ndf, 4, 2, 1, bias=False),
            # input is (nc) x 256 x 7
            nn.Conv2d(nc, ndf, (4, 3), 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 4
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 2
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 1
            nn.Conv2d(ndf * 4, self.nz, (32, 1), 1, 0, bias=False)
            # state size. a latent vector of size 16
            # (32,1) first 'int' is for the height dimension, and second int is for the width dimension
        ]

        # module here is
        # 0: Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        # 1: LeakyReLU(negative_slope=0.2, inplace)
        # 2: Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        # 3: BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # 4: LeakyReLU(negative_slope=0.2, inplace)
        # 5: Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        # 6: BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # 7: LeakyReLU(negative_slope=0.2, inplace)
        # 8: Conv2d(256, 16, kernel_size=(32, 1), stride=(1, 1), bias=False)
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)  # type(encoder._modules): collections.OrderedDict

    # define the network structure.
    # usage: after Initiating a model through AutoEncoder, model(inputdata) can pass the input data into network
    def forward(self, x):
        for layer in self.main:
            x = layer(x)   # layer: Conv2d, LeakyRelU.....
        return x


class Decoder(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(Decoder, self).__init__()
        # Number of channels in the input image, latent vector of size nz: 16
        # Number of channels produced by the convolution, ngf: 64
        # nc: 1
        self.main = [
            # input goes into a convolution
            # nz: 16; ngf: 64
            #
            nn.ConvTranspose2d(nz, ngf * 4, (32,1), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf,  nc, (4,3), 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, nc, ndf, ngf, nz):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(nc, ndf, nz)
        self.decoder = Decoder(nc, ngf, nz)
        self.add_module('encoder', self.encoder)
        self.add_module('decoder', self.decoder)

    def forward(self, x):
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)
        return reconstruction, encoding


def save_model(path, encoder_state, decoder_state, optimizer_state, **kwargs):
    torch.save({
        'encoder_state': encoder_state,
        'decoder_state': decoder_state,
        'optimizer_state': optimizer_state,
        **kwargs
    }, path)


def load_model_trainig(path, cuda):
    chkpt = torch.load(path, torch.device('cpu'))

    model = AutoEncoder(chkpt['nc'], chkpt['ndf'], chkpt['ngf'], chkpt['nz'])
    model.encoder.load_state_dict(chkpt['encoder_state'])
    model.decoder.load_state_dict(chkpt['decoder_state'])

    if cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=chkpt['lr'])
    optimizer.load_state_dict(chkpt['optimizer_state'])
    # return the state of the optimizer as a dict. It contains two entries: state, param_groups
    if cuda:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    epoch = chkpt['epoch']
    loss = chkpt['loss']

    model.train()
    return model, optimizer, loss, epoch


def load_model_predict(path, cuda):
    chkpt = torch.load(path, torch.device('cpu'))

    model = AutoEncoder(chkpt['nc'], chkpt['ndf'], chkpt['ngf'], chkpt['nz'])
    model.encoder.load_state_dict(chkpt['encoder_state'])
    model.decoder.load_state_dict(chkpt['decoder_state'])

    if cuda:
        model.cuda()

    model.eval()

    return model, chkpt['x_norm'], chkpt['y_norm']


def load_encoder(path, cuda):
    chkpt = torch.load(path, torch.device('cpu'))

    encoder = Encoder(chkpt['nc'], chkpt['ndf'], chkpt['nz'])
    encoder.load_state_dict(chkpt['encoder_state'])
    if cuda:
        encoder.cuda()

    encoder.eval()
    return encoder


def load_decoder(path, cuda):
    chkpt = torch.load(path, torch.device('cpu'))

    decoder = Decoder(chkpt['nc'], chkpt['ngf'], chkpt['nz'])
    decoder.load_state_dict(chkpt['decoder_state'])
    if cuda:
        decoder.cuda()

    decoder.eval()
    return decoder


def save_plot(sample, file):
    samples = sample
    samples = samples.data.numpy()[:16]

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        sample = np.squeeze(sample)
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if len(sample.shape) == 3:
            sample = np.swapaxes(sample, 0, 2)
        plt.imshow(sample)

    plt.savefig(file, bbox_inches='tight')
    plt.close(fig)


def main(args):
    cuda = not args.no_cuda and torch.cuda.is_available()
    loss_name = args.loss
    loss_func = LOSSES[loss_name]['func']
    negative_loss = LOSSES[loss_name]['negative']

    # fix the seeds
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if cuda:
            # Sets the seed for generating random numbers on all GPUs.
            # If CUDA is not available; in that case, it is silently ignored.
            torch.cuda.manual_seed(args.seed)
            # cuDNN is a NVIDIA CUDA GPU-accelerated deep neural network library
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # set model parameters
    lr = 1e-4 # learning rate
    nc = 1  # number of channels
    nz = args.vector_size  # size of latent vector
    ngf = args.filter_factor  # decoder (generator) filter factor
    ndf = args.filter_factor  # encoder filter factor

    out_dir = "{outdir}/l-{ln}_batch{bs}_filters{nf}_vector{nz}_in{input}_out{out}".format(
        outdir = args.out_dir,
        ln=loss_name,
        bs=args.batch_size,
        nf=args.filter_factor,
        nz=nz,
        input=args.input,
        out=args.label
    )

    dataset = NpzDataset(args.data, keys=[args.input, args.label], reshape=(args.datalength, args.matrixsize[0], args.matrixsize[1])) # list
    x_norm, y_norm = dataset.max

    # dataset = np.load(args.data)[args.input]
    data_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.workers, drop_last=True)

    m_path = os.path.join(out_dir, "ae_latest.tar")
    if os.path.exists(m_path):
        print("Resuming training from saved state:")
        model, optimizer, loss, cnt = load_model_trainig(m_path, cuda)
    else:
        model = AutoEncoder(nc, ndf, ngf, nz)
        if cuda:
            model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        cnt = 0

    dataset_pass = 0
    while cnt <= args.epochs:
        for batch_idx, batch_item in enumerate(data_loader):
            X, Y = batch_item
            if cuda:
                X = X.cuda()
                Y = Y.cuda()

            # return the reconstruction and encoding (latent vector)  from the autoencoder model with input

            X_sample, X_encoded = model(X)
            if negative_loss:
                loss = -loss_func(X_sample, Y)
            else:
                loss = loss_func(X_sample, Y)

            loss.backward()
            optimizer.step()
            model.zero_grad()

            if batch_idx % args.log_interval == 0:
                print('{}: Iter-{}; recon_loss: {:.4}'.format(str(dataset_pass).zfill(2), batch_idx, loss.data))

            if cnt % args.save_interval == 0:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                if cuda:
                    # X_encoded = X_encoded.cpu()
                    X = X.cpu()
                    Y = Y.cpu()
                    X_sample = X_sample.cpu()
                # print(np.around(X_encoded.data.numpy()[:16, :, 0, 0], decimals=2))
                save_plot(Y, '{}/{}_{}-{}_gt.png'.format(out_dir, str(cnt).zfill(5), dataset_pass, batch_idx))
                save_plot(X, '{}/{}_{}-{}_in.png'.format(out_dir, str(cnt).zfill(5), dataset_pass, batch_idx))
                save_plot(X_sample, '{}/{}_{}-{}_out.png'.format(out_dir, str(cnt).zfill(5), dataset_pass, batch_idx))
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                save_model(os.path.join(out_dir, "ae_{}.tar".format(str(cnt).zfill(7))),
                           model.encoder.state_dict(),
                           model.decoder.state_dict(),
                           optimizer.state_dict(),
                           loss_name=loss_name,
                           loss=loss, epoch=cnt, lr=lr,
                           nc=nc, ndf=ndf, ngf=ngf, nz=nz, x_norm=x_norm, y_norm=y_norm)
            cnt += 1
        dataset_pass += 1

    save_model(m_path,
               model.encoder.state_dict(),
               model.decoder.state_dict(),
               optimizer.state_dict(),
               loss_name=loss_name,
               loss=loss, epoch=cnt, lr=lr,
               nc=nc, ndf=ndf, ngf=ngf, nz=nz, x_norm=x_norm, y_norm=y_norm)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CAE Encoder')

    # data options
    parser.add_argument('--data', type=str, required=True, help='npz file')
    parser.add_argument('--input', type=str, required=True,
                        help='name of the input tensor in the npz (X)')
    parser.add_argument('--label', type=str, required=True,
                        help='name of the target label tensor in the npz (Y)')
    # settings
    parser.add_argument('--datalength', type=int, required=True,
                        help='data original data files length')
    parser.add_argument('--matrixsize', type=int, nargs=2, required=True,
                        help='data original data files length')
    parser.add_argument('--out_dir', type=str, default=os.getcwd(),
                        help='output folder of the model')

    # model training options (metaparameters)
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='batch size for training (default: 128)')
    parser.add_argument('--filter-factor', type=int, default=64, metavar='N',
                        help='base number of filters for the encoder/decoder (default: 64)')
    parser.add_argument('--vector-size', type=int, default=16, metavar='N',
                        help='size of the encoded vector (default: 16)')
    parser.add_argument('--loss', type=str, default='mse', metavar='LOSS',
                        choices=LOSSES.keys(),
                        help='loss function for training (default: mse)')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of batches to train (default: 10000)')

    # settings
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1, set 0 for random seed)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--save-interval', type=int, default=2000, metavar='N',
                        help='how many batches to wait before saving model to disk (default: 2000)')
    parser.add_argument('--workers', type=int, default=20, metavar='N',
                        help='number of parallel workers to process input images (default: 20)')
    main(parser.parse_args())


"""
time pipenv run python /amethyst/s0/fbx5002/PythonWorkingDir/DeepLearning/train.py --workers 48 --data /home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage2_BHData/data_256x7.npz --input total --label trans --datalength 13140 --matrixsize 256 7 --out_dir /home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/Stage2_BHData/trainModel
"""
