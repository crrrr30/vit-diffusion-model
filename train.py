import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import os, glob, argparse
import numpy as np

from data_loader import get_train_dataloader
from model import ViT
from solver import Solver
from optimizers import Ranger


def parse_args():

    parser = argparse.ArgumentParser(description = 'Train ViT-based diffusion model for image synthesis on Celeba 64x64.')
    
    parser.add_argument('--num-iters', help = 'Number of thousand iterations to train the model for', default = 200, type = int)
    parser.add_argument('--timesteps', help = 'Number of timesteps for the Markov Chain', default = 1000, type = int)
    parser.add_argument('--batch-size', help = 'Batch size', default = 64, type = int)
    parser.add_argument('--log', help = 'Directory to save logs to', default = 'logs', type = str)
    parser.add_argument('--log-interval', help = 'Frequency of logging, in thousand iterations', default = 0.1, type = str)
    parser.add_argument('--checkpoint-interval', help = 'Frequency of saving checkpoints, in thousand iterations', default = 1, type = str)
    parser.add_argument('--gpu', help = 'Specify which GPU to use (separate with commas). -1 means to use CPU', default = None, type = str)
    parser.add_argument('--resume', help = 'Directory of the checkpoint to resume from or the path to the checkpoint', default = None, type = str)
    
    return parser.parse_args()


if __name__ == '__main__':

    start_iter = 0

    args = parse_args()
    
    if args.gpu is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    print('==> Prepping data...')
    train_dataloader = get_train_dataloader(args.batch_size)

    print('==> Building Network...')
    model = ViT(
        image_size = 64,
        patch_size = 16,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 1024,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    model.to(device)

    optimizer = Ranger(model.parameters())

    if args.resume:

        if os.path.isdir(args.resume):
            checkpoint = torch.load(
                sorted(glob.glob(os.path.join(args.resume, 'checkpoint*.pkl'))).pop(),
            map_location = lambda _, __: _)
            print(f'Resuming from iteration {checkpoint["iter"] + 1}...')
            start_iter = checkpoint["iter"] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        elif os.path.isdir(args.resume):
            checkpoint = torch.load(
                sorted(glob.glob(os.path.join(args.log, args.resume, 'checkpoint*.pkl'))).pop(),
            map_location = lambda _, __: _)
            print(f'Resuming from iteration {checkpoint["iter"] + 1}...')
            start_iter = checkpoint["iter"] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        else:
            raise FileNotFoundError(f'Checkpoint not found at {args.resume}!')  

    print(f'Number of total params: {sum([np.prod(p.shape) for p in model.parameters()])}')

    if start_iter >= args.num_iters: 
        print('The model has already been trained for the number of iters specified.')

    if not os.path.exists(args.log):
        os.makedirs(args.log)
    index = 0 if len(os.listdir(args.log)) == 0 else int(sorted(os.listdir(args.log)).pop()[:4]) + 1
    args.log = os.path.join(args.log, '%.4d-train' % index)
    os.makedirs(args.log)
    print(f'==> Saving logs to {args.log}')

    solver = Solver(
        model = model, hyperparams = {'timesteps': args.timesteps}, optimizer = optimizer,
        start_iter = start_iter, num_iters = args.num_iters, device = device, 
        log_dir = args.log, log_interval = args.log_interval, checkpoint_interval = args.checkpoint_interval,
    )
    
    print(f'==> Training for {args.num_iters} iterations...')
    solver.train(train_dataloader)