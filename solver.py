# TODO - add inference
import io
import os
import tqdm
import logging
import numpy as np

import torch


class OneLineHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write('\u001b[1000D' + msg)
            self.flush()
        except Exception:
            self.handleError(record)


class Solver:

    def __init__(self, model, hyperparams, optimizer, start_iter, num_iters, device, log_dir, log_interval, checkpoint_interval):
        
        self.model = model
        self.optimizer = optimizer
        self.T = hyperparams['timesteps']
        self.beta = np.linspace(0.001, 0.2, self.T)
        self.alpha = 1 - self.beta
        
        self.start_iter = start_iter
        self.num_iters = num_iters

        self.device = device

        self.log_dir = log_dir
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        self.logger.addHandler(logging.StreamHandler())
        file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s")) 
        self.logger.addHandler(file_handler)


    def train(self, train_dataloader):
        
        self.model.train()
        generator = iter(train_dataloader)
        iteration = self.start_iter * 1000
        training_loss = []

        while True:
            try: 
                x0 = next(generator)
                x0 = x0.to(self.device)
                
                alpha_bar = 1

                for t in range(self.T):
                    alpha_bar *= self.alpha[t]
                    self.optimizer.zero_grad()
                    epsilon = torch.randn_like(x0)
                    loss = torch.square(epsilon - self.model(np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * epsilon)).sum()
                    print(loss.item())
                    loss.backward()
                    training_loss.append(loss.item())
                    self.optimizer.step()
                    iteration += 1

                    if iteration % int(self.log_interval * 1000) == 0:
                        self.logger.info(f'Iter {iteration / 1000}k\t\tLoss: {np.mean(training_loss):.3f}\t\t')
                        training_loss = []

                    if iteration % int(self.checkpoint_interval * 1000) == 0:
                        checkpoint = {
                            'iter': iteration,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()
                        }
                        torch.save(checkpoint, os.path.join(self.log_dir, f'checkpoint{int(iteration/1000):04d}.pkl'))

                    if iteration - self.start_iter * 1000 >= self.num_iters * 1000:
                        break

            except StopIteration:
                generator = iter(train_dataloader)
                x0 = next(generator)
    
        self.logger.info(f'Training finished...')
        

    def validate(self, test_dataloader):
        
        self.model.eval()
        test_loss = []; correct = 0; total = 0
        
        with torch.no_grad():     
            for x, y in test_dataloader:
            
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)

                test_loss.append(loss.item())
                _, predicted = y_hat.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

        self.logger.info(f'Testing loss: {np.mean(test_loss):.3f} | Testing acc: {100 * correct / total:.3f}%')
