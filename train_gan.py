# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:26:05 2018

@author: prodi
"""

import os, time
import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
from tqdm import tqdm
import dataloader
import gan
import utils

def initialize(mean, std, lr):
    
    param_cuda = torch.cuda.is_available()
    
    G = gan.generator(128).cuda() if param_cuda else gan.generator(128)
    D = gan.discriminator(128).cuda() if param_cuda else gan.discriminator(128)
    G.weight_init(mean=mean, std=std)
    D.weight_init(mean=mean, std=std)
    G_opt = optim.Adam(G.parameters(), lr=lr, betas=(.5, .999))
    D_opt = optim.Adam(D.parameters(), lr=lr, betas=(.5, .999))
    
    return G, D, G_opt, D_opt

def epoch_train(D, G, D_opt, G_opt, batch_size, lr, loss_fn, data_dir, train_hist):
    
    param_cuda = torch.cuda.is_available()
    
    train_loader = dataloader.fetch_data(data_dir, batch_size, normalize=True)
    
    #Set the generator and discrimantor to trainning mode
    G.train()
    D.train()
    
    #save discriminator and genrator losses for each epoch
    D_model_losses = []
    G_model_losses = []
    
    with tqdm(total=len(train_loader)) as t:

        for i, x_real in enumerate(train_loader):

            mini_batch_size = x_real.size()[0]
            
            #Trainning the discriminator:
            
            #Input real images
            x_real = x_real.cuda(async=True) if param_cuda else x_real
            #Input fake images
            x_fake = torch.FloatTensor(mini_batch_size, 100, 1, 1).normal_(0, 1)
            x_fake = x_fake.cuda() if param_cuda else x_fake
            #Labels for fake and real images
            y_real = torch.ones(mini_batch_size).cuda() if param_cuda else torch.ones(mini_batch_size)
            y_fake = torch.zeros(mini_batch_size).cuda() if param_cuda else torch.zeros(mini_batch_size)

            #Create Pytorch Variables to use autograd
            x_real, y_real, x_fake, y_fake = Variable(x_real), Variable(y_real), Variable(x_fake), Variable(y_fake)

            #Forward the real input through D and calculate the loss
            D_output = D(x_real).squeeze()
            D_real_loss = loss_fn(D_output, y_real)

            #Forward the fake input from G through D and calculate the loss
            G_output = G(x_fake)
            D_output = D(G_output).squeeze()
            D_fake_loss = loss_fn(D_output, y_fake)

            #D loss is the sum of real + fake losses:
            D_train_loss = D_real_loss + D_fake_loss

            '''
            Here I can put a threshold to make the discriminator learn slower at first,
            only update if loss is bigger thar a threshold
            if D_model_train_loss.data[0] >= 0.1:
            '''
            
            #One step of backward propagation on D
            D.zero_grad()
            D_train_loss.backward()
            D_opt.step()

            #Trainning the generator:

            #create new fake inputs and convert them to Variable to use autograd
            x_fake = torch.randn((mini_batch_size, 100)).view(-1, 100, 1, 1)
            x_fake = x_fake.cuda() if param_cuda else x_fake
            x_fake = Variable(x_fake)

            #Forward fake inputs from G through D and calculate the loss
            G_output = G(x_fake)
            D_output = D(G_output).squeeze()
            G_train_loss = loss_fn(D_output, y_real)

            #One step of backward propagation on G
            G.zero_grad()
            G_train_loss.backward()
            G_opt.step()

            #store losses
            D_model_losses.append(D_train_loss.data[0])
            G_model_losses.append(G_train_loss.data[0])

            #print losses to see progress
            print("D loss: " + str(D_train_loss.data[0]))
            print("G loss: " + str(G_train_loss.data[0]))

            t.update()

        #store the average loss of every epoch
        train_hist['D_model_mean_losses'].append(torch.mean(torch.FloatTensor(D_model_losses)))
        train_hist['G_model_mean_losses'].append(torch.mean(torch.FloatTensor(G_model_losses)))

        #return the losses
        return train_hist
    
def train(epochs, batch_size, lr, loss_fn, data_dir):
    
    param_cuda = torch.cuda.is_available()
    
    #check training starting time
    start_time = time.time()
    
    G, D, G_opt, D_opt = initialize(mean=0.0, std=0.02, lr=lr)
    
    
    #train_hist dict will store the losses of every epoch
    train_hist = {}
    train_hist['D_model_mean_losses'] = []
    train_hist['G_model_mean_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    #folder for saving the images
    if not os.path.isdir('GAN_results'):
        os.mkdir('GAN_results')

    for epoch in range(epochs):

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, epochs))
        epoch_start_time = time.time()

        #One epoch of trainning over all the dataset
        train_hist = epoch_train(G=G, D=D, G_opt=G_opt, D_opt=D_opt, batch_size=batch_size, lr=lr, loss_fn=loss_fn, data_dir=data_dir, train_hist=train_hist)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        #print progress information for every epoch:
        print("iteration number "+str(epoch))
        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(train_hist['D_model_mean_losses'])), torch.mean(torch.FloatTensor(train_hist['G_model_mean_losses']))))

        #Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'D_model_state_dict': D.state_dict(),
                               'G_model_state_dict': G.state_dict(),
                               'D_optim_dict': D_opt.state_dict(),
                               'G_optim_dict': G_opt.state_dict()},
                               is_best=False,
                               checkpoint =  'GAN_results/')

        #Generate and save pictures for every epoch:
        p = 'GAN_results/result_epoch_' + str(epoch + 1) + '.png'
        utils.show_result(param_cuda, G, (epoch+1), p, save=True)

        #add epoch time to the training history
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), epochs, total_ptime))
    print("Training finish!... save learned parameters")

    #plot training history
    utils.show_train_hist(train_hist, save=True, path= 'GAN_results/_train_hist.png')
    
if __name__ == '__main__':

    #training parameters
    batch_size = 128
    lr = 0.0002
    epochs = 10
    data_dir = 'data/'

    #call the loss function defined in gan.py
    loss_fn = gan.loss_fn

    #Set the logger
    utils.set_logger(os.path.join('GAN_results', 'train.log'))

    #Create the input data pipeline
    logging.info("Loading the datasets...")


    #Trainning the model
    logging.info("Starting training for {} epoch(s)".format(epochs))
    train(epochs=epochs, batch_size=batch_size, lr=lr, loss_fn=loss_fn, data_dir=data_dir)
