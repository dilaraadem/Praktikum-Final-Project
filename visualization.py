import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
tfd = tf.contrib.distributions
import matplotlib.pyplot as plt
import pandas as pd
import random


def print_loss(loss_history):
    for i in range(0, len(loss_history)):
        plt.scatter(i,loss_history[i])
    plt.show()
    
def plot_loss(train_loss):
    # Plotting the loss
    for i in range(0, len(train_loss)):
        plt.plot(train_loss)
        plt.xlabel("epoch")
        plt.ylabel("loss")
    plt.show()
    
    
def plot_latent_space(z, yval, e):
    #plt.scatter(z[:,0], z[:,1], cmap = plt.cm.rainbow, c = yval)
    plt.figure(figsize=(15,10))
    plt.scatter(z[:,0], z[:,1], cmap = plt.cm.rainbow, c = yval)
    #for i in range(0,z.shape[0]):
        #r = random.randint(0,z.shape[0]-1)
        #plt.annotate(yval[i][0], xy = (z[i,0], z[i,1]), xytext = (z[i,0], z[i,1]))
    plt.colorbar()
    plt.title("Epoch: "+str(e))
    plt.show()
    
def visualize_latent_space(z, yval, e):
    #plt.scatter(z[:,0], z[:,1], cmap = plt.cm.rainbow, c = yval)
    plt.figure(figsize=(15,10))
    plt.scatter(z[:,0], z[:,1])
    for i in range(0,z.shape[0]):
        #r = random.randint(0,z.shape[0]-1)
        plt.annotate(yval[i][0], xy = (z[i,0], z[i,1]), xytext = (z[i,0], z[i,1]))
    #plt.colorbar()
    plt.title("Epoch: "+str(e))
    plt.show()
    
def print_comparison_test(data, o, real_dimension):
    for i in range(0,15):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2,4))
        ax1.set_title('original')
        ax1.imshow(np.reshape(data[i], real_dimension), cmap='gray')
        o1 = o[i].reshape(real_dimension)
        ax2.set_title('decoded')
        ax2.imshow(o1, cmap='gray')
        plt.show()
           
        
def visualize_generated(imgs, e, real_dimension):
    images = []
    for i in range(len(imgs)):
        images.append(np.reshape(imgs[i], real_dimension))
    i = 0
    fig, ax = plt.subplots(1,15, figsize=(15,25))
    for img in images:
        if(i<15):
            ax[i].imshow(img, cmap='gray')
        i = i + 1
    plt.title("Epoch: "+str(e))
    plt.show()
    
    
""" Parameters
X_dmap: data embedding to be visualized
bl: string labels
labels: integer labels for coloring
text_data: boolean parameter specified for Word2Vec data visualization
text_labels: optional,
p: optional, integer. Used as modulo for adjusting number of annotations
figsize: optional figure size of plot. Default is assigned.
name: string, name of the file for plot to be saved
"""
def plot_embedding(X_dmap, bl, labels, text_data, text_labels=[], p=1, figsize=(16,10), name='embed_fig'):
    x = []
    y = []
    for value in X_dmap:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=figsize)
    plt.scatter(x,y, cmap = plt.cm.rainbow, c = labels) # scatter points with corresponding color labels
    if text_data==True: # condition for annotation for Word2Vec dataset
        colors = []
        for i in range(len(x)):
            if (i+1)%p==0: # divide to p to get class label annotations
                plt.annotate(bl[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
                colors.append(bl[i]) # store the string to use as colorbar label
        
    color_list = np.arange(min(labels), max(labels)+1).tolist()
    cbar = plt.colorbar() # plot colorbar
    cbar.set_ticks(color_list)
    
    if text_data==True:
        cbar.set_ticklabels(colors)
    else:
        cbar.set_ticklabels(text_labels)
        
    plt.savefig(str(name)+'.png') # save image
    plt.show()