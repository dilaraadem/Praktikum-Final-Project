import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
tfd = tf.contrib.distributions
import matplotlib.pyplot as plt
import pandas as pd
import gensim
from skimage.transform import resize

# create batch
def create_batch(train_x, train_y, batch_size):
    batches = []
    batches_y = []
    index_list = np.arange(train_x.shape[0])
    np.random.shuffle(index_list)
    
    batch_nr = int(train_x.shape[0]/batch_size)
    for i in range(0, batch_nr):
        loc = i * batch_size
        batch_indices = index_list[loc:loc+batch_size]
        batch = train_x[batch_indices]
        batch_y = train_y[batch_indices]
        batches.append(batch)
        batches_y.append(batch_y)
    return batches, batches_y

# create test batch
def create_batch_test(test_x, test_y, batch_size):
    batches = []
    batches_y = []
    index_list = np.arange(test_x.shape[0])
    np.random.shuffle(index_list)
    
    batch_nr = int(test_x.shape[0]/batch_size)
    for i in range(0, batch_nr):
        loc = i * batch_size
        batch_indices = index_list[loc:loc+batch_size]
        batch_x = test_x[batch_indices]
        batch_y = test_y[batch_indices]
        batches.append(batch_x)
        batches_y.append(batch_y)
    
    return batches[0], batches_y[0]

# parameters: arrays of training set and test set
def resize_images(train_x, test_x):
    # initialize the new sized arrays
    return_train = np.zeros((train_x.shape[0], 20*20*3))
    return_test = np.zeros((test_x.shape[0], 20*20*3))
    
    # reshape the arrays and store in initialized arrays
    for i in range(0, train_x.shape[0]):
        img_resized = resize(train_x[i].reshape((32,32,3)), (20, 20, 3), anti_aliasing = True)
        return_train[i,:] = img_resized.reshape(-1, 20*20*3)
    for i in range(0, test_x.shape[0]):
        img_resized = resize(test_x[i].reshape((32,32,3)), (20, 20, 3), anti_aliasing = True)
        return_test[i,:] = img_resized.reshape(-1, 20*20*3)
        
    return return_train, return_test

## WORD2VEC FUNCTIONS BELOW

# parameters: array of words, size of the list to be generated batch_size, Word2Vec model
# returns: batch_list list of string labels, vector_list vector embeddings, label_list list of integers used in colored visualization
def create_corpus(word_array, batch_size, model):
    # initialize
    batch_list = []
    vector_list = []
    label_list = []
    j = 0
    for i in word_array:
        similar = model.most_similar(i) # get the most similar 10 words of the selected word
        sim = similar
        for s in sim:
            word = s[0]
            vecc = model.word_vec(word) # get word vector embedding
            batch_list.append(word) # store name of word as label
            vector_list.append(vecc) # store embedding
            label_list.append(j) # store number of word as color label (to be used in visualization)
        batch_list.append(i)
        vec = model.word_vec(i)
        label_list.append(j)
        vector_list.append(vec)
        j = j+1
        
    # limit the size of arrays if necessary and convert to numpy for easier processing
    batch_list = np.asarray(batch_list, dtype='object')[0:batch_size]
    vector_list = np.asarray(vector_list)[0:batch_size,:]
    label_list = np.asarray(label_list)[0:batch_size]
    vector_list = (vector_list - np.min(vector_list)) / (np.max(vector_list) - np.min(vector_list)) # normalize, just in case
    
    return batch_list, vector_list, label_list

# calculates similarity matrix of the selected words in and array
# parameters: array of strings word_array, Word2Vec model 
def similarity_matrix(word_array, model):
    sim_mat = np.zeros(shape=(word_array.shape[0], word_array.shape[0])) # empty matrix of size word array by word array
    for i in range(0, word_array.shape[0]):
        for j in range(0, word_array.shape[0]):
            sim_mat[i, j] = model.similarity(word_array[i], word_array[j]) # use built-in function similarity
    return sim_mat # return matrix

def create_embedding_matrix(model):
    # Create the embedding matrix where words are indexed alphabetically
    vec_size = model.vector_size
    vocab_size = len(model.vocab)
    # initialize matrix and labels
    embedding_matrix = np.zeros(shape=(vocab_size, vec_size), dtype='int32')
    labels = np.zeros(shape=(vocab_size, 1), dtype='object')
    for idx, word in enumerate(sorted(model.vocab)):
        embedding_matrix[idx] = model.get_vector(word) # store the vector embedding
        labels[idx] = word
    # normalize the embedding
    embedding_mat = (embedding_matrix - np.min(embedding_matrix)) / (np.max(embedding_matrix) - np.min(embedding_matrix))
    
    return embedding_mat, labels # returns embbedding matrix and label array
    
