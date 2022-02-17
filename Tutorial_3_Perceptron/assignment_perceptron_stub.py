import pylab as pl
import scipy as sp
import numpy as np
from scipy.io import loadmat
import pdb


def load_data(fname):
    # load the data
    data = loadmat(fname)
    # extract images and labels
    imgs = data['data_patterns']
    labels = data['data_labels']
    return imgs, labels


	
def perceptron_train(X,Y,Xtest,Ytest,iterations=100,eta=.1):
    # initialize accuracy vector
    acc = 
    # initialize weight vector
    weights = 
    # loop over iterations    
    for it in sp.arange(iterations):
	# find all indices of misclassified data
	wrong = 
        # check if there really are misclassified data
	if wrong.shape[0] > 0:
	    # pick a random misclassified data point
	    rit = 
	    # update weight vector
	    weights = # or 'weights +='
	    # compute accuracy vector
	    acc[it] = 
    # return weight vector and accuracy
    return weights,acc



def digits(digit):
    fname = "usps.mat"
    imgs,labels = load_data(fname)
    # we only want to classify one digit 
    labels = sp.sign((labels[digit,:]>0)-.5)

    # please think about what the next lines do
    permidx = sp.random.permutation(sp.arange(imgs.shape[-1]))
    trainpercent = 70.
    stopat = sp.floor(labels.shape[-1]*trainpercent/100.)
    stopat= int(stopat)

    # cut segment data into train and test set into two non-overlapping sets:
    X = 
    Y = 
    Xtest = 
    Ytest = 
    #check that shapes of X and Y make sense..
    # it might makes sense to print them
    
    w,acc_perceptron = perceptron_train(X,Y,Xtest,Ytest)

    fig = pl.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(acc_perceptron*100.)
    pl.xlabel('Iterations')
    pl.title('Linear Perceptron')
    pl.ylabel('Accuracy [%]')

    # and imshow the weight vector
    ax2 = fig.add_subplot(1,2,2)
    # reshape weight vector
    weights = sp.reshape(w,(int(sp.sqrt(imgs.shape[0])),int(sp.sqrt(imgs.shape[0]))))
    # plot the weight image
    imgh = ax2.imshow(weights)
    # with colorbar
    pl.colorbar(imgh)
    ax2.set_title('Weight vector')
    # remove axis ticks
    pl.xticks(())
    pl.yticks(())
    # remove axis ticks
    pl.xticks(())
    pl.yticks(())

    # write the picture to pdf
    fname = 'Perceptron_digits-%d.pdf'%digit
    pl.savefig(fname)


digits(0)




