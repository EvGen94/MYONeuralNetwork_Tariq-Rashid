#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy
import scipy.special
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


class neuralNetwork:
    
    # neural network initialization
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        
        #self.wih = (numpy.random.rand(self.hnodes, self.inodes)- 0.5)
        #self.who = (numpy.random.rand(self.onodes, self.hnodes)- 0.5)
        
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # using the sigmoid as an activation function   
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass
    
    # neural network training
    def train(self, inputs_list, targets_list):
        
        # convert the input list to a two-dimensional array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # calculation of inputs signals for the hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculation of outputs signals for the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) 
        # calculation of inputs signals for the output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)  
        # calculation of outputs signals for the output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error = (target value - actual value) 
        output_errors = targets - final_outputs
        # hidden layer error 
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        # obnovit vesovie kof dlya svyazeiy for hidden and outputs layer
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        
        # obnovit vesovie kof dlya svyazeiy for input and hidden layer
       
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        
        pass
    
    # neural network query():
    def query(self, inputs_list):
        
        # convert the input list to a two-dimensional array
        inputs = numpy.array(inputs_list, ndmin=2).T
        # calculation of inputs signals for the hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)   
        # calculation of outputs signals for the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)   
        # calculation of inputs signals for the output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)       
        # calculation of outputs signals for the output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
        
        pass
    


# In[ ]:


input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)


# In[ ]:


traning_data_file = open("mnist_train.csv", 'r')
traning_data_list = traning_data_file.readlines()
traning_data_file.close()
#traning_data_list[0]


# In[ ]:


# Training !!!
epochs = 2
for e in range(epochs):

    for record in traning_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])/ 255.0 *0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
pass    


# In[ ]:


# Testing !!!
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
test_data_list[0]


# In[ ]:


all_values = test_data_list[0].split(',')
print(all_values[0])


# In[ ]:


image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys',interpolation='None')


# In[ ]:


n.query((numpy.asfarray(all_values[1:])/255.0 * 0.99) +0.01)


# In[ ]:


# Testing in the loop 
scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    #right answer - first value 
    correct_label = int(all_values[0])
    #print(correct_label, "true label")
    inputs = (numpy.asfarray(all_values[1:])/ 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax  (outputs)
    #print(label, "network response") 
    if (label==correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass
        


# In[ ]:


scorecard_array = numpy.asarray(scorecard)
print("efficiency = ",scorecard_array.sum() / scorecard_array.size) # share of correct answers


# In[ ]:


# check out my own numbers 
import scipy.misc
#img_array = imageio.imread("1.png", flatten = True) 

