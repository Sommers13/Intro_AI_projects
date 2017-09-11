# ALLISON SOMMERS

import copy
import sys
from datetime import datetime
from math import exp
from random import random, randint, choice

class Perceptron(object):
    """
    Class to represent a single Perceptron in the net.
    """
    def __init__(self, inSize=1, weights=None):
        self.inSize = inSize+1#number of perceptrons feeding into this one; add one for bias
        if weights is None:
            #weights of previous layers into this one, random if passed in as None
            self.weights = [1.0]*self.inSize
            self.setRandomWeights()
        else:
            self.weights = weights

    def getWeightedSum(self, inActs):
        """
        Returns the sum of the input weighted by the weights.

        Inputs:
            inActs (list<float/int>): input values, same as length as inSize
        Returns:
            float
            The weighted sum
        """
        return sum([inAct*inWt for inAct,inWt in zip(inActs,self.weights)])

    def sigmoid(self, value):
        """
        Return the value of a sigmoid function.

        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the sigmoid function parametrized by
            the value.
        """
        # Q1 CODEES ----------------------------------------------------

        return ( 1 / ( 1 + (exp(0 - value))))

        # Q1 CODEES ----------------------------------------------------

    def sigmoidActivation(self, inActs):
        """
        Returns the activation value of this Perceptron with the given input.
        Same as g(z) in book.
        Remember to add 1 to the start of inActs for the bias input.

        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            float
            The value of the sigmoid of the weighted input
        """
        # Q1 CODEES ----------------------------------------------------

        inputbias = copy.deepcopy(inActs)
        inputbias.insert(0,1)

        return self.sigmoid(self.getWeightedSum(inputbias))

        # Q1 CODEES ----------------------------------------------------


    def sigmoidDeriv(self, value):
        """
        Return the value of the derivative of a sigmoid function.

        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the derivative of a sigmoid function
            parametrized by the value.
        """

        # Q2 CODEES ----------------------------------------------------

        return (exp(value) / ( (exp(value) + 1) * (exp(value) + 1) ) )

        # Q2 CODEES ----------------------------------------------------

    def sigmoidActivationDeriv(self, inActs):
        """
        Returns the derivative of the activation of this Perceptron with the
        given input. Same as g'(z) in book (note that this is not rounded.
        Remember to add 1 to the start of inActs for the bias input.

        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            int
            The derivative of the sigmoid of the weighted input
        """

        # Q2 CODEES ----------------------------------------------------

        inputbias = copy.deepcopy(inActs)
        inputbias.insert(0, 1)

        #return self.sigmoid(self.getWeightedSum(inputbias))
        return self.sigmoidDeriv(self.getWeightedSum(inputbias))

        # Q2 CODEES ----------------------------------------------------

    def updateWeights(self, inActs, alpha, delta):
        """
        Updates the weights for this Perceptron given the input delta.
        Remember to add 1 to the start of inActs for the bias input.

        Inputs:
            inActs (list<float/int>): input values, not including bias
            alpha (float): The learning rate
            delta (float): If this is an output, then g'(z)*error
                           If this is a hidden unit, then the as defined-
                           g'(z)*sum over weight*delta for the next layer
        Returns:
            float
            Return the total modification of all the weights (sum of each abs(modification))
        """
        totalModification = 0

        # Q2 CODEES ----------------------------------------------------

        updates = copy.deepcopy(inActs)
        updates.insert(0, 1)

        i = 0

        # here the cs student wonders why python cannot be bothered to have a standard for loop
        while (i < len(self.weights)):
            #self.weights[i] = abs(updates[i] * alpha * delta)
            self.weights[i] = self.weights[i] + (updates[i] * (alpha * delta))

            totalModification = totalModification + abs(updates[i] * (alpha * delta))

            i = i + 1

        return totalModification

        # Q2 CODEES ----------------------------------------------------

    def setRandomWeights(self):
        """
        Generates random input weights that vary from -1.0 to 1.0
        """
        for i in range(self.inSize):
            self.weights[i] = (random() + .0001) * (choice([-1,1]))

    def __str__(self):
        """ toString """
        outStr = ''
        outStr += 'Perceptron with %d inputs\n'%self.inSize
        outStr += 'Node input weights %s\n'%str(self.weights)
        return outStr

class NeuralNet(object):
    """
    Class to hold the net of perceptrons and implement functions for it.
    """
    def __init__(self, layerSize):#default 3 layer, 1 percep per layer
        """
        Initiates the NN with the given sizes.

        Args:
            layerSize (list<int>): the number of perceptrons in each layer
        """
        self.layerSize = layerSize #Holds number of inputs and percepetrons in each layer
        self.outputLayer = []
        self.numHiddenLayers = len(layerSize)-2
        self.hiddenLayers = [[] for x in range(self.numHiddenLayers)]
        self.numLayers =  self.numHiddenLayers+1

        #build hidden layer(s)
        for h in range(self.numHiddenLayers):
            for p in range(layerSize[h+1]):
                percep = Perceptron(layerSize[h]) # num of perceps feeding into this one
                self.hiddenLayers[h].append(percep)

        #build output layer
        for i in range(layerSize[-1]):
            percep = Perceptron(layerSize[-2]) # num of perceps feeding into this one
            self.outputLayer.append(percep)

        #build layers list that holds all layers in order - use this structure
        # to implement back propagation
        self.layers = [self.hiddenLayers[h] for h in xrange(self.numHiddenLayers)] + [self.outputLayer]

    def __str__(self):
        """toString"""
        outStr = ''
        outStr +='\n'
        for hiddenIndex in range(self.numHiddenLayers):
            outStr += '\nHidden Layer #%d'%hiddenIndex
            for index in range(len(self.hiddenLayers[hiddenIndex])):
                outStr += 'Percep #%d: %s'%(index,str(self.hiddenLayers[hiddenIndex][index]))
            outStr +='\n'
        for i in range(len(self.outputLayer)):
            outStr += 'Output Percep #%d:%s'%(i,str(self.outputLayer[i]))
        return outStr

    def feedForward(self, inActs):
        """
        Propagate input vector forward to calculate outputs.

        Args:
            inActs (list<float>): the input to the NN (an example)
        Returns:
            list<list<float/int>>
            A list of lists. The first list is the input list, and the others are
            lists of the output values of all perceptrons in each layer.
        """

        # Q1 CODESES --------------------------------------------------------------

        output = []
        #output = inActs[:]
        output.append(inActs)

        #running = 0
        i = 0

        for layer in self.hiddenLayers:

            # i = i +1

            val = output[i]
            tmp = []

            for l in layer:
                tmp.append(l.sigmoidActivation(val))

            output.append(tmp)
            i = i + 1 # incriment our place holder

        lastval = output[i] # get the value of last i
        tmp2 = []

        for layer in self.outputLayer:
            tmp2.append(layer.sigmoidActivation(lastval))

        output.append(tmp2)

        return output


        # Q1 CODESES --------------------------------------------------------------

    def backPropLearning(self, examples, alpha):
        """
        Run a single iteration of backward propagation learning algorithm.
        See the text and slides for pseudo code.

        Args:
            examples (list<tuple<list<float>,list<float>>>):
              for each tuple first element is input(feature)"vector" (list)
              second element is output "vector" (list)
            alpha (float): the alpha to training with
        Returns
           tuple<float,float>

           A tuple of averageError and averageWeightChange, to be used as stopping conditions.
           averageError is the summed error^2/2 of all examples, divided by numExamples*numOutputs.
           averageWeightChange is the summed absolute weight change of all perceptrons,
           divided by the sum of their input sizes (the average weight change for a single perceptron).
        """
        #keep track of output
        averageError = 0
        averageWeightChange = 0
        numWeights = 0

        for example in examples:#for each example
            #keep track of deltas to use in weight change
            deltas = []
            #Neural net output list
            allLayerOutput = self.feedForward(example[0]) #done
            lastLayerOutput = allLayerOutput[-1]
            #Empty output layer delta list
            outDelta = []
            #iterate through all output layer neurons
            for outputNum in xrange(len(example[1])):
                gPrime = self.outputLayer[outputNum].sigmoidActivationDeriv(allLayerOutput[-2]) #done
                error = (example[1][outputNum] - lastLayerOutput[outputNum]) #done
                delta = (error * gPrime) #done
                averageError+=error*error/2
                outDelta.append(delta)
            deltas.append(outDelta)

            """
            Backpropagate through all hidden layers, calculating and storing
            the deltas for each perceptron layer.
            """
            for layerNum in xrange(self.numHiddenLayers-1,-1,-1):
                layer = self.layers[layerNum]
                nextLayer = self.layers[layerNum+1]
                hiddenDelta = []
                #Iterate through all neurons in this layer
                for neuronNum in xrange(len(layer)):
                    gPrime = layer[neuronNum].sigmoidActivationDeriv(allLayerOutput[layerNum]) #done

                    #delta formula here
                    delta = 0
                    testdeltas = deltas[0]
                    for i, nextDelta in enumerate(testdeltas):
                        delta = delta + (nextDelta * nextLayer[i].weights[neuronNum + 1])

                    # ALMOST FORGOT THIS MOFO
                    delta = delta * gPrime
                    # end delta formula

                    #hiddenDelta.add the thing
                    hiddenDelta.append(delta)
                deltas = [hiddenDelta]+deltas
            """Get output of all layers"""

            """
            Having aggregated all deltas, update the weights of the
            hidden and output layers accordingly.
            """
            for numLayer in xrange(0,self.numLayers):
                layer = self.layers[numLayer]
                for numNeuron in xrange(len(layer)):
                    weightMod = layer[numNeuron].updateWeights(allLayerOutput[numLayer], alpha, deltas[numLayer][numNeuron]) #done
                    averageWeightChange += weightMod
                    numWeights += layer[numNeuron].inSize
            #end for each example
        #calculate final output
        averageError /= (len(examples)*len(examples[0][1]))             #number of examples x length of output vector
        averageWeightChange/=(numWeights)
        return averageError, averageWeightChange

def buildNeuralNet(examples, alpha=0.1, weightChangeThreshold = 0.00008,hiddenLayerList = [1], maxItr = sys.maxint, startNNet = None):
    """
    Train a neural net for the given input.

    Args:
        examples (tuple<list<tuple<list,list>>,
                        list<tuple<list,list>>>): A tuple of training and test examples
        alpha (float): the alpha to train with
        weightChangeThreshold (float):           The threshold to stop training at
        maxItr (int):                            Maximum number of iterations to run
        hiddenLayerList (list<int>):             The list of numbers of Perceptrons
                                                 for the hidden layer(s).
        startNNet (NeuralNet):                   A NeuralNet to train, or none if a new NeuralNet
                                                 can be trained from random weights.
    Returns
       tuple<NeuralNet,float>

       A tuple of the trained Neural Network and the accuracy that it achieved
       once the weight modification reached the threshold, or the iteration
       exceeds the maximum iteration.
    """
    examplesTrain,examplesTest = examples
    numIn = len(examplesTrain[0][0])
    numOut = len(examplesTest[0][1])
    time = datetime.now().time()
    if startNNet is not None:
        hiddenLayerList = [len(layer) for layer in startNNet.hiddenLayers]
    print "Starting training at time %s with %d inputs, %d outputs, %s hidden layers, size of training set %d, and size of test set %d"\
                                                    %(str(time),numIn,numOut,str(hiddenLayerList),len(examplesTrain),len(examplesTest))
    layerList = [numIn]+hiddenLayerList+[numOut]
    nnet = NeuralNet(layerList)
    if startNNet is not None:
        nnet =startNNet

    iteration = 0
    trainError = 0
    weightMod = sys.maxint

    # Q5 POWER THROUGH ------------------------------------------------------------------

    i = 0

    # again python y u no have regular for loop
    while ( (iteration <= maxItr) and (abs(weightMod) >= weightChangeThreshold) ):
        # get the tuple from back
        trainError, weightMod = nnet.backPropLearning(examplesTrain, alpha)

        iteration = iteration +1

    # Q5 POWER THROUGH ------------------------------------------------------------------

    """
    Iterate for as long as it takes to reach weight modification threshold
    """
        #if iteration%10==0:
        #    print '! on iteration %d; training error %f and weight change %f'%(iteration,trainError,weightMod)
        #else :
        #    print '.',


    time = datetime.now().time()
    print 'Finished after %d iterations at time %s with training error %f and weight change %f'%(iteration,str(time),trainError,weightMod)

    """
    Get the accuracy of your Neural Network on the test examples.
	For each text example, you should first feedforward to get the NN outputs. Then, round the list of outputs from the output layer of the neural net.
	If the entire rounded list from the NN matches with the known list from the test example, then add to testCorrect, else add to  testError.
    """

    testError = 0
    testCorrect = 0

    flag = 0

    for squishbutt in examplesTest:
        flag = flag + 1
        trump = map(lambda x: round(x), nnet.feedForward(squishbutt[0])[-1])

        if trump == squishbutt[1]:
            testCorrect += 1

        else:
            testError += 1

        #print(flag,testCorrect,testError)

    #testAccuracy=0#num correct/num total

    testAccuracy = float (testCorrect) / ( testCorrect + testError)

    print 'Feed Forward Test correctly classified %d, incorrectly classified %d, test percent error  %f\n'%(testCorrect,testError,testAccuracy)

    """return something"""
    return (nnet, testAccuracy)


