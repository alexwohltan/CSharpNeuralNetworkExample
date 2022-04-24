# C# Neural Network Example



<div id="top"></div>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">C# Neural Network Example</h3>

  <p align="center">
    Understand how Neural Networks are implemented

  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li>
      <a href="#structure">Structure</a>
      <ul>
        <li><a href="#neuralnetwork">Neural Network</a></li>
        <li><a href="#layer">Layer</a></li>
        <li><a href="#neuron">Neuron</a></li>
      </ul>
    </li>
    <li><a href="#explanation">Explanation</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



## About The Project

I always wanted to develop my own implementation of a neural network to understand how they work and how they learn. There are many assets out there that explain the concept of neural networks and a ton of Python/R implementations. 
As the math is quite complex I wanted to build a neural network from scratch and put it into an object oriented structure, to understand how it really works (without 3rd-Party-Libraries). Most of the python implementations use Numpy and as a C# developer it sometimes takes some googling on what those functions are doing.
Therefore I built this example Neural Network in an object oriented structure. This code is not optimized for speed but rather for understanding the individual steps. 

I recommend going through 1 or 2 tutorials that explain the concept before diving into the code. 

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

This project is built solely with native C#/.Net Framework Code. No extra libraries needed.

<p align="right">(<a href="#top">back to top</a>)</p>



## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

Using the network is quite easy. You need an array containing training data. The training data must inherit the TrainTestSet Class

```cs
    public class TrainTestSet
    {
        public virtual double[] Input { get; set; }
        public virtual double[] Label { get; set; }
    }
```

Then initialize your network:

```cs
    NeuralNetwork stairsnetwork = new NeuralNetwork(
        sizes: new int[] { 4, 10, 2 },
        activations: new Activation[] { Activation.Sigmoid, Activation.Sigmoid },
        costFunction: Cost.QuadraticCost,
        rand: rand);
```

'sizes' defines the shape of the network. In this example you would get a network with 4 input nodes, 1 hidden layer containing 10 neurons and 2 output neurons. Afterwards you specifiy the Activation Function and the Cost Function. If you do not want to think about them just always use Sigmoid and the QuadraticCost Function. (You can also specify a System.Random to use so you can debug easily. This can be left out.)

## Structure

[![Concept Picture][concept-picture]](https://en.wikipedia.org/wiki/Feedforward_neural_network)

The network basically consists of three classes:

1. NeuralNetwork
2. Layer
3. Neuron

### NeuralNetwork

This is the object containing all the other objects. It has 2 arrays

1. The Layers ('WorkingLayers') -> all the layers excluding the input layer.
2. The Structure ('Sizes') -> Tells us the shape of the network. (see Getting Started)

Otherwise it mainly just calls the 3 important methods:

1. Feed Forward -> Using the Network. Gets an input and returns the calculated output.
2. Stochastic Gradient Descent -> Training the network using the Stochastic Gradient Descent Algorithm
3. Evaluate -> Evaluate the network with a test set.

### Layer

Layer does not contain very interesting stuff. It mainly just forwards the calls from the NeuralNetwork-Object to the Neurons.
This class is mainly for understanding purposes. If you would optimize for speed, you would probably remove this class.

### Neuron

This is where the fun is happening. 

![Concept Picture][neuron-concept-picture]

If you watched at least 1 youtube video regarding neural networks you proably will understand this graphic. It's quite straightforward. In the code this is even implemented in the Properties.

```cs
public double ScratchSum { ... } // All the weights multiplied with the corresponding input -> w0 * a0 + w1 * a1 + w2 * a2
public double Z { ... } // ScratchSum + Bias -> w0 * a0 + w1 * a1 + w2 * a2 + bias
public double Output { ... } // ActivationFunction(Z) -> Most of the time Sigmoid is used but I also implemented HyperTan.
```

The interesting part is the Learning using Stochastic Gradient Descent. See 'Explanation' for this.

<p align="right">(<a href="#top">back to top</a>)</p>



## Explanation

Let's dive into the interesting part. The Learning with Stochastic Gradient Descent.

We have different parameters we can use:
- epochs -> How often we want to train the model. One epoch = training the model on the training set once
- miniBatchSize -> We split the train set into smaller sets so it is easier to compute. 
- learningRate -> each miniBatch-calculation gives us a direction in which we should move our weights and biases. The learning rate states how much we want to listen to this change. e.g. the algorithm says we should add +5 to weight0. with a learning rate of 0.1 we add 0.5 to weight0.
- learningRateEnd -> The more we learn the smaller we want our learning rate to be. This would be the learning rate we want in the last epoch and the model gradually lowers the learning rate to this number.


PSEUDO Code:

For each epoch:

1. Split Training Data into mini batches (small chunks of the overall training set)
2. For each miniBatch containing trainSets
   2.1 For each trainSet
      2.1.1 Feed Forward -> For each layer compute the outputs
      2.1.2 Compute the output error in the last layer
      2.1.3 Backpropagate the Error -> Compute the Error for the other layers
   2.2 Update the Weights and Biases



### Split Training Data

Using two Extension methods stolen from Stackoverflow ;)

```cs
    Rand.Shuffle(trainingData);
    var miniBatches = trainingData.Split(miniBatchSize);
```

### Feed Forward

```cs
    // input into the network
    double[] inputs = trainSet.Input;

    // as is - output
    double[] result = FeedForward(inputs);
```

### Compute the Gradient in the Output Layer

We want to calculate by how much we need to change the output of each neuron in the last layer. Afterwards we will add all the changes in the miniBatch together and then apply the average to our weights and biases.

Code in the NeuralNetwork Class
```cs
    // should be - output
    double[] labels = trainSet.Label;
    
    OutputLayer.CalculateGradient(labels);
```

In the Neuron we are calculating the Derivative of the Cost Function.
This looks a little bit complex in the code because of the delegates but in reality it is quite easy.
Here is the 'raw' code for the QuadraticCost & the CrossEntropyCost Function. Note that we only use the Sigmoid Function as Activation Function here. Otherwise the CrossEntropyCostFunction would not work (in my code - a good mathematician surely can make it work)

```cs
// Quadratic Cost Function.
// y -> label
// Z -> w0 * a0 + ... + bias
Gradient = (Output - y) * SigmoidActivation.FirstDerivative(Z);

// Cross Entropy Cost Function.
// y -> label
Gradient = (Output - y); // see Michael Nielsen's ebook (link below) why we the Sigmoid Function cancelles out.
```

After that we add the Gradient to all the other Gradients in the miniBatch. This is done in the 'AddDeltas()' Function.

```cs
    DeltaBias += Gradient;
    for (int i = 0; i < DeltaWeights.Length; i++)
    {
        DeltaWeights[i] += Inputs[i] * Gradient;
    }
```


### Backpropagate the Gradient in the other Layers

Now we want to calculate on how we need to change the other neurons. We will go from right to left (excluding the output layer)

Iterate through all the other layers by starting at the second last layer

```cs
    for (int i = WorkingLayers.Length - 2; i >= 0; i--)
    {
        WorkingLayers[i].CalculateGradient(WorkingLayers[i + 1]); // we will need the gradients from the next layer so we give the next layer as argument.
    }
```
The Gradient for each neuron is calculated by this formula:

![Backpropagation Gradient][backpropagation-gradient]

Here is the C# code:

```cs
    double gradient = 0;
    for (int i = 0; i < followingLayer.Neurons.Length; i++)
    {
        gradient += followingLayer.Neurons[i].Weights[indexInLayer] * followingLayer.Neurons[i].Gradient;
    }
    Gradient = gradient * ActivationFunctionFirstDerivative(Z);
```

Note that we use the Derivative of the Activation Function here. If you use the Sigmoid Function it would look like this:

```cs
   Gradient = gradient * SigmoidActivation.FirstDerivative(Z)
```

Afterwards we add the gradient to the other gradients of the miniBatch again with 'AddDeltas()'

### Update the Weights and Biases

Now we just need to update the weights and the biases.

Weights
```cs
    // Update Weights
    for (int i = 0; i < Weights.Length; i++)
    {
        Weights[i] = Weights[i] - learningRate * (DeltaWeights[i] / miniBatchSize);
    }

    // Dont forget to reset the Gradients Sum
    for (int i = 0; i < DeltaWeights.Length; i++)
    {
        DeltaWeights[i] = 0;
    }
```

Bias
```cs
    // Update Bias
    Bias = Bias - learningRate * (DeltaBias / miniBatchSize);

    // Reset Delta
    DeltaBias = 0;
```

And thats it. :)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Alexander Wohltan - [@alexanderwohltan](https://www.linkedin.com/in/alexanderwohltan/) - alexander@wohltan.at

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

I mainly used these Sources

* [Michael Nielsen's Ebook](http://neuralnetworksanddeeplearning.com/chap1.html)
* [James McCaffrey's Microsoft Blog Post](https://docs.microsoft.com/en-us/archive/msdn-magazine/2012/october/test-run-neural-network-back-propagation-for-programmers)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/alexwohltan/CSharpNeuralNetworkExample.svg?style=for-the-badge
[contributors-url]: https://github.com/alexwohltan/CSharpNeuralNetworkExample/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/alexwohltan/CSharpNeuralNetworkExample.svg?style=for-the-badge
[forks-url]: https://github.com/alexwohltan/CSharpNeuralNetworkExample/network/members
[stars-shield]: https://img.shields.io/github/stars/alexwohltan/CSharpNeuralNetworkExample.svg?style=for-the-badge
[stars-url]: https://github.com/alexwohltan/CSharpNeuralNetworkExample/stargazers
[issues-shield]: https://img.shields.io/github/issues/alexwohltan/CSharpNeuralNetworkExample.svg?style=for-the-badge
[issues-url]: https://github.com/alexwohltan/CSharpNeuralNetworkExample/issues
[license-shield]: https://img.shields.io/github/license/alexwohltan/CSharpNeuralNetworkExample.svg?style=for-the-badge
[license-url]: https://github.com/alexwohltan/CSharpNeuralNetworkExample/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/alexanderwohltan/
[concept-picture]: NeuralNetworkConcept.png
[neuron-concept-picture]: NeuronConcept.png
[backpropagation-gradient]: BackpropagationGradient.png
