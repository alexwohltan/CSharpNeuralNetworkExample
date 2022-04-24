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

[![Concept Picture][concept-picture]](https://en.wikipedia.org/wiki/Feedforward_neural_network)

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

TODO

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
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[concept-picture]: NeuralNetworkConcept.png
[neuron-concept-picture]: NeuronConcept.png
