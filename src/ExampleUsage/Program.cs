using System;
using System.Linq;
using CSharpNeuralNetworkExample;

namespace ExampleUsage
{
    class Program
    {
        static void Main(string[] args)
        {
            MNISTTest();

            MicrosoftExample();
        }

        static void MNISTTest()
        {
            // before using this method download the mnist data set and put it into the folder specified in MNISTReader.cs

            Random rand = new Random(100);

            NeuralNetwork mnistNetwork = new NeuralNetwork(
                sizes: new int[] { 784, 100, 10 },
                activations: new Activation[] { Activation.Sigmoid, Activation.Sigmoid },
                costFunction: Cost.CrossEntropyCost,
                rand: rand);

            TrainTestImage[] trainData = MnistReader.ReadTrainingData().ToArray();
            TrainTestImage[] testData = MnistReader.ReadTestData().ToArray();

            mnistNetwork.FeedForward(trainData[0].Input);

            mnistNetwork.StochasticGradientDescent(trainData, 30, 50, 0.05, 0.005, testData);

            mnistNetwork.Evaluate(testData);
        }

        static void MicrosoftExample()
        {
            Random rand = new Random(100);

            NeuralNetwork testNetwork = new NeuralNetwork(
                sizes: new int[] { 3, 4, 2 },
                activations: new Activation[] { Activation.Sigmoid, Activation.HyperTan },
                costFunction: Cost.QuadraticCost,
                rand: rand);

            var trainSet = MicrosoftExampleTestSet();

            // Hidden Layer
            // Neuron 0
            testNetwork.WorkingLayers[0].Neurons[0].Bias = -2.0;
            testNetwork.WorkingLayers[0].Neurons[0].Weights[0] = 0.1;
            testNetwork.WorkingLayers[0].Neurons[0].Weights[1] = 0.5;
            testNetwork.WorkingLayers[0].Neurons[0].Weights[2] = 0.9;
            // Neuron 1
            testNetwork.WorkingLayers[0].Neurons[1].Bias = -6.0;
            testNetwork.WorkingLayers[0].Neurons[1].Weights[0] = 0.2;
            testNetwork.WorkingLayers[0].Neurons[1].Weights[1] = 0.6;
            testNetwork.WorkingLayers[0].Neurons[1].Weights[2] = 1.0;
            // Neuron 2
            testNetwork.WorkingLayers[0].Neurons[2].Bias = -1.0;
            testNetwork.WorkingLayers[0].Neurons[2].Weights[0] = 0.3;
            testNetwork.WorkingLayers[0].Neurons[2].Weights[1] = 0.7;
            testNetwork.WorkingLayers[0].Neurons[2].Weights[2] = 1.1;
            // Neuron 3
            testNetwork.WorkingLayers[0].Neurons[3].Bias = -7.0;
            testNetwork.WorkingLayers[0].Neurons[3].Weights[0] = 0.4;
            testNetwork.WorkingLayers[0].Neurons[3].Weights[1] = 0.8;
            testNetwork.WorkingLayers[0].Neurons[3].Weights[2] = 1.2;

            // Output Layer
            // Neuron 0
            testNetwork.OutputLayer.Neurons[0].Bias = -2.5;
            testNetwork.OutputLayer.Neurons[0].Weights[0] = 1.3;
            testNetwork.OutputLayer.Neurons[0].Weights[1] = 1.5;
            testNetwork.OutputLayer.Neurons[0].Weights[2] = 1.7;
            testNetwork.OutputLayer.Neurons[0].Weights[3] = 1.9;
            // Neuron 1
            testNetwork.OutputLayer.Neurons[1].Bias = -5.0;
            testNetwork.OutputLayer.Neurons[1].Weights[0] = 1.4;
            testNetwork.OutputLayer.Neurons[1].Weights[1] = 1.6;
            testNetwork.OutputLayer.Neurons[1].Weights[2] = 1.8;
            testNetwork.OutputLayer.Neurons[1].Weights[3] = 2.0;

            testNetwork.FeedForward(trainSet[0].Input);

            testNetwork.OutputLayer.CalculateGradient(trainSet[0].Label);
            testNetwork.WorkingLayers[0].CalculateGradient(testNetwork.OutputLayer);

            Console.WriteLine(testNetwork.ToString());

            Console.ReadLine();

            // not implemented 
        }
        static TrainTestSet[] MicrosoftExampleTestSet()
        {
            TrainTestSet[] set = new TrainTestSet[40];

            for (int i = 0; i < set.Length; i++)
            {
                set[i] = new TrainTestSet() { Input = new double[] { 1.0, 2.0, 3.0 }, Label = new double[] { -0.85, 0.75 } };
            }

            return set;
        }
    }
}
