using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace CSharpNeuralNetworkExample
{
    public class NeuralNetwork
    {
        public Random Rand { get; set; }

        public int[] Sizes { get; set; } // incl. input & output. e.g. if Sizes.Length = 5 -> 1 input layer, 3 hidden layers, 1 output layer
        public int num_Layers => Sizes.Length;

        public Layer[] WorkingLayers { get; set; } // hidden + output layer(s)

        #region Properties that are stored in the individual Layers
        public Layer OutputLayer => WorkingLayers[WorkingLayers.Length - 1];

        public Activation[] ActivationFunctions => WorkingLayers.Select(e => e.ActivationFunction).ToArray();
        public Cost CostFunction => OutputLayer.CostFunction;

        public double[][,] Weights
        {
            get
            {
                return WorkingLayers.Select(e => e.Weights).ToArray();
            }
            set
            {
                for (int i = 0; i < value.Length; i++)
                {
                    for (int j = 0; j < WorkingLayers[i].num_Neurons; j++)
                    {
                        for (int k = 0; k < WorkingLayers[i].num_Inputs; k++)
                        {
                            WorkingLayers[i].Neurons[j].Weights[k] = value[i][j, k];
                        }
                    }
                }
            }
        }
        public double[][] Biases {
            get { return WorkingLayers.Select(e => e.Biases).ToArray(); }
            set
            {
                for (int i = 0; i < value.Length; i++)
                {
                    for (int j = 0; j < value[i].Length; j++)
                    {
                        WorkingLayers[i].Neurons[j].Bias = value[i][j];
                    }
                }
            } }
        #endregion

        #region Initializing
        public NeuralNetwork(int[] sizes, Activation[] activations, Cost costFunction)
            : this(sizes, activations, costFunction, new Random())
        {
            
        }
        public NeuralNetwork(int[] sizes, Activation[] activations, Cost costFunction, Random rand)
        {
            Sizes = sizes;

            WorkingLayers = new Layer[num_Layers - 1];
            for (int i = 0; i < num_Layers - 1; i++)
            {
                WorkingLayers[i] = new Layer();
            }

            Rand = rand;

            Initialize(rand, activations, costFunction);
        }

        public void Initialize(Random rand, Activation[] activationFunctions = null, Cost costFunction = Cost.QuadraticCost)
        {
            for (int i = 0; i < num_Layers - 1; i++)
            {
                WorkingLayers[i].Initialize(rand, Sizes[i], Sizes[i + 1], activationFunctions[i], costFunction);
            }
        }
        #endregion

        #region Feed Forward - Using the Network
        public double[] FeedForward(double[] inputs)
        {
            if (inputs.Length != Sizes[0])
                throw new ArgumentException("incorrect length of inputs");

            double[] arr = inputs;

            for (int i = 0; i < num_Layers - 1; i++)
            {
                arr = WorkingLayers[i].FeedForward(arr);
            }
            return arr;
        }
        #endregion     

        #region Learning - Using Stochastic Gradient Descent
        public void StochasticGradientDescent(TrainTestSet[] trainingData, int epochs, int miniBatchSize, double learningRate, double learningRateEnd = 0, TrainTestSet[] testData = null)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                Rand.Shuffle(trainingData);
                var miniBatches = trainingData.Split(miniBatchSize);

                int miniBatchCounter = 0;
                foreach (var miniBatch in miniBatches)
                {
                    if (miniBatch.Count() <= 0)
                        break;

                    UpdateMiniBatch(miniBatch.ToArray(), learningRate);
                    Debug.WriteLine("Epoch " + epoch + " / Mini-Batch " + miniBatchCounter);
                    miniBatchCounter++;
                }

                Console.WriteLine("Epoch {0} complete. Learning Rate {1}", epoch, learningRate);

                if(learningRateEnd != 0)
                    learningRate = learningRate - (learningRate - learningRateEnd) / epochs;

                if (testData != null)
                {
                    Evaluate(testData);
                }
            }
        }
        

        #region Update Mini Batch
        private void UpdateMiniBatch(TrainTestSet[] miniBatch, double learningRate)
        {
            // Pseudo Code
            // 1. Input a set of training examples
            // 2. For each training example: Set the corresponding input activation, and perform the following steps:
            //      2.1 Feedforward: For each layer compute z and a
            //          z = weights * inputs + biases
            //          a = sigma(z)
            //      2.2 Output Error: Compute the output error in the last layer
            //      2.3 Backpropagate the Error: Compute the output error for the other layers
            // 3. Gradient Descent: Update the weights and biases

            foreach (var trainSet in miniBatch)
            {
                // input into the network
                double[] inputs = trainSet.Input;
                // should be - output
                double[] labels = trainSet.Label;
                // as is - output
                double[] result = FeedForward(inputs);

                // 2.2 Output Error: Compute the output error in the last layer
                OutputLayer.CalculateGradient(labels);

                // 2.3 Backpropagate the Error: Compute the output error for the other layers
                for (int i = WorkingLayers.Length - 2; i >= 0; i--)
                {
                    WorkingLayers[i].CalculateGradient(WorkingLayers[i + 1]); // we will need the gradients from the next layer so we give the next layer as argument.
                }
            }

            // Update Weights & Biases
            foreach (var layer in WorkingLayers)
            {
                layer.UpdateWeigthsAndBiases(learningRate, miniBatch.Length);
            }
        }
        #endregion
        #endregion

        #region Evaluating the Network
        public List<bool> Evaluate(TrainTestSet[] testData)
        {
            List<bool> results = new List<bool>();

            int correctPredictions = 0;

            int counter = 0;
            foreach (var testSet in testData)
            {
                double[] result = FeedForward(testSet.Input);

                double max = result.Max();

                bool networkPredictedCorrectly = true;

                if (OutputLayer.Neurons.Length > 1)
                    for (int i = 0; i < testSet.Label.Length; i++)
                    {
                        if ((result[i] == max && testSet.Label[i] == 0) || (result[i] != max && testSet.Label[i] == 1))
                            networkPredictedCorrectly = false;
                    }
                else
                {
                    if (testSet.Label[0] == 1)
                        networkPredictedCorrectly = result[0] > 0.5;
                    else
                        networkPredictedCorrectly = result[0] <= 0.5;
                }


                results.Add(networkPredictedCorrectly);

                if (networkPredictedCorrectly)
                {
                    Debug.WriteLine("Test #" + counter + " CORRECT Guess");
                    correctPredictions++;
                }
                else
                {
                    Debug.WriteLine("Test #" + counter + "Guessed WRONG");
                }

                counter++;
            }

            Console.WriteLine("{0}/{1} Predictions guessed correctly. Accuracy: {2:P2}", correctPredictions, testData.Length, (double)correctPredictions / (double)testData.Length);

            return results;
        }
        #endregion

        #region Helper Functions
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("Number of Inputs: ");
            sb.AppendLine("" + Sizes[0]);
            sb.AppendLine();

            for (int i = 0; i < WorkingLayers.Length; i++)
            {
                sb.AppendLine(WorkingLayers[i].ToString());
                sb.AppendLine();
            }

            return sb.ToString();
        }
        #endregion
    }

    #region Extensions
    static class ExtensionsNeuralNetwork
    {
        public static void Shuffle<T>(this Random rng, T[] array)
        {
            int n = array.Length;
            while (n > 1)
            {
                int k = rng.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }

        public static IEnumerable<IEnumerable<T>> Split<T>(this T[] arr, int size)
        {
            for (var i = 0; i < arr.Length / size + 1; i++)
            {
                yield return arr.Skip(i * size).Take(size);
            }
        }
    }
    #endregion
}
