using System;
using System.Linq;
using System.Text;

namespace CSharpNeuralNetworkExample
{
    public class Layer
    {
        public Neuron[] Neurons { get; set; }

        #region Properties that are stored in the individual Neurons
        public int num_Neurons => Neurons.Length;
        public int num_Inputs => Neurons.First().num_Inputs;

        public Activation ActivationFunction => Neurons.First().Activation;
        public Cost CostFunction => Neurons.First().CostFunction;

        public double[,] Weights { get
            {
                var w = new double[num_Neurons, num_Inputs];
                for (int j = 0; j < num_Neurons; j++)
                {
                    for (int k = 0; k < num_Inputs; k++)
                    {
                        w[j, k] = Neurons[j].Weights[k];
                    }
                }
                return w;
            } }
        public double[] Biases => Neurons.Select(e => e.Bias).ToArray();

        public double[] Inputs => Neurons[0].Inputs;
        public double[] Outputs => Neurons.Select(e => e.Output).ToArray();
        public double[] Zs => Neurons.Select(e => e.Z).ToArray();

        public double[] Gradients => Neurons.Select(e => e.Gradient).ToArray();

        public double[,] DeltasWeights
        {
            get
            {
                var w = new double[num_Neurons, num_Inputs];
                for (int j = 0; j < num_Neurons; j++)
                {
                    for (int k = 0; k < num_Inputs; k++)
                    {
                        w[j, k] = Neurons[j].DeltaWeights[k];
                    }
                }
                return w;
            }
        }
        public double[] DeltaBiases => Neurons.Select(e => e.DeltaBias).ToArray();
        #endregion

        #region Initializing
        public void Initialize(Random rand, int num_inputs, int num_neurons, Activation activation = Activation.Sigmoid, Cost costFunction = Cost.QuadraticCost)
        {
            Neurons = new Neuron[num_neurons];
            for (int i = 0; i < num_neurons; i++)
            {
                Neurons[i] = new Neuron();
                Neurons[i].Initialize(rand, num_inputs, activation, costFunction);
            }
        }
        #endregion

        #region Feed Forward - Using the Network
        public double[] FeedForward(double[] inputs)
        {
            if (inputs.Length != num_Inputs)
                throw new ArgumentException("inputs.Length not equal to Neurons.num_Inputs");

            var result = new double[num_Neurons];
            for (int i = 0; i < num_Neurons; i++)
            {
                result[i] = Neurons[i].FeedForward(inputs);
            }
            return result;
        }
        #endregion

        #region Learning - Calculating Gradients & Storing Deltas to Current Weights + Biases
        public void CalculateGradient(double[] labels)
        {
            for (int j = 0; j < Neurons.Length; j++)
            {
                Neurons[j].CalculateGradient(labels[j]);
            }
        }
        public void CalculateGradient(Layer nextLayer)
        {
            for (int j = 0; j < Neurons.Length; j++)
            {
                Neurons[j].CalculateGradient(nextLayer, j);
            }
        }
        #endregion

        #region Learning - Updating Weights & Biases
        public void UpdateWeigthsAndBiases(double learningRate, int miniBatchSize)
        {
            foreach (var neuron in Neurons)
            {
                neuron.UpdateWeights(learningRate, miniBatchSize);
                neuron.UpdateBias(learningRate, miniBatchSize);
            }
        }
        #endregion

        #region Helper Functions
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < num_Neurons; j++)
            {
                for (int k = 0; k < num_Inputs; k++)
                {
                    sb.Append(Math.Round(Weights[j, k], 2));
                    sb.Append(" ");
                }
                sb.AppendLine(" b" + Math.Round(Neurons[j].Bias,2));
            }
            return sb.ToString();
        }
        #endregion
    }
}
