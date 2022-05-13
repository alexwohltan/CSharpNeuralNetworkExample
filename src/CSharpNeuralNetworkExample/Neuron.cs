using System;
using System.Text;

namespace CSharpNeuralNetworkExample
{
    public class Neuron
    {
        public double[] Weights { get; set; }
        public double Bias { get; set; }

        public Activation Activation
        {
            get
            {
                return _Activation;
            }
            set
            {
                switch (value)
                {
                    case Activation.Sigmoid:
                        ActivationFunction = SigmoidActivation.Function;
                        ActivationFunctionFirstDerivative = SigmoidActivation.FirstDerivative;
                        break;
                    case Activation.HyperTan:
                        ActivationFunction = HyperTanActivation.Function;
                        ActivationFunctionFirstDerivative = HyperTanActivation.FirstDerivative;
                        break;
                }
                _Activation = value;
            }
        }
        public Cost CostFunction
        {
            get { return _CostFunction; }
            set
            {
                switch (value)
                {
                    case Cost.QuadraticCost:

                        _Cost = QuadraticCost.Function;
                        _CostFirstDerivative = QuadraticCost.FirstDerivative;

                        break;

                    case Cost.CrossEntropyCost:

                        if (Activation != Activation.Sigmoid)
                            throw new Exception("Cannot use CrossEntropyCost without SigmoidFunction");

                        _Cost = CrossEntropyCost.Function;
                        _CostFirstDerivative = CrossEntropyCost.FirstDerivative;

                        break;
                }
                _CostFunction = value;
            }
        }

        #region Private Cost & Activation Functions
        private Activation _Activation;
        private Cost _CostFunction;

        private ActivationFunction ActivationFunction;
        private ActivationFunctionFirstDerivative ActivationFunctionFirstDerivative;

        private CostFunction _Cost;
        private CostFunctionFirstDerivative _CostFirstDerivative;
        #endregion

        public double[] Inputs { get; set; }

        /// <summary>
        /// Output without Activation Function and Bias => x1 * w1 + x2 * w2
        /// </summary>
        public double ScratchSum
        {
            get
            {
                double sum = 0;
                for (int i = 0; i < Inputs.Length; i++)
                {
                    sum += Weights[i] * Inputs[i];
                }
                return sum;
            }
        }
        /// <summary>
        /// Output without Activation Function => x1 * w1 + x2 * w2 + ... + b
        /// </summary>
        public double Z => ScratchSum + Bias;
        /// <summary>
        /// Output => Activation(x1 * w1 + x2 * w2 + ... + b)
        /// </summary>
        public double Output => ActivationFunction(Z);

        public double Gradient { get; set; }

        public double[] DeltaWeights { get; set; } // => in which direction and by how much should we change the weights?
        public double DeltaBias { get; set; } // => in which direction and by how much should we change the bias?

        public int num_Inputs => Weights.Length;

        #region Initializing
        public Neuron()
        {
        }
        public Neuron(double[] weights, double bias)
        {
            Weights = weights;
            Bias = bias;
        }

        public void Initialize(Random rand, int num_inputs, Activation activation = Activation.Sigmoid, Cost costFunction = CSharpNeuralNetworkExample.Cost.QuadraticCost)
        {
            Weights = new double[num_inputs];
            for (int i = 0; i < num_Inputs; i++)
            {
                Weights[i] = RandomGauss(rand);
            }
            Bias = RandomGauss(rand);

            DeltaBias = 0;
            DeltaWeights = new double[num_Inputs];
            Gradient = 0;

            Activation = activation;
            CostFunction = costFunction;
        }
        #endregion

        #region Learning - Calculating Gradients & Storing Deltas to Current Weights + Bias
        public double CalculateGradient(double y)
        {
            Gradient = _CostFirstDerivative(Output, ActivationFunctionFirstDerivative(Z), y);

            AddDeltas();

            return Gradient;
        }

        /// <param name="followingLayer">The Layer after the layer this Neuron sits in. e.g. if this Neuron is in the 2nd layer 'followingLayer' contains the 3rd layer</param>
        /// <param name="indexInLayer">Index of the Neuron in the layer it sits in. e.g. this is the 3rd Neuron in the 2nd layer -> indexInLayer = 3</param>
        public double CalculateGradient(Layer followingLayer, int indexInLayer)
        {
            double gradient = 0;
            for (int i = 0; i < followingLayer.Neurons.Length; i++)
            {
                gradient += followingLayer.Neurons[i].Weights[indexInLayer] * followingLayer.Neurons[i].Gradient;
            }
            Gradient = gradient * ActivationFunctionFirstDerivative(Z);

            AddDeltas();

            return Gradient;
        }

        public void AddDeltas()
        {
            DeltaBias += Gradient;
            for (int i = 0; i < DeltaWeights.Length; i++)
            {
                DeltaWeights[i] += Inputs[i] * Gradient;
            }
        }
        #endregion

        #region Learning - Updating Weights & Bias
        public void UpdateWeights(double learningRate, int miniBatchSize)
        {
            // Update Weights
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = Weights[i] - learningRate * (DeltaWeights[i] / miniBatchSize);
            }

            // Reset Delta
            for (int i = 0; i < DeltaWeights.Length; i++)
            {
                DeltaWeights[i] = 0;
            }
        }
        public void UpdateBias(double learningRate, int miniBatchSize)
        {
            // Update Bias
            Bias = Bias - learningRate * (DeltaBias / miniBatchSize);

            // Reset Delta
            DeltaBias = 0;
        }
        #endregion

        #region Feed Forward - Using the Network
        public double FeedForward(double[] inputs)
        {
            if (inputs.Length != Weights.Length)
                throw new ArgumentException("inputs.Length does not equal Weights.Length");

            Inputs = inputs;

            return Output;
        }
        #endregion

        #region Helper Functions
        static double RandomGauss(Random rand)
        {
            double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)

            return randStdNormal;
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("Formula:");
            for (int i = 0; i < num_Inputs; i++)
            {
                var weight_i_rounded = Math.Round(Weights[i], 2);
                sb.Append(" ");
                sb.Append(weight_i_rounded);
                sb.Append(" * x_");
                sb.Append(i);
                sb.Append(" +");
            }
            sb.Append(" ");
            sb.Append(Math.Round(Bias, 2));
            return sb.ToString();
        }
        #endregion
    }
}
