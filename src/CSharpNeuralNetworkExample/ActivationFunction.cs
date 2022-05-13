using System;
namespace CSharpNeuralNetworkExample
{
    public enum Activation
    {
        Sigmoid,
        HyperTan
    }

    public delegate double ActivationFunction(double z);
    public delegate double ActivationFunctionFirstDerivative(double z);

    public static class SigmoidActivation
    {
        public static double Function(double z)
        {
            return 1.0 / (1.0 + Math.Exp(-z));
        }
        public static double FirstDerivative(double z)
        {
            return Function(z) * (1 - Function(z));
        }
    }

    public static class HyperTanActivation
    {
        public static double Function(double z)
        {
            return Math.Tanh(z);
        }
        public static double FirstDerivative(double z)
        {
            return (1 - Math.Tanh(z)) * (1 + Math.Tanh(z));
        }
    }
}
