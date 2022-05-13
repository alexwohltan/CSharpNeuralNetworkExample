using System;
namespace CSharpNeuralNetworkExample
{
    public enum Cost
    {
        QuadraticCost,
        CrossEntropyCost
    }

    public delegate double CostFunction(double a, double y);
    public delegate double CostFunctionFirstDerivative(double a, double a_derived, double y);

    public static class QuadraticCost
    {
        /// <summary>
        /// Return the cost associated with an output 'a' and the desired output 'y'
        /// </summary>
        /// <param name="a">the output of the network</param>
        /// <param name="y">the desired output</param>
        /// <returns></returns>
        public static double Function(double a, double y)
        {
            return ((a - y) * (a - y)) / 2;
        }
        public static double FirstDerivative(double a, double a_derived, double y)
        {
            return (a - y) * a_derived;
        }
    }

    public static class CrossEntropyCost
    {
        /// <summary>
        /// Return the cost associated with an output 'a' and the desired output 'y'
        /// </summary>
        /// <param name="a">the output of the network</param>
        /// <param name="y">the desired output</param>
        /// <returns></returns>
        public static double Function(double a, double y)
        {
            return -y * Math.Log(a) - (1 - y) * Math.Log(1 - a);
        }
        public static double FirstDerivative(double a, double a_derived, double y)
        {
            return (a - y); // * a_derived / Sigmoid(Z) -> cancels out when using the Sigmoid Function. If another Activation function is used do not use CrossEntropy
        }
    }
}
