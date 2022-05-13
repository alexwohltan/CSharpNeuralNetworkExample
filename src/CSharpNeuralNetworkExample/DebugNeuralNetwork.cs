using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace CSharpNeuralNetworkExample
{
    public class DebugNeuralNetwork
    {
        public static List<DebugNeuralNetwork> DebugList { get; set; }

        public TrainTestSet TrainSet { get; set; }
        public double[] InputLastLayer { get; set; }
        public double[] OutputLastLayer { get; set; }
        public double[,] WeightsLastLayer { get; set; }
        public double[] BiasesLastLayer { get; set; }
        public double[] GradientsLastLayer { get; set; }
        public double[] DeltasBiases { get; set; }
        public double[,] DeltasWeights { get; set; }

        public DebugNeuralNetwork(TrainTestSet trainSet, Layer outputLayer)
        {
            TrainSet = trainSet;

            InputLastLayer = outputLayer.Inputs;
            OutputLastLayer = outputLayer.Outputs;
            WeightsLastLayer = outputLayer.Weights;
            BiasesLastLayer = outputLayer.Biases;
            GradientsLastLayer = outputLayer.Gradients;
            DeltasBiases = outputLayer.DeltaBiases;
            DeltasWeights = outputLayer.DeltasWeights;
        }

        public static void ExportDebugList()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("Input, Label, a_0, a_1, output_0, grad_0, bias_0, weights_0_0, weights_0_1, delta_bias_0, delta_w_0_0, delta_w_0_1");
            foreach (var item in DebugList)
            {
                sb.Append("(" + item.TrainSet.Input[0] + "|" + item.TrainSet.Input[1] + "),");
                sb.Append(item.TrainSet.Label[0] + ",");
                sb.Append(item.InputLastLayer[0] + ",");
                sb.Append(item.InputLastLayer[1] + ",");
                sb.Append(item.OutputLastLayer[0] + ",");
                sb.Append(item.GradientsLastLayer[0] + ",");
                sb.Append(item.BiasesLastLayer[0] + ",");
                sb.Append(item.WeightsLastLayer[0,0] + ",");
                sb.Append(item.WeightsLastLayer[0,1] + ",");
                sb.Append(item.DeltasBiases[0] + ",");
                sb.Append(item.DeltasWeights[0,0] + ",");
                sb.AppendLine(item.DeltasWeights[0,1] + "");
            }

            File.WriteAllText("debug.csv", sb.ToString());
        }
    }
}
