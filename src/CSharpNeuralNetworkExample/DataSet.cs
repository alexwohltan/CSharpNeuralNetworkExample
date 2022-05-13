using System;
using System.Text;

namespace CSharpNeuralNetworkExample
{
    public class TrainTestSet
    {
        public virtual double[] Input { get; set; }
        public virtual double[] Label { get; set; }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("( ");
            sb.Append(Input[0]);

            if(Input.Length > 1)
                for (int i = 1; i < Input.Length; i++)
                {
                    sb.Append(" | ");
                    sb.Append(Input[i]);
                }

            sb.Append(" ) -> ( ");
            sb.Append(Label[0]);

            if (Label.Length > 1)
                for (int i = 1; i < Label.Length; i++)
                {
                    sb.Append(" | ");
                    sb.Append(Label[i]);
                }

            sb.Append(" )");
            return sb.ToString();
        }
    }

    public class TrainTestImage : TrainTestSet
    {
        public byte NumberInImage { get; set; }
        public byte[,] Pixels { get; set; }

        public override double[] Input
        {
            get
            {
                // Step 1: get total size of 2D array, and allocate 1D array.
                int size = Pixels.Length;
                double[] result = new double[size];

                // Step 2: copy 2D array elements into a 1D array.
                int write = 0;
                for (int i = 0; i <= Pixels.GetUpperBound(0); i++)
                {
                    for (int z = 0; z <= Pixels.GetUpperBound(1); z++)
                    {
                        result[write++] = Pixels[i, z];
                    }
                }
                // Step 3: return the new array.
                return result;
            }
        }
        public override double[] Label
        {
            get
            {
                double[] result = new double[10];
                result[NumberInImage] = 1;
                return result;
            }
        }
    }

    public class TrainTestStairs : TrainTestSet
    {
        public int ImageId { get; set; }
        public double R0C0 { get; set; }
        public double R0C1 { get; set; }
        public double R1C0 { get; set; }
        public double R1C1 { get; set; }
        public bool IsStairs { get; set; }

        public override double[] Input => new double[] { R0C0, R0C1, R1C0, R1C1 };
        public override double[] Label => IsStairs ? new double[] { 1.0, 0.0 } : new double[] { 0.0, 1.0 };

        public TrainTestStairs(int id, double r0c0, double r0c1, double r1c0, double r1c1, bool isStairs)
        {
            ImageId = id;
            R0C0 = r0c0;
            R0C1 = r0c1;
            R1C0 = r1c0;
            R1C1 = r1c1;
            IsStairs = isStairs;
        }
    }
}
