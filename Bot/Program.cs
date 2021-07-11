using System;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Bot
{
    class Program
    {
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "..\\Saved Models\\Shallow LSTM");
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            TensorFlowModel tensorFlowModel = mlContext.Model.LoadTensorFlowModel(_modelPath);
            var pipeline = tensorFlowModel.ScoreTensorFlowModel("StatefulPartitionedCall:0", "serving_default_lstm_input:0", false);
        }
    }
}
