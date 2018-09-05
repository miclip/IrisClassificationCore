using Microsoft.ML;
using Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.IO;

namespace Prediction
{
   class Program
    {
        static readonly IEnumerable<ClassificationData> predictClassData = new[]
        {
            new ClassificationData
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            }
        };

         static readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Learned", "Model.zip");

        public static void Main(string[] args)
        {
            Task.Run(async () =>
            {
                var model = await PredictAsync(modelPath, predictClassData);

                Console.WriteLine();
                Console.WriteLine("Please enter another string to classify or just <Enter> to exit the program.");
                Console.WriteLine("Sample string (without quotes): '5.7,2.8,4.5,1.3'");

                var input = string.Empty;

                while (string.IsNullOrEmpty(input = Console.ReadLine()) == false)
                {
                    try
                    {
                        var inputObj = readLine(input);

                        IEnumerable<ClassificationData> predictInput = new[]
                        {
                            inputObj
                        };

                        model = await PredictAsync(modelPath, predictInput, model);
                    }
                    catch
                    {
                        Console.WriteLine("Syntax error. Please input a value string...");
                    }
                }

                Console.WriteLine("Press any key to end program...");
                Console.ReadKey();

            }).GetAwaiter().GetResult();
        }

        internal static ClassificationData readLine(string input)
        {
            if (string.IsNullOrEmpty(input) == true)
                return null;

            string[] commaSepList = input.Split(',');

            if (commaSepList == null)
                return null;

            if (commaSepList.Length != 4)
                return null;

            return new ClassificationData
            {
                SepalLength = float.Parse(commaSepList[0]),
                SepalWidth = float.Parse(commaSepList[1]),
                PetalLength = float.Parse(commaSepList[2]),
                PetalWidth = float.Parse(commaSepList[3])
            };
        }

        /// <summary>
        /// Predicts the test data outcomes based on a model that can be
        /// loaded via path or be given via parameter to this method.
        /// 
        /// Creates test data.
        /// Predicts classification based on test data.
        /// Combines test data and predictions for reporting.
        /// Displays the predicted results.
        /// </summary>
        /// <param name="model"></param>
        internal static async Task<PredictionModel<ClassificationData, ClassPrediction>> PredictAsync(
            string modelPath,
            IEnumerable<ClassificationData> predicts = null,
            PredictionModel<ClassificationData, ClassPrediction> model = null)
        {
            if (model == null)
            {
              model = await PredictionModel.ReadAsync<ClassificationData, ClassPrediction>(modelPath);
            }

            if (predicts == null) // do we have input to predict a result?
                return model;

            // Use the model to predict the classification of the data.
            IEnumerable<ClassPrediction> predictions = model.Predict(predicts);

            Console.WriteLine();
            Console.WriteLine("Classification Predictions");
            Console.WriteLine("--------------------------");

            // Builds pairs of (input, prediction)
            IEnumerable<(ClassificationData input, ClassPrediction prediction)> inputsAndPredictions =
                predicts.Zip(predictions, (input, prediction) => (input, prediction));

            foreach (var item in inputsAndPredictions)
            {
                Console.WriteLine("    Petal Length: {0}", item.input.PetalLength);
                Console.WriteLine("    Petal  Width: {0}", item.input.PetalWidth);
                Console.WriteLine("    Sepal Length: {0}", item.input.SepalLength);
                Console.WriteLine("    Sepal  Width: {0}", item.input.SepalWidth);
                Console.WriteLine("Predicted Flower: {0}", item.prediction.Class);
            }
            Console.WriteLine();

            return model;
        }
    }
}
