using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Hubbup.MikLabelModel;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace CreateMikLabelModel
{
    public static class MLHelper
    {
        public static void BuildAndTrainModel(string inputDataSetPath, string inputTestDataSetPath, string outputModelPath, MyTrainerStrategy selectedStrategy)
        {
            var stopWatch = Stopwatch.StartNew();

            Console.WriteLine($"Reading input TSV {inputDataSetPath}...");

            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 0);

            var columnInference = InferColumns(mlContext, inputDataSetPath);
            try
            {
                // STEP 1: Load the data
                TextLoader textLoader = mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
                var trainingDataView = textLoader.Load(inputDataSetPath);
                var trainingTestDataView = textLoader.Load(inputTestDataSetPath);

                // STEP 2-b: Customize column information returned by InferColumns API
                ColumnInformation columnInformation = columnInference.ColumnInformation;

                columnInformation.IgnoredColumnNames.Remove("Title");
                columnInformation.TextColumnNames.Add("Title");

                // STEP 3: Run an AutoML multiclass classification experiment
                var experimentSettings = new MulticlassExperimentSettings();
                experimentSettings.Trainers.Clear();
                experimentSettings.Trainers.Add(MulticlassClassificationTrainer.SdcaMaximumEntropy); // FastTreeOva
                experimentSettings.MaxExperimentTimeInSeconds = 60;//ExperimentTime;

                var cts = new System.Threading.CancellationTokenSource();
                experimentSettings.CancellationToken = cts.Token;

                experimentSettings.CacheDirectory = new DirectoryInfo(Path.GetTempPath());

                Console.WriteLine($"Running AutoML multiclass classification experiment for {60} seconds...");
                ExperimentResult<MulticlassClassificationMetrics> experimentResult = mlContext.Auto()
                    .CreateMulticlassClassificationExperiment(experimentSettings)
                    .Execute(trainingDataView, "Area", progressHandler: null);

                // Print top models found by AutoML
                Console.WriteLine(Environment.NewLine);
                Console.WriteLine($"num models created: {experimentResult.RunDetails.Count()}");
                PrintTopModels(experimentResult);

                // STEP 4: Evaluate the model and print metrics
                RunDetail<MulticlassClassificationMetrics> bestRun = experimentResult.BestRun;
                ITransformer trainedModel = bestRun.Model;
                EvaluateModelAndPrintMetrics(mlContext, trainedModel, bestRun.TrainerName, trainingTestDataView);

                // STEP 6: Save/persist the trained model to a .ZIP file
                Console.WriteLine($"Saving the model to {outputModelPath}...");
                mlContext.Model.Save(trainedModel, trainingDataView.Schema, outputModelPath);

                stopWatch.Stop();
                Console.WriteLine($"Done creating model in {stopWatch.ElapsedMilliseconds}ms");
            }
            catch (Exception )
            { }
        }

        /// <summary>
        /// Print top models from AutoML experiment.
        /// </summary>
        private static void PrintTopModels(ExperimentResult<MulticlassClassificationMetrics> experimentResult)
        {
            // Get top few runs ranked by accuracy
            var topRuns = experimentResult.RunDetails
                .Where(r => r.ValidationMetrics != null && !double.IsNaN(r.ValidationMetrics.MicroAccuracy))
                .OrderByDescending(r => r.ValidationMetrics.MicroAccuracy).Take(3);

            Console.WriteLine("Top models ranked by accuracy --");
            PrintMulticlassClassificationMetricsHeader();
            for (var i = 0; i < topRuns.Count(); i++)
            {
                var run = topRuns.ElementAt(i);
                PrintIterationMetrics(i + 1, run.TrainerName, run.ValidationMetrics, run.RuntimeInSeconds);
            }
        }

        internal static void PrintIterationMetrics(int iteration, string trainerName, MulticlassClassificationMetrics metrics, double? runtimeInSeconds)
        {
            CreateRow($"{iteration,-4} {trainerName,-35} {metrics?.MicroAccuracy ?? double.NaN,14:F4} {metrics?.MacroAccuracy ?? double.NaN,14:F4} {runtimeInSeconds.Value,9:F1}", Width);
        }

        internal static void PrintMulticlassClassificationMetricsHeader()
        {
            CreateRow($"{"",-4} {"Trainer",-35} {"MicroAccuracy",14} {"MacroAccuracy",14} {"Duration",9}", Width);
        }
        private const int Width = 114;

        private static void CreateRow(string message, int width)
        {
            Console.WriteLine("|" + message.PadRight(width - 2) + "|");
        }

        /// <summary>
        /// Evaluate the model and print metrics.
        /// </summary>
        private static void EvaluateModelAndPrintMetrics(MLContext mlContext, ITransformer model, string trainerName, IDataView dataView)
        {
            Console.WriteLine("===== Evaluating model's accuracy with test data =====");
            var predictions = model.Transform(dataView);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Area", scoreColumnName: "Score");

            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for {trainerName} multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    MacroAccuracy = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    MicroAccuracy = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
            Console.WriteLine($"************************************************************");
        }

        /// <summary>
        /// Infer columns in the dataset with AutoML.
        /// </summary>
        private static ColumnInferenceResults InferColumns(MLContext mlContext, string dataPath)
        {
            //ConsoleHelper.ConsoleWriteHeader("=============== Inferring columns in dataset ===============");
            ColumnInferenceResults columnInference = mlContext.Auto().InferColumns(dataPath, "Area", groupColumns: false);
            //ConsoleHelper.Print(columnInference);
            return columnInference;
        }
    }
}
