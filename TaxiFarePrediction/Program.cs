using System;
using System.IO;
using System.Linq; // for .Count for example
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.Transforms.NormalizingEstimator; // for normalizerMode
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using TaxiFarePrediction.Common;

namespace TaxiFarePrediction
{

    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        //static TextLoader _textLoader; not used 

        static void Main(string[] args)
        {
            // STEP 1: Common data loading configuration
            MLContext mlContext = new MLContext(seed: 0);

            IDataView baseTrainingDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_trainDataPath, hasHeader: true, separatorChar: ','); // ReadFromTextFile is deprecated
            IDataView testDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');

            //Sample code of removing extreme data like "outliers" for FareAmounts higher than $150 and lower than $1 which can be error-data 
            var cnt = baseTrainingDataView.GetColumn<float>(mlContext, nameof(TaxiTrip.FareAmount)).Count();
            IDataView trainingDataView = mlContext.Data.FilterRowsByColumn(baseTrainingDataView, nameof(TaxiTrip.FareAmount), lowerBound: 1, upperBound: 150);


            // STEP 2: Common data process configuration with pipeline data transformations
            var dataProcessPipeline = mlContext.Transforms.CopyColumns(outputColumnName: DefaultColumnNames.Label, inputColumnName: nameof(TaxiTrip.FareAmount))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: nameof(TaxiTrip.VendorId)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: nameof(TaxiTrip.RateCode)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: nameof(TaxiTrip.PaymentType)))
                            .Append(mlContext.Transforms.Normalize(outputColumnName: nameof(TaxiTrip.PassengerCount), mode: NormalizerMode.MeanVariance))
                            .Append(mlContext.Transforms.Normalize(outputColumnName: nameof(TaxiTrip.TripTime), mode: NormalizerMode.MeanVariance))
                            .Append(mlContext.Transforms.Normalize(outputColumnName: nameof(TaxiTrip.TripDistance), mode: NormalizerMode.MeanVariance))
                            .Append(mlContext.Transforms.Concatenate(DefaultColumnNames.Features, "VendorIdEncoded", "RateCodeEncoded", "PaymentTypeEncoded", nameof(TaxiTrip.PassengerCount)
                            , nameof(TaxiTrip.TripTime), nameof(TaxiTrip.TripDistance)));


            // STEP 3: Set the training algorithm, then create and config the modelBuilder - Selected Trainer (SDCA Regression algorithm)                            
            var trainer = mlContext.Regression.Trainers.StochasticDualCoordinateAscent(labelColumnName: DefaultColumnNames.Label, featureColumnName: DefaultColumnNames.Features);
            var trainingPipeline = dataProcessPipeline.Append(trainer);


            //STEP 4: Train the model fitting to the DataSet
            //The pipeline is trained on the dataset that has been loaded and transformed.
            ITransformer trainedModel = trainingPipeline.Fit(trainingDataView); // changed var trainedModel to ITransformer trainedModel


            // STEP 5: Evaluate the model and show accuracy stats
            // create a copy of the testDataView in IDataView format since IDataView is immutable?
            IDataView predictions = trainedModel.Transform(testDataView); 
            // assign label column (original regression value) and score column (predictive regression values) to columns in data i.e. predictions
            // apply metrics e.g. root mean squared error on data
            var metrics = mlContext.Regression.Evaluate(predictions, label: DefaultColumnNames.Label, score: DefaultColumnNames.Score); 

            // print the results of the metrics
            Common.ConsoleHelper.PrintRegressionMetrics(trainer.ToString(), metrics);


            // STEP 6: Save/persist the trained model to a .ZIP file
            using (var fs = File.Create(_modelPath))
                trainedModel.SaveTo(mlContext, fs);

            Console.WriteLine("The model is saved to {0}", _modelPath);        


            // STEP 7: Prediction
            //Sample: 
            //vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
            //VTS,1,1,1140,3.75,CRD,15.5

            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };


            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                trainedModel = mlContext.Model.Load(stream);
            }

            // Create prediction engine related to the loaded trained model
            var predEngine = trainedModel.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(mlContext);

            //Score
            var resultprediction = predEngine.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {resultprediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");

        }


    }
}
