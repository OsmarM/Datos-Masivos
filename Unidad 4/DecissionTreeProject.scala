import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler, OneHotEncoder}

// Load the data stored  as a DataFrame.
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")
data.printSchema()
//val dataall = (data.select(data("y").as("label"), $"age",$"job", $"marital",$"education",$"default",$"balance",$"housing",$"loan",$"contact",$"day",$"month",$"duration",$"campaign",$"pdays",$"previous",$"poutcome"))
// Index labels, adding metadata to the label column.
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label").fit(data)
//val labelEncoder = new OneHotEncoder().setInputCol("label").setOutputCol("labelVec")
// Automatically identify categorical features, and index them.
val assembler = (new VectorAssembler()
                  .setInputCols(Array("age","balance","day","duration","campaign","pdays","previous"))
                  .setOutputCol("features"))


// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a DecisionTree model.
val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")

// Convert indexed labels back to original labels.

// Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, assembler, dt))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println("Learned classification tree model:\n" + treeModel.toDebugString)