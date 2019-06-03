import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.log4j._
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

data.printSchema()

data.head(1)
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label").fit(data)
val assembler = (new VectorAssembler()
                  .setInputCols(Array("age","balance","day","duration","campaign","pdays","previous"))
                  .setOutputCol("features"))


val Array(training, test) = data.randomSplit(Array(0.7, 0.3), seed = 12345)



val lr = new LogisticRegression()


val pipeline = new Pipeline().setStages(Array(labelIndexer,assembler,lr))

val model = pipeline.fit(training)

val results = model.transform(test)


val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

// Matriz de confusion
println("Confusion matrix:")
println(metrics.confusionMatrix)

metrics.accuracy
