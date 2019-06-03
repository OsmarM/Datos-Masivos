import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.Pipeline


val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label").fit(data)
val assembler = (new VectorAssembler()
                  .setInputCols(Array("age","balance","day","duration","campaign","pdays","previous"))
                  .setOutputCol("features"))

//Los datos se dividen en entrenamiento y prueba(70% de los datos seran de entranamiento
//y 30% de prueba)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed=1234L)
//Se crea 7 capas para la red neuronal donde la primera tiene el numero de entradas,
//las dos capas de enmedio 5 cada una
//y la ultima capa tiene 2 que son los resultas que se pueden obtener
val capas = Array[Int](7, 5, 5, 2)
//Creamos nuestro clasificador asignando las capas y columnas corrspondiente
val mlp = new MultilayerPerceptronClassifier().setLayers(capas).setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setBlockSize(128).setSeed(1234L).setMaxIter(100)
//Utilizamos el metodo Pipeline para 
val pipeline = new Pipeline().setStages(Array(labelIndexer,assembler,mlp))

val modelo = pipeline.fit(trainingData)

val resultado = modelo.transform(testData)

//En base a los resultados obtenidos
val predictionAndLabels = resultado.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))