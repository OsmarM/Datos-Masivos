import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.Pipeline

// Cargamos los datos en formato csv y los cambiamos a DataFramework para poder trabajar con ellos
val df = spark.read.format("csv").option("header", "true").load("/home/osmar/Descargas/spark/bin/Iris.csv")
//Le damos una estructura correcta a los campos que vamos a utilizar
val estructura = StructType(
StructField ("Spal-Lenght",DoubleType,true)::
StructField("Spal-W",DoubleType,true)::
StructField("Petal-Lenght",DoubleType,true)::
StructField("Petal-W",DoubleType,true)::
StructField("Label",StringType,true)::Nil)
//Asignamos la estructura a nuestro csv
val dfestructura = spark.read.option("header","false").schema(estructura)csv("Iris.csv")

val labelindex = new StringIndexer().setInputCol("Label").setOutputCol("Labels")
val assembler = (new VectorAssembler()
                  .setInputCols(Array("Spal-Lenght","Spal-W", "Petal-Lenght","Petal-W"))
                  .setOutputCol("features"))

//Los datos se dividen en entrenamiento y prueba(60% de los datos seran de entranamiento
//y 40% de prueba)
val Array(entrenamiento,examen)=dfestructura.randomSplit(Array(0.6, 0.4), seed = 1234L)
val capas = Array[Int](4, 5, 5, 3)

val mlp = new MultilayerPerceptronClassifier().setLayers(capas).setLabelCol("Labels").setFeaturesCol("features").setPredictionCol("predict").setBlockSize(128).setSeed(1234L).setMaxIter(100)
val pipeline = new Pipeline().setStages(Array(labelindex,assembler,mlp))

val modelo = pipeline.fit(entrenamiento)

val resultado = modelo.transform(examen)
// Split the data into train and test

//val splits = df.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)


//Se crea 4 capas para la red neuronal donde la primera tiene el numero de entradas,
//las dos capas de enmedio 5 cada una
//y la ultima capa tiene 3 que es una neurona por cada una de las categorias


// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier().setLayers(capas).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// train the model


// compute accuracy on the test set

val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))