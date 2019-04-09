import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.Pipeline

//Le damos una estructura correcta a los campos que vamos a utilizar
val estructura = StructType(
StructField ("Spal-Lenght",DoubleType,true)::
StructField("Spal-W",DoubleType,true)::
StructField("Petal-Lenght",DoubleType,true)::
StructField("Petal-W",DoubleType,true)::
StructField("Etiqueta",StringType,true)::Nil)
// Cargamos los datos en formato csv y los cambiamos a DataFramework para poder trabajar con ellos
//Asignamos la estructura a nuestro csv
val dfestructura = spark.read.option("header","false").schema(estructura)csv("Iris.csv")

val labelindex = new StringIndexer().setInputCol("Etiqueta").setOutputCol("label")
val assembler = (new VectorAssembler()
                  .setInputCols(Array("Spal-Lenght","Spal-W", "Petal-Lenght","Petal-W"))
                  .setOutputCol("features"))

//Los datos se dividen en entrenamiento y prueba(70% de los datos seran de entranamiento
//y 30% de prueba)
val Array(entrenamiento,validacion)=dfestructura.randomSplit(Array(0.7, 0.3), seed = 1234L)

//Se crea 4 capas para la red neuronal donde la primera tiene el numero de entradas,
//las dos capas de enmedio 5 cada una
//y la ultima capa tiene 3 que es una neurona por cada una de las categorias
val capas = Array[Int](4, 5, 5, 3)
//Creamos nuestro clasificador asignando las capas y columnas corrspondiente
val mlp = new MultilayerPerceptronClassifier().setLayers(capas).setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setBlockSize(128).setSeed(1234L).setMaxIter(100)
//Utilizamos el metodo Pipeline para 
val pipeline = new Pipeline().setStages(Array(labelindex,assembler,mlp))

val modelo = pipeline.fit(entrenamiento)

val resultado = modelo.transform(validacion)

//En base a los resultados obtenidos
val predictionAndLabels = resultado.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))
