import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

import org.apache.spark.ml.clustering.KMeans

//val dataset = spark.read.format("libsvm").load("sample_kmeans_data.txt")
val dataset  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Wholesale customers data.csv")

dataset.printSchema()
val caracterisiticas = dataset.select($"Channel",$"Region",$"Fresh",$"Milk",$"Grocery",$"Frozen",$"Detergents_Paper",$"Delicassen")
val assembler = (new VectorAssembler()
                  .setInputCols(Array("Channel","Region", "Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"))
                  .setOutputCol("features"))
val training = assembler.transform(caracterisiticas)
//trains the k-means model

val kmeans = new KMeans().setK(2).setSeed(1L)
val model = kmeans.fit(training)

// Evaluate clustering by calculate Within Set Sum of Squared Errors.
val WSSE = model.computeCost(training)
println(s"Within set sum of Squared Errors = $WSSE")

// Show results
println("Cluster Centers: ")
model.clusterCenters.foreach(println)