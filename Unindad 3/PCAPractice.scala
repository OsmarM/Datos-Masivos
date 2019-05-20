import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().appName("PCA_Example").getOrCreate()
//Importamos los datos
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Cancer_Data")
//Imprimimo el schema que hemos cargado
data.printSchema()

import org.apache.spark.ml.feature.{PCA,StandardScaler,VectorAssembler}

import org.apache.spark.ml.linalg.Vectors
//Creamos un arreglo con las columnas que vamos a utilizasr
val colnames = (Array("mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
"mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
"radius error", "texture error", "perimeter error", "area error", "smoothness error", "compactness error",
"concavity error", "concave points error", "symmetry error", "fractal dimension error", "worst radius",
"worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity",
"worst concave points", "worst symmetry", "worst fractal dimension"))
//Creamos un ensamblador con nuestro arreglo y las columnas del arreglo se llamaran features
val assembler = new VectorAssembler().setInputCols(colnames).setOutputCol("features")

//
val output = assembler.transform(data).select($"features")

//Normalizamos los datos
val scaler = (new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(false))

//Asignamos lo datos ya normalizados en nuestra variable de ensamblador
val scalerModel = scaler.fit(output)


val scaledData = scalerModel.transform(output)
//Ayuda a descubrir cuales son las 4 caracteristicas mas importantes del dataset
val pca = (new PCA()
  .setInputCol("scaledFeatures")
  .setOutputCol("pcaFeatures")
  .setK(4)
  .fit(scaledData))

val pcaDF = pca.transform(scaledData)

val result = pcaDF.select("pcaFeatures")
//Imprimimos los resultados
result.show()

result.head(1)
