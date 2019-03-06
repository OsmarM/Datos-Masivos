//1.-Comenzar sesion en spark
import org.apache.spark.sql.SparkSession

val spar = SparkSession.builder().getOrCreate()
//2.-Cargar archivo Netflix_2011_2016
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
//4.-Esquema
df.printSchema()
//3.-Nombres de las columnas
df.columns()
//5.-Mostrar los primeros 5 renglones

df.head(5)

for(row <- df.head(5)){
    println(row)
}
//6.-Usa describe() para aprender sobre el DataFrame
df.describe()
//7.-Nuevo DataFrame "HV Ratio", entre precio de "High" y "Volumen"
val df2 = df.withColumn("HV Ratio", df("High")/df("Volume"))
df2.select("HV Ratio").show()
//8.-¿Que dia tuvo el pico mas alto en la columna High?
df2.orderBy($"High".desc).show()
//9.-¿Cual es el significado de la columna "Close"?
//Es el valor de las acciones de Netflix con el cual termino el dìa entre los años 2011- 2016
//10.-¿Cual es el maximo y minimo de la coumna "Volume"?
df2.select(max ($"Volume")).show()
df2.select(min ($"Volume")).show()
//11.-a)Close<$600
val result = df2.filter($"Close" < 600).collect()
val result = df.filter($"Close" < 600).count()
//b)¿Que porcentaje del tiempo fue la columna "High" mayor que $500?
val result2 = df2.filter($"High" > 500).collect()
val result2 = (df.filter($"High" > 500).count()*100)/1259
//c)Correlacion de Pearson
df2.select(corr("High", "Volume")).show()
//d)Maximo de la columna "High" por año
val dfyear=df2.withColumn("Year",year(df2("Date")))
val dfmax=dfyear.groupBy("Year").max()
dfmax.select($"Year",$"max(High)").show()
//e)¿Cual es ek promedio de la columna "Close" para cada mes del calendario?
val dfmes=df2.withColumn("Month",month(df2("Date")))
val dfprom=dfmes.select($"Month",$"Close").groupBy("Month").mean()
dfprom.orderBy($"Month".desc).show()
