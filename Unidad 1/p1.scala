//Calculando perimetro
val p=20 //Se declara el valor del perímetro
val pi=3.1416 //Se declara el valor de Pi
val r=p/(2*pi) //Despejando la formula para calcular el perímetro de un circulo obtenemos la formula para calcular el radio
println(r) //Imprimimos el valor del radio
//Interpolacion de strings
val bird="tweet" //Asignamos el valor a la variable
println(s"Estoy escribiendo un $bird") //Usamos Interpolacion para imprimir el mensaje
//Slice Luke
val sw="Hola Luke soy tu padre"
sw slice (5,9)
//Var y val, las variables pueden ser cambiadas por el mismo tipo de dato, en cambio value no.
var myvar=12
val myval="Hola"
myvar=200
myval="Bye"
//Tuplas
val mytup= (2,4,5,1,2,3,3.1416,23)
mytup._7
