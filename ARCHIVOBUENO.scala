/* Master en Ingeniería Informática - Universidad de Valladolid
*
*  TECNICAS ESCLABLES DE ANÁLISIS DE DATOS EN ENTORNOS BIG DATA: CLASIFICADORES
*  Proyecto Software: Construcción y validación de un modelo de clasificación usando la metodología CRISP-DM y Spark
*
*  Grupo 2: Sergio Agudelo Bernal
*           Miguel Ángel Collado Alonso 
*           José María Lozano Olmedo.
*/

/* Importamos funciones de estas bibliotecas */
import org.apache.spark.sql.types.{IntegerType, StringType, DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession,Row}
import org.apache.spark.sql.functions.col
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.{Matrix, Vectors, DenseVector}
import org.apache.spark.ml.stat.Correlation

import java.io.PrintWriter
import java.io.File

/* Ubicación de los datos a tratar*/
val PATH="/home/usuario/Scala/Proyecto/"
val FILE_CENSUS="census-income.data"


/*creamos un esquema para leer los datos */
val censusSchema = StructType(Array(
  StructField("age", IntegerType, false),
  StructField("class_of_worker", StringType, true),
  StructField("industry_code", IntegerType, true),
  StructField("occupation_code", IntegerType, true),
  StructField("education", StringType, true),
  StructField("wage_per_hour", IntegerType, false),
  StructField("enrolled_in_edu_last_wk", StringType, true),  
  StructField("marital_status", StringType, true),
  StructField("major_industry_code", StringType, true),
  StructField("major_occupation_code", StringType, true),
  StructField("race", StringType, true),
  StructField("hispanic_Origin", StringType, true),
  StructField("sex", StringType, true),
  StructField("member_of_labor_union", StringType, true),
  StructField("reason_for_unemployment", StringType, true),
  StructField("full_or_part_time_employment_status", StringType, true),
  StructField("capital_gains", IntegerType, false),
  StructField("capital_losses", IntegerType, false),
  StructField("dividends_from_stocks", IntegerType, false),
  StructField("tax_filer_status", StringType, true),
  StructField("region_of_previous_residence", StringType, true),
  StructField("state_of_previous_residence", StringType, true),
  StructField("detailed_household_and_family_status", StringType, true),
  StructField("detailed_household_summary_in_house_instance_weight", StringType, false),
  StructField("total_person_earnings", DoubleType, false),  
  StructField("migration_code_change_in_msa", StringType, true),
  StructField("migration_code_change_in_reg", StringType, true),
  StructField("migration_code_move_within_reg", StringType, true),
  StructField("live_in_this_house_one_year_ago", StringType, true),
  StructField("migration_prev_res_in_sunbelt", StringType, true),
  StructField("num_persons_worked_for_employer", IntegerType, false),
  StructField("family_members_under_18", StringType, true),  
  StructField("country_of_birth_father", StringType, true),
  StructField("country_of_birth_mother", StringType, true),
  StructField("country_of_birth_self", StringType, true),
  StructField("citizenship", StringType, true),
  StructField("own_business_or_self_employed", IntegerType, true),
  StructField("fill_inc_questionnaire_for_veterans_ad", StringType, true),
  StructField("veterans_benefits", IntegerType, false),
  StructField("weeks_worked_in_year", IntegerType, false),
  StructField("year", IntegerType, false),
  StructField("income", StringType, false)
));


/* leemos los datos y asignamos nombre a las columnas con el esquema
   con la opción ignoreLeadingWhiteSpace para quitar espacios en blanco de los atributos Integer*/
val census_df = spark.read.format("csv").
option("delimiter", ",").option("ignoreLeadingWhiteSpace","true").
schema(censusSchema).load(PATH + FILE_CENSUS)



//----------------------------------------ATRIBUTOS NUMÉRICOS-------------------------//

val listaDeAtributosNumericos = List("age", "industry_code", "occupation_code","wage_per_hour","capital_gains","capital_losses","dividends_from_stocks","total_person_earnings","num_persons_worked_for_employer","own_business_or_self_employed","veterans_benefits","weeks_worked_in_year","year")



// Recorrer la lista y mostrar el nombre de cada valor
  for (nombre_columna <- listaDeAtributosNumericos) {    
	println("Atributo: "+nombre_columna);
	val valores_ausentes=census_df.filter(col(nombre_columna).isNull).count();
	println("Valores ausentes: "+valores_ausentes);
	val valores_distintos = census_df.select(nombre_columna).distinct.count();
	println("Valores distintos: "+valores_distintos);
	val describe = census_df.describe(nombre_columna).show();	
	val distribucion = census_df.groupBy(nombre_columna).agg(count("*").alias("cantidad")).orderBy(desc("cantidad"))
    // Guardar la distribución en un archivo CSV sobreescribiendo si ya existe
    distribucion.write.mode("overwrite").csv(nombre_columna)
		
}



//----------------------------------------ATRIBUTOS CATEGORICOS-----------------------


  def numero_diferentes(nombre_columna:String): Long = {
    val cuenta=census_df.select(nombre_columna).distinct().count()
    cuenta
  }

  def valores_diferentes(nombre_columna:String): String = {
    val distintos=census_df.select(nombre_columna).distinct().collect().map(row => row.getString(0))
    val resultado_linea = distintos.mkString(", ")
    resultado_linea
  }

  def numero_cada_uno_diferentes(nombre_columna:String): DataFrame = {
      val numero_cada_uno=census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
      numero_cada_uno
  }

  def crear_fichero_resultados(df:DataFrame,nombre_columna:String):Unit={
    sc.parallelize(df.collect().toSeq,1).saveAsTextFile(nombre_columna+"_ordered")
  }


val listaAtributosCategoricos = List("class_of_worker","education","enrolled_in_edu_last_wk","marital_status","major_industry_code","major_occupation_code","member_of_labor_union","race","sex","full_or_part_time_employment_status","reason_for_unemployment","hispanic_Origin","tax_filer_status","region_of_previous_residence","state_of_previous_residence","detailed_household_and_family_status","detailed_household_summary_in_house_instance_weight","migration_code_change_in_msa","migration_code_change_in_reg","migration_code_move_within_reg","live_in_this_house_one_year_ago","migration_prev_res_in_sunbelt","family_members_under_18","country_of_birth_father","country_of_birth_mother","country_of_birth_self","citizenship","fill_inc_questionnaire_for_veterans_ad")



var array_valores_columnas_categoricas = Array[String]()

for (nombre_columna <- listaAtributosCategoricos) { 

  val numero_atributo_diferentes =numero_diferentes(nombre_columna)
  val valores_atributo_diferentes = valores_diferentes(nombre_columna)

  val numero_cada_uno_diferentes_atributo = numero_cada_uno_diferentes(nombre_columna)
  numero_cada_uno_diferentes_atributo.show(numero_cada_uno_diferentes_atributo.count().toInt, false)

  crear_fichero_resultados(numero_cada_uno_diferentes_atributo,nombre_columna)
  val moda_atributo = numero_cada_uno_diferentes_atributo.first()

  var escribir=nombre_columna+": "+valores_atributo_diferentes
  array_valores_columnas_categoricas=array_valores_columnas_categoricas:+escribir

  

}


sc.parallelize(array_valores_columnas_categoricas.toSeq,1).saveAsTextFile("resumen")





//----------------------------------------CORRELACIÓN ATRIBUTOS CONTINUOS-----------------------

// Atributos continuos son los numéricos menos los que representan categorías, como el código de ocupación y de industria, o own_business_or_self_employed, por ejemplo
val listaDeAtributosContinuos = List("age","wage_per_hour","capital_gains","capital_losses","dividends_from_stocks","total_person_earnings","num_persons_worked_for_employer","weeks_worked_in_year")
val census_continuous_cols_df = census_df.select(listaDeAtributosContinuos.map(col): _*)

def export_corr_matrix(df: DataFrame, cols: List[String], filepath: String): Unit = {
  val assembled = new VectorAssembler().setInputCols(df.columns).setOutputCol("correlations").transform(df)
  val correlations = Correlation.corr(assembled, column = "correlations", method = "pearson")
  val Row(corr_matrix: Matrix) = correlations.head

  var corr_export = ""
  val corr_header = ","+cols.mkString(",")
  corr_export = corr_export + corr_header + "\n"

  for (i <- 0 until corr_matrix.numRows) {
    corr_export = corr_export + cols(i)
    for (j <- 0 until corr_matrix.numCols) {
      corr_export = corr_export + "," + corr_matrix(i,j).toString
    }
    corr_export = corr_export + "\n"
  }

  val file = new File(filepath)
  val pw = new PrintWriter(file)
  pw.write(corr_export)
  pw.close()
}

export_corr_matrix(census_continuous_cols_df, listaDeAtributosContinuos, "corr_export.csv")




/-------------------------------------CORRELACION ATRIBUTOS CATEGORICOS----------------------



var correlaccion_categoricas=Array[String]()



def comprobarValoresMayores(df: DataFrame):Long= {
  
  val columnNames = df.columns
  //obtengo maximo de cada columna
  val maxValues = columnNames.map(col => df.agg(max(col)).collect()(0)(0).asInstanceOf[Long])

  val maxAmongMaxValues = maxValues.max

  maxAmongMaxValues
}



//para cada atributo categorico
for (i <- 0 until listaAtributosCategoricos.length) {
  val columna_actual = listaAtributosCategoricos(i)
    //compruebo su correlacion con el resto de atributos
    for (j <- (i + 1) until listaAtributosCategoricos.length) {
      val siguiente_columna = listaAtributosCategoricos(j)
      //se hace la tabla de contingecnia
      var tablaContingencia = census_df.groupBy(columna_actual).pivot(siguiente_columna).count().na.fill(0)
      
      tablaContingencia = tablaContingencia.drop(columna_actual)
     
      //tablaContingencia.show()

      //tablaContingencia.printSchema()

      //se guardan los resultados
      val resultado = comprobarValoresMayores(tablaContingencia)
     
      val fila = columna_actual+";"+siguiente_columna+";"+resultado
      println(fila)
      correlaccion_categoricas = correlaccion_categoricas :+ fila
      val nombre=columna_actual+"-"+siguiente_columna
      tablaContingencia.write.mode("overwrite").csv(nombre)
     

    }

}

sc.parallelize(correlaccion_categoricas.toSeq,1).saveAsTextFile("correlacion_categoricas")

