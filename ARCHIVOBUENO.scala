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
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

/* Ubicación de los datos a tratar*/
val PATH="/home/usuario/Scala/Proyecto/"
val FILE_CENSUS="census-income.data"


/*creamos un esquema para leer los datos */
val censusSchema = StructType(Array(
  StructField("age", IntegerType, false),
  StructField("class_of_worker", StringType, true),
  StructField("industry_code", StringType, true),
  StructField("occupation_code", StringType, true),
  StructField("education", StringType, true),
  StructField("wage_per_hour", DoubleType, false),
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
  StructField("capital_gains", DoubleType, false),
  StructField("capital_losses", DoubleType, false),
  StructField("dividends_from_stocks", DoubleType, false),
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
  StructField("num_persons_worked_for_employer", StringType, false),
  StructField("family_members_under_18", StringType, true),  
  StructField("country_of_birth_father", StringType, true),
  StructField("country_of_birth_mother", StringType, true),
  StructField("country_of_birth_self", StringType, true),
  StructField("citizenship", StringType, true),
  StructField("own_business_or_self_employed", DoubleType, true),
  StructField("fill_inc_questionnaire_for_veterans_ad", StringType, true),
  StructField("veterans_benefits", DoubleType, false),
  StructField("weeks_worked_in_year", StringType, false),
  StructField("year", StringType, false),
  StructField("income", StringType, false)
));

/* leemos los datos y asignamos nombre a las columnas con el
esquema*/
val census_df = spark.read.format("csv").
option("delimiter", ",").
schema(censusSchema).load(PATH + FILE_CENSUS)

/* evaluamos algunos datos */
/*
val primero = census_df.first()
val numreg = census_df.count()
val num_income = census_df.select("income").distinct.count()
val num_age = census_df.select("age").distinct.count()
val num_family_members_under_18 = census_df.select("family_members_under_18").distinct().count()
val num_citizenship = census_df.select("citizenship").distinct().count() 
val num_dividends_from_stocks = census_df.select("dividends_from_stocks").distinct().count() 
*/


//COLUMNA AGE---------------------CORRECTO

var nombre_columna = "age"

/*
val numero_cada_uno_diferentes_age = census_df.groupBy(nombre_columna).count()
numero_cada_uno_diferentes_age.show(numero_cada_uno_diferentes_age.count().toInt, false)

val filtrada_columna_age = census_df.filter(col(nombre_columna) > 17 && col(nombre_columna) < 100)
val maximo_columna_age = filtrada_columna_age.agg(max(nombre_columna)).head().getInt(0)
val minimo_columna_age = filtrada_columna_age.agg(min(nombre_columna)).head().getInt(0)
val media_columna_age = filtrada_columna_age.select(avg(col(nombre_columna))).first().getDouble(0)
val desviacion_columna_age = filtrada_columna_age.agg(stddev(nombre_columna)).head().getDouble(0)
*/



//COLUMNA WAGE PER HOUR----------------------CORRECTO

/*
nombre_columna = "wage_per_hour"
val numero_cada_uno_diferentes_wage_per_hour = census_df.groupBy(nombre_columna).count()
numero_cada_uno_diferentes_wage_per_hour.show(numero_cada_uno_diferentes_wage_per_hour.count().toInt, false)


val maximo_columna_wage_per_hour = census_df.agg(max(nombre_columna)).head().getDouble(0)
val minimo_columna_wage_per_hour = census_df.agg(min(nombre_columna)).head().getDouble(0)
val media_columna_wage_per_hour = census_df.select(avg(col(nombre_columna))).first().getDouble(0)
val desviacion_columna_wage_per_hour = census_df.agg(stddev(nombre_columna)).head().getDouble(0)
*/





//COLUMNA CAPITAL GAINS----------------------CORRECTO
/*
nombre_columna = "capital_gains"
val numero_cada_uno_diferentes_capital_gains = census_df.groupBy(nombre_columna).count()
numero_cada_uno_diferentes_capital_gains.show(numero_cada_uno_diferentes_capital_gains.count().toInt, false)


val maximo_columna_capital_gains = census_df.agg(max(nombre_columna)).head().getDouble(0)
val minimo_columna_capital_gains = census_df.agg(min(nombre_columna)).head().getDouble(0)
val media_columna_capital_gains = census_df.select(avg(col(nombre_columna))).first().getDouble(0)
val desviacion_columna_capital_gains = census_df.agg(stddev(nombre_columna)).head().getDouble(0)
*/







//COLUMNA CLASS OF WORKER





nombre_columna = "class_of_worker"
val numero_class_of_worker_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_class_of_worker_diferentes = census_df.select(nombre_columna).distinct()
val numero_cada_uno_diferentes_class_of_worker = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_class_of_worker.show()

sc.parallelize(numero_cada_uno_diferentes_class_of_worker.collect().toSeq,1).saveAsTextFile("class_of_work_ordered")
val moda_class_of_worker = numero_cada_uno_diferentes_class_of_worker.first()