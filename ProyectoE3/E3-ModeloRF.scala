/* Master en Ingeniería Informática - Universidad de Valladolid
*
*  TECNICAS ESCLABLES DE ANÁLISIS DE DATOS EN ENTORNOS BIG DATA: CLASIFICADORES
*  Proyecto de clasificación. Tercera etapa: Creación, selección y evaluación de modelos
*
*  Script para la creación del modelo utilizando Random forest en Spark ML
*
*  Grupo 2: Sergio Agudelo Bernal
*           Miguel Ángel Collado Alonso
*           José María Lozano Olmedo.
*/

import org.apache.spark.sql.types.{IntegerType, StringType, DoubleType, StructField, StructType}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator



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


val PATH="/home/usuario/Scala/ProyectoE3/"
val FILE_CENSUS="census-income.data"

// leemos los datos y creamos el DataFrame

val census_df = spark.read.format("csv").
option("delimiter", ",").option("ignoreLeadingWhiteSpace","true").
schema(censusSchema).load(PATH + FILE_CENSUS)



/*** NOTA: Hay que cargar previamente los script en spark-shell
*   :load TransformDataframe.scala
*   :load CleanDataframe.scala
*/

// limpiamos y transformamos el DataFrame

import TransformDataframe._
import CleanDataframe._
val census_df_limpio=cleanDataframe(census_df)
val trainCensusDFProcesado = transformDataFrame(census_df_limpio)


/* Realizamos una partición aleatoria de los datos */
/* 66% para entrenamiento, 34% para prueba */
/* Fijamos seed para usar la misma partición en distintos ejemplos*/

val dataSplits = trainCensusDFProcesado.randomSplit(Array(0.66, 0.34), seed=0)
val trainCensus = dataSplits(0)
val testCensus = dataSplits(1)



// Usamos el clasificador RamdomForest

val RFCensus = new RandomForestClassifier().
setFeaturesCol("features").
setLabelCol("label").
setNumTrees(10).
setMaxDepth(7).
setMaxBins(10).
setMinInstancesPerNode(1).
setMinInfoGain(0.0).
setCacheNodeIds(false).
setCheckpointInterval(10)

/* Entrenamos el modelo: Random Forest */
val RFCensusModel =RFCensus.fit(trainCensus)

// Examinamos el Árbol
RFCensusModel.toDebugString

// Guardamos el modelo
RFCensusModel.write.overwrite().save(PATH + "modelo")



/* ** NO FUNCIONAN LOS CÁLCULOS DE LAS MÉTRICAS*** 

/* Predecimos la clase de los ejemplos de prueba */

val FILE_CENSUS_TEST="census-income.test"

val census_df_test = spark.read.format("csv").
option("delimiter", ",").option("ignoreLeadingWhiteSpace","true").
schema(censusSchema).load(PATH + FILE_CENSUS_TEST)


/*** NOTA: Hay que cargar previamente los script en spark-shell
*   :load TransformDataframe.scala
*   :load CleanDataframe.scala
*/

import TransformDataframe._
import CleanDataframe._
val census_df_limpio=cleanDataframe(census_df_test)
val testCensusDF = transformDataFrame(census_df_limpio)
*/


/* Predecimos la clase de los ejemplos de prueba */
val predictionsAndLabelsDF = RFCensusModel.transform(testCensus).select("prediction", "label")


//Creamos una instancia de clasificacion multiclass 
val RFMetrica = new MulticlassClassificationEvaluator()

/* Fijamos como métrica la tasa de error: accuracy */
RFMetrica.setMetricName("accuracy")

/* Calculamos la tasa de acierto */
val aciertoRF = RFMetrica.evaluate(predictionsAndLabelsDF)

/* Calculamos el error */
val errorRF = 1 - aciertoRF

// Lo mostramos
println(f"Tasa de error= $errorRF%1.3f")



