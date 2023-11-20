/* Master en Ingeniería Informática - Universidad de Valladolid
*
*  TECNICAS ESCLABLES DE ANÁLISIS DE DATOS EN ENTORNOS BIG DATA: CLASIFICADORES
*  Proyecto Software: Construcción y validación de un modelo de clasificación usando la metodología CRISP-DM y Spark
*
*  Script para la evaluación del modelo
*
*  Grupo 2: Sergio Agudelo Bernal
*           Miguel Ángel Collado Alonso
*           José María Lozano Olmedo.
*/

import org.apache.spark.sql.types.{IntegerType, StringType, DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession,Row}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.OneHotEncoderModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer

// Función para transformar el DataFrame

def transformDataFrame(census_df: DataFrame): DataFrame = {

var attributeColumns = census_df.columns.toSeq.filter(_ != "income").toArray
var outputColumns = attributeColumns.map(_ + "-num").toArray
var siColumns= new StringIndexer().setInputCols(attributeColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")
var simColumns = siColumns.fit(census_df)
var census_df_numeric = simColumns.transform(census_df).drop(attributeColumns:_*)

// Generamos los nombres de las nuevas columnas
var inputColumns = outputColumns
outputColumns = attributeColumns.map(_ + "-hot").toArray

// Creamos OneHotEncoder para transformar todos los atributos, salvo la clase
var hotColumns = new OneHotEncoder().setInputCols(inputColumns).setOutputCols(outputColumns)

//Creamos el OneHotEncoderModel
var hotmColumns = hotColumns.fit(census_df_numeric)

var censusDFhot = hotmColumns.transform(census_df_numeric).drop(inputColumns:_*)
    
//  Definimos columna Features con todos los atributos, menos la clase, con VectorAssempler
var va = new VectorAssembler().setOutputCol("features").setInputCols(outputColumns)

// Creamos el DataFrame carFeaturesClaseDF con columnas features y clase
var censusFeaturesClaseDF = va.transform(censusDFhot).select("features", "income")

// creamos el StringIndexer para la clase
var indiceClase= new StringIndexer().setInputCol("income").setOutputCol("label").setStringOrderType("alphabetDesc")

// Creamos el DataFrame carFeaturesLabelDF con columnas features y label
var censusFeaturesLabelDF = indiceClase.fit(censusFeaturesClaseDF).transform(censusFeaturesClaseDF).drop("income")
//censusFeaturesLabelDF.show(10)

censusFeaturesLabelDF
}


// Función para limpiar el DataFrame


def cleanDataframe(census_df: DataFrame): DataFrame = {

val intervalosAge = Array(0, 18, 35, 50, 70, 100)
val labelsAge = Array("0-18", "19-35", "36-50", "51-70", "71-100")

var census_df_nuevo = census_df.withColumn("age-converted",
  when(col("age").between(intervalosAge(0), intervalosAge(1)), labelsAge(0))
    .when(col("age").between(intervalosAge(1), intervalosAge(2)), labelsAge(1))
    .when(col("age").between(intervalosAge(2), intervalosAge(3)), labelsAge(2))
    .when(col("age").between(intervalosAge(3), intervalosAge(4)), labelsAge(3))
    .when(col("age").between(intervalosAge(4), intervalosAge(5)), labelsAge(4))
    .otherwise("Out")
)


val intervalosWageHour = Array(0, 740,  1480,  2220, 3000, 10000)
val labelsWageHour = Array("0-740", "740-1480", "1480-2220", "2220-3000", "3000-10000")

census_df_nuevo = census_df_nuevo.withColumn("wage_per_hour-converted",
  when(col("wage_per_hour").between(intervalosWageHour(0), intervalosWageHour(1)), labelsWageHour(0))
    .when(col("wage_per_hour").between(intervalosWageHour(1), intervalosWageHour(2)), labelsWageHour(1))
    .when(col("wage_per_hour").between(intervalosWageHour(2), intervalosWageHour(3)), labelsWageHour(2))
    .when(col("wage_per_hour").between(intervalosWageHour(3), intervalosWageHour(4)), labelsWageHour(3))
    .when(col("wage_per_hour").between(intervalosWageHour(4), intervalosWageHour(5)), labelsWageHour(4))
    .otherwise("Out")
)


val intervalosCapitalGains = Array(0, 500,  1000,  1500, 2000, 100000)
val labelsCapitalGains = Array("0-500", "500-1000", "1000-1500", "1500-2000", "2000-100000")


census_df_nuevo = census_df_nuevo.withColumn("capital_gains-converted",
  when(col("capital_gains").between(intervalosCapitalGains(0), intervalosCapitalGains(1)), labelsCapitalGains(0))
    .when(col("capital_gains").between(intervalosCapitalGains(1), intervalosCapitalGains(2)), labelsCapitalGains(1))
    .when(col("capital_gains").between(intervalosCapitalGains(2), intervalosCapitalGains(3)), labelsCapitalGains(2))
    .when(col("capital_gains").between(intervalosCapitalGains(3), intervalosCapitalGains(4)), labelsCapitalGains(3))
    .when(col("capital_gains").between(intervalosCapitalGains(4), intervalosCapitalGains(5)), labelsCapitalGains(4))
    .otherwise("Out")
)

val intervalosCapitalLosses = Array(0, 1120,  1680,  2240, 2800, 5000)
val labelsCapitalLosses = Array("0-1120", "1120-1680", "1680-2240", "2240-2800", "2800-5000")

census_df_nuevo = census_df_nuevo.withColumn("capital_losses-converted",
  when(col("capital_losses").between(intervalosCapitalLosses(0), intervalosCapitalLosses(1)), labelsCapitalLosses(0))
    .when(col("capital_losses").between(intervalosCapitalLosses(1), intervalosCapitalLosses(2)), labelsCapitalLosses(1))
    .when(col("capital_losses").between(intervalosCapitalLosses(2), intervalosCapitalLosses(3)), labelsCapitalLosses(2))
    .when(col("capital_losses").between(intervalosCapitalLosses(3), intervalosCapitalLosses(4)), labelsCapitalLosses(3))
    .when(col("capital_losses").between(intervalosCapitalLosses(4), intervalosCapitalLosses(5)), labelsCapitalLosses(4))
    .otherwise("Out")
)

val intervalosDividens = Array(0, 500,  1000,  5000, 10000, 100000)
val labelsDividens = Array("0-500", "500-1000", "1000-5000", "5000-10000", "10000-100000")


census_df_nuevo = census_df_nuevo.withColumn("dividends_from_stocks-converted",
  when(col("dividends_from_stocks").between(intervalosDividens(0), intervalosDividens(1)), labelsDividens(0))
    .when(col("dividends_from_stocks").between(intervalosDividens(1), intervalosDividens(2)), labelsDividens(1))
    .when(col("dividends_from_stocks").between(intervalosDividens(2), intervalosDividens(3)), labelsDividens(2))
    .when(col("dividends_from_stocks").between(intervalosDividens(3), intervalosDividens(4)), labelsDividens(3))
    .when(col("dividends_from_stocks").between(intervalosDividens(4), intervalosDividens(5)), labelsDividens(4))
    .otherwise("Out")
)


val intervalosTotalEarnings = Array(0, 740,  1480,  2220, 3000, 100000)
val labelsTotalEarnings = Array("0-740", "740-1480", "1480-2220", "2220-3000", "3000-100000")


census_df_nuevo = census_df_nuevo.withColumn("total_person_earnings-converted",
  when(col("total_person_earnings").between(intervalosTotalEarnings(0), intervalosTotalEarnings(1)), labelsTotalEarnings(0))
    .when(col("total_person_earnings").between(intervalosTotalEarnings(1), intervalosTotalEarnings(2)), labelsTotalEarnings(1))
    .when(col("total_person_earnings").between(intervalosTotalEarnings(2), intervalosTotalEarnings(3)), labelsTotalEarnings(2))
    .when(col("total_person_earnings").between(intervalosTotalEarnings(3), intervalosTotalEarnings(4)), labelsTotalEarnings(3))
    .when(col("total_person_earnings").between(intervalosTotalEarnings(4), intervalosTotalEarnings(5)), labelsTotalEarnings(4))
    .otherwise("Out")
)


val intervalosPersonsWorkerEmployer = Array(0, 2,  4,  6, 8, 10)
val labelsPersonsWorkerEmployer= Array("0-2", "2-4", "4-6", "6-8", "8-10")

census_df_nuevo = census_df_nuevo.withColumn("num_persons_worked_for_employer-converted",
  when(col("num_persons_worked_for_employer").between(intervalosPersonsWorkerEmployer(0), intervalosPersonsWorkerEmployer(1)), labelsPersonsWorkerEmployer(0))
    .when(col("num_persons_worked_for_employer").between(intervalosPersonsWorkerEmployer(1), intervalosPersonsWorkerEmployer(2)), labelsPersonsWorkerEmployer(1))
    .when(col("num_persons_worked_for_employer").between(intervalosPersonsWorkerEmployer(2), intervalosPersonsWorkerEmployer(3)), labelsPersonsWorkerEmployer(2))
    .when(col("num_persons_worked_for_employer").between(intervalosPersonsWorkerEmployer(3), intervalosPersonsWorkerEmployer(4)), labelsPersonsWorkerEmployer(3))
    .when(col("num_persons_worked_for_employer").between(intervalosPersonsWorkerEmployer(4), intervalosPersonsWorkerEmployer(5)), labelsPersonsWorkerEmployer(4))
    .otherwise("Out")
)

val intervalosWeeksWorked = Array(0, 10,  20,  30, 40, 55)
val labelsWeeksWorked = Array("0-10", "10-20", "20-30", "30-40", "40-55")


census_df_nuevo = census_df_nuevo.withColumn("weeks_worked_in_year-converted",
  when(col("weeks_worked_in_year").between(intervalosWeeksWorked(0), intervalosWeeksWorked(1)), labelsWeeksWorked(0))
    .when(col("weeks_worked_in_year").between(intervalosWeeksWorked(1), intervalosWeeksWorked(2)), labelsWeeksWorked(1))
    .when(col("weeks_worked_in_year").between(intervalosWeeksWorked(2), intervalosWeeksWorked(3)), labelsWeeksWorked(2))
    .when(col("weeks_worked_in_year").between(intervalosWeeksWorked(3), intervalosWeeksWorked(4)), labelsWeeksWorked(3))
    .when(col("weeks_worked_in_year").between(intervalosWeeksWorked(4), intervalosWeeksWorked(5)), labelsWeeksWorked(4))
    .otherwise("Out")
)

val intervalosOwnBusiness = Array(0, 1,2,6)
val labelsOwnBusiness  = Array("0-1", "1-2", "2-6")


census_df_nuevo = census_df_nuevo.withColumn("own_business_or_self_employed-converted",
  when(col("own_business_or_self_employed").between(intervalosOwnBusiness(0), intervalosOwnBusiness(1)), labelsOwnBusiness(0))
    .when(col("own_business_or_self_employed").between(intervalosOwnBusiness(1), intervalosOwnBusiness(2)), labelsOwnBusiness(1))
    .when(col("own_business_or_self_employed").between(intervalosOwnBusiness(2), intervalosOwnBusiness(3)), labelsOwnBusiness(2))
    .otherwise("own_business_or_self_employed")
)


val intervalosVeterans = Array(0, 1,2,6)
val labelsVeterans  = Array("0-1", "1-2", "2-6")


census_df_nuevo = census_df_nuevo.withColumn("veterans_benefits-converted",
  when(col("veterans_benefits").between(intervalosVeterans(0), intervalosVeterans(1)), labelsVeterans(0))
    .when(col("veterans_benefits").between(intervalosVeterans(1), intervalosVeterans(2)), labelsVeterans(1))
    .when(col("veterans_benefits").between(intervalosVeterans(2), intervalosVeterans(3)), labelsVeterans(2))
    .otherwise("veterans_benefits")
)


    // Removidos para disminuir error: ,"age-converted","capital_gains-converted","capital_losses-converted","dividends_from_stocks-converted","num_persons_worked_for_employer-converted","weeks_worked_in_year-converted"
    val nuevoDataFrame = census_df_nuevo.select("class_of_worker","education","marital_status","major_industry_code","major_occupation_code","member_of_labor_union","race","sex","full_or_part_time_employment_status","hispanic_Origin","tax_filer_status","region_of_previous_residence","detailed_household_and_family_status","detailed_household_summary_in_house_instance_weight","migration_code_change_in_msa","migration_code_change_in_reg","migration_code_move_within_reg","live_in_this_house_one_year_ago","family_members_under_18","citizenship","wage_per_hour-converted","total_person_earnings-converted","veterans_benefits-converted","own_business_or_self_employed-converted","income")

    nuevoDataFrame
}


val PATH="/home/usuario/Scala/ProyectoCorregido/"
val loadedDTcensusModel = DecisionTreeClassificationModel.load(PATH + "modelo")

val FILE_CENSUS_TEST="census-income.test"

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


val census_df_test = spark.read.format("csv").
option("delimiter", ",").option("ignoreLeadingWhiteSpace","true").
schema(censusSchema).load(PATH + FILE_CENSUS_TEST)


val census_df_limpio=cleanDataframe(census_df_test)
val testCensusDF = transformDataFrame(census_df_limpio)

val predictionsAndLabelsDF = loadedDTcensusModel.transform(testCensusDF).select("prediction", "label")

val diferentes = predictionsAndLabelsDF.filter(not(col("prediction") === col("label"))).count()
val total=testCensusDF.count()
val error=(diferentes.toDouble /total.toDouble)*100

println(f"Tasa de error= $error%1.3f")
