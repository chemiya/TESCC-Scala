import org.apache.spark.sql.types.{IntegerType, StringType, DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession,Row}
import org.apache.spark.{SparkConf, SparkContext}
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



val PATH="/home/usuario/Scala/Proyecto/"
val FILE_CENSUS="census-income.data"



/*creamos un esquema para leer los datos */
val censusSchema = StructType(Array(
  StructField("age", IntegerType, false),
  StructField("class_of_worker", StringType, true),
  StructField("industry_code", StringType, true),
  StructField("occupation_code", StringType, true),
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






val census_df = spark.read.format("csv").
option("delimiter", ",").option("ignoreLeadingWhiteSpace","true").
schema(censusSchema).load(PATH + FILE_CENSUS)


val listaAtributosCategoricos = List("class_of_worker","industry_code","occupation_code","education","enrolled_in_edu_last_wk","marital_status","major_industry_code","major_occupation_code","member_of_labor_union","race","sex","full_or_part_time_employment_status","reason_for_unemployment","hispanic_Origin","tax_filer_status","region_of_previous_residence","state_of_previous_residence","detailed_household_and_family_status","detailed_household_summary_in_house_instance_weight","migration_code_change_in_msa","migration_code_change_in_reg","migration_code_move_within_reg","live_in_this_house_one_year_ago","migration_prev_res_in_sunbelt","family_members_under_18","country_of_birth_father","country_of_birth_mother","country_of_birth_self","citizenship","fill_inc_questionnaire_for_veterans_ad")





for (i <- 0 until listaAtributosCategoricos.length) {
  val columna_actual = listaAtributosCategoricos(i)

  var tablaContingencia = census_df.groupBy(columna_actual).pivot("income").count().na.fill(0)
  tablaContingencia.show()







}
