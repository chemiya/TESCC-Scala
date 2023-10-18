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
val primero = census_df.first()
val numreg = census_df.count()
val num_income = census_df.select("income").distinct.count()
val num_age = census_df.select("age").distinct.count()
val num_family_members_under_18 = census_df.select("family_members_under_18").distinct().count()
val num_citizenship = census_df.select("citizenship").distinct().count() 
val num_dividends_from_stocks = census_df.select("dividends_from_stocks").distinct().count() 
