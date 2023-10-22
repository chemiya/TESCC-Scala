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
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.ml.stat.Correlation

import java.io.PrintWriter
import java.io.File

/* Ubicación de los datos a tratar*/
val PATH="/home/usuario/Scala/Proyecto/"
val FILE_CENSUS="census-income-reducido.data"


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


//----------------------------------------ATRIBUTOS NUMÉRICOS-------------------------//

val listaDeAtributosNumericos = List("age", "industry_code", "occupation_code","wage_per_hour","capital_gains","capital_losses","dividends_from_stocks","total_person_earnings","num_persons_worked_for_employer","own_business_or_self_employed","veterans_benefits","weeks_worked_in_year","year")

// // Recorrer la lista y mostrar el nombre de cada valor
//   for (nombre_columna <- listaDeAtributosNumericos) {
// 	println("Atributo: "+nombre_columna);
// 	val valores_ausentes=census_df.filter(col(nombre_columna).isNull).count();
// 	println("Valores ausentes: "+valores_ausentes);
// 	val valores_distintos = census_df.select(nombre_columna).distinct.count();
// 	println("Valores distintos: "+valores_distintos);
// 	val describe = census_df.describe(nombre_columna).show();
// 	val distribucion = census_df.groupBy(nombre_columna).agg(count("*").alias("cantidad")).orderBy(desc("cantidad"))
//     // Guardar la distribución en un archivo CSV sobreescribiendo si ya existe
//     distribucion.write.mode("overwrite").csv(nombre_columna)

// }

//----------------------------------------ATRIBUTOS NOMINALES-------------------------//

// val listaDeAtributosNominales = List("class_of_worker","industry_code","occupation_code","education","enrolled_in_edu_last_wk","marital_status","major_industry_code","major_occupation_code","race","hispanic_Origin","sex","member_of_labor_union","full_or_part_time_employment_status","tax_filer_status","region_of_previous_residence","state_of_previous_residence","detailed_household_and_family_status","detailed_household_summary_in_house_instance_weight","migration_code_change_in_msa","migration_code_change_in_reg","migration_code_move_within_reg","live_in_this_house_one_year_ago","migration_prev_res_in_sunbelt","family_members_under_18","country_of_birth_father","country_of_birth_mother","country_of_birth_self","citizenship","own_business_or_self_employed","fill_inc_questionnaire_for_veterans_ad","veterans_benefits","year")

// // Recorrer la lista y mostrar el nombre de cada valor
//   for (nombre_columna <- listaDeAtributosNominales) {
// 	println("Atributo: "+nombre_columna);
// 	val valores_ausentes=census_df.filter(col(nombre_columna).isNull).count();
// 	println("Valores ausentes: "+valores_ausentes);
// 	val valores_distintos = census_df.select(nombre_columna).distinct.count();
// 	println("Valores distintos: "+valores_distintos);
// 	val distribucion = census_df.groupBy(nombre_columna).agg(count("*").alias("cantidad")).orderBy(desc("cantidad"))
// 	distribucion.show();
//     // Guardar la distribución en un archivo CSV sobreescribiendo si ya existe
//     distribucion.write.mode("overwrite").csv(nombre_columna)

// }







//----------------------------------------ATRIBUTOS CATEGORICOS-----------------------


  // def numero_diferentes(): Long = {
  //   val cuenta=census_df.select(nombre_columna).distinct().count()
  //   cuenta
  // }

  // def valores_diferentes(): String = {
  //   val distintos=census_df.select(nombre_columna).distinct().collect().map(row => row.getString(0))
  //   val resultado_linea = distintos.mkString(", ")
  //   resultado_linea
  // }

  // def numero_cada_uno_diferentes(): DataFrame = {
  //     val numero_cada_uno=census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
  //     numero_cada_uno
  // }

  // def crear_fichero_resultados(df:DataFrame):Unit={
  //   sc.parallelize(df.collect().toSeq,1).saveAsTextFile(nombre_columna+"_ordered")
  // }



//COLUMNA CLASS OF WORKER
/*
nombre_columna = "class_of_worker"

val numero_class_of_worker_diferentes =numero_diferentes()
val valores_class_of_worker_diferentes = valores_diferentes()

val numero_cada_uno_diferentes_class_of_worker = numero_cada_uno_diferentes()
numero_cada_uno_diferentes_class_of_worker.show(numero_cada_uno_diferentes_class_of_worker.count().toInt, false)

crear_fichero_resultados(numero_cada_uno_diferentes_class_of_worker)
val moda_class_of_worker = numero_cada_uno_diferentes_class_of_worker.first()

*/



/*

//COLUMNA INDUSTRY CODE

nombre_columna = "industry_code"
val numero_industry_code_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_industry_code_diferentes = census_df.select(nombre_columna).distinct()
valores_industry_code_diferentes.show()
val numero_cada_uno_diferentes_industry_code = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_industry_code.show(numero_cada_uno_diferentes_industry_code.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_industry_code.collect().toSeq,1).saveAsTextFile("industry_code_ordered")
val moda_industry_code = numero_cada_uno_diferentes_industry_code.first()




//COLUMNA OCCUPATION CODE



nombre_columna = "occupation_code"
val numero_occupation_code_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_occupation_code_diferentes = census_df.select(nombre_columna).distinct()
valores_occupation_code_diferentes.show()
val numero_cada_uno_diferentes_occupation_code = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_occupation_code.show(numero_cada_uno_diferentes_occupation_code.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_occupation_code.collect().toSeq,1).saveAsTextFile("occupation_code_ordered")
val moda_occupation_code = numero_cada_uno_diferentes_occupation_code.first()




//COLUMNA EDUCATION
nombre_columna = "education"
val numero_education_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_education_diferentes = census_df.select(nombre_columna).distinct()
valores_education_diferentes.show()
val numero_cada_uno_diferentes_education = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_education.show(numero_cada_uno_diferentes_education.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_education.collect().toSeq,1).saveAsTextFile("education_ordered")
val moda_education = numero_cada_uno_diferentes_education.first()




//COLUMNA ENROLLED IN EDU LAST WK
nombre_columna = "enrolled_in_edu_last_wk"
val numero_enrolled_in_edu_last_wk_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_enrolled_in_edu_last_wk_diferentes = census_df.select(nombre_columna).distinct()
valores_enrolled_in_edu_last_wk_diferentes.show()
val numero_cada_uno_diferentes_enrolled_in_edu_last_wk = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_enrolled_in_edu_last_wk.show(numero_cada_uno_diferentes_enrolled_in_edu_last_wk.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_enrolled_in_edu_last_wk.collect().toSeq,1).saveAsTextFile("enrolled_in_edu_last_wk_ordered")
val moda_enrolled_in_edu_last_wk = numero_cada_uno_diferentes_enrolled_in_edu_last_wk.first()









//COLUMNA MARITAL STATUS

nombre_columna = "marital_status"
val numero_marital_status_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_marital_status_diferentes = census_df.select(nombre_columna).distinct()
valores_marital_status_diferentes.show()
val numero_cada_uno_diferentes_marital_status = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_marital_status.show(numero_cada_uno_diferentes_marital_status.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_marital_status.collect().toSeq,1).saveAsTextFile("marital_status_ordered")
val moda_marital_status = numero_cada_uno_diferentes_marital_status.first()


//COLUMNA MAJOR INDUSTRY CODE

nombre_columna = "major_industry_code"
val numero_major_industry_code_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_major_industry_code_diferentes = census_df.select(nombre_columna).distinct()
valores_major_industry_code_diferentes.show()
val numero_cada_uno_diferentes_major_industry_code = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_major_industry_code.show(numero_cada_uno_diferentes_major_industry_code.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_major_industry_code.collect().toSeq,1).saveAsTextFile("major_industry_code_ordered")
val moda_major_industry_code = numero_cada_uno_diferentes_major_industry_code.first()



//COLUMNA MAJOR OCCUPATION CODE
nombre_columna = "major_occupation_code"
val numero_major_occupation_code_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_major_occupation_code_diferentes = census_df.select(nombre_columna).distinct()
valores_major_occupation_code_diferentes.show()
val numero_cada_uno_diferentes_major_occupation_code = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_major_occupation_code.show(numero_cada_uno_diferentes_major_occupation_code.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_major_occupation_code.collect().toSeq,1).saveAsTextFile("major_occupation_code_ordered")
val moda_major_occupation_code = numero_cada_uno_diferentes_major_occupation_code.first()



//COLUMNA RACE
nombre_columna="race"
val numero_race_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_race_diferentes = census_df.select(nombre_columna).distinct()
valores_race_diferentes.show()
val numero_cada_uno_diferentes_race = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_race.show(numero_cada_uno_diferentes_race.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_race.collect().toSeq,1).saveAsTextFile("race_ordered")
val moda_race = numero_cada_uno_diferentes_race.first()




//COLUMNA HISPANIC ORIGIN
nombre_columna="hispanic_origin"
val numero_hispanic_origin_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_hispanic_origin_diferentes = census_df.select(nombre_columna).distinct()
valores_hispanic_origin_diferentes.show()
val numero_cada_uno_diferentes_hispanic_origin = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_hispanic_origin.show(numero_cada_uno_diferentes_hispanic_origin.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_hispanic_origin.collect().toSeq,1).saveAsTextFile("hispanic_origin_ordered")
val moda_hispanic_origin = numero_cada_uno_diferentes_hispanic_origin.first()




//COLUMNA SEX
nombre_columna="sex"
val numero_sex_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_sex_diferentes = census_df.select(nombre_columna).distinct()
valores_sex_diferentes.show()
val numero_cada_uno_diferentes_sex = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_sex.show(numero_cada_uno_diferentes_sex.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_sex.collect().toSeq,1).saveAsTextFile("sex_ordered")
val moda_sex = numero_cada_uno_diferentes_sex.first()




//COLUMNA MEMBER OF LABOR UNION
nombre_columna="member_of_labor_union"
val numero_member_of_labor_union_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_member_of_labor_union_diferentes = census_df.select(nombre_columna).distinct()
valores_member_of_labor_union_diferentes.show()
val numero_cada_uno_diferentes_member_of_labor_union = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_member_of_labor_union.show(numero_cada_uno_diferentes_member_of_labor_union.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_member_of_labor_union.collect().toSeq,1).saveAsTextFile("member_of_labor_union_ordered")
val moda_member_of_labor_union = numero_cada_uno_diferentes_member_of_labor_union.first()




//COLUMNA REASON FOR UNEMPLOYMENT
nombre_columna="reason_for_unemployment"
val numero_reason_for_unemployment_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_reason_for_unemployment_diferentes = census_df.select(nombre_columna).distinct()
valores_reason_for_unemployment_diferentes.show()
val numero_cada_uno_diferentes_reason_for_unemployment = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_reason_for_unemployment.show(numero_cada_uno_diferentes_reason_for_unemployment.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_reason_for_unemployment.collect().toSeq,1).saveAsTextFile("reason_for_unemployment_ordered")
val moda_reason_for_unemployment = numero_cada_uno_diferentes_reason_for_unemployment.first()



//COLUMNA FULL OR PART TIME EMPLOYMENT STATUS
nombre_columna="full_or_part_time_employment_status"
val numero_full_or_part_time_employment_status_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_full_or_part_time_employment_status_diferentes = census_df.select(nombre_columna).distinct()
valores_full_or_part_time_employment_status_diferentes.show()
val numero_cada_uno_diferentes_full_or_part_time_employment_status = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_full_or_part_time_employment_status.show(numero_cada_uno_diferentes_full_or_part_time_employment_status.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_full_or_part_time_employment_status.collect().toSeq,1).saveAsTextFile("full_or_part_time_employment_status_ordered")
val moda_full_or_part_time_employment_status = numero_cada_uno_diferentes_full_or_part_time_employment_status.first()



//COLUMNA TAX FILER STATUS
nombre_columna="tax_filer_status"
val numero_tax_filer_status_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_tax_filer_status_diferentes = census_df.select(nombre_columna).distinct()
valores_tax_filer_status_diferentes.show()
val numero_cada_uno_diferentes_tax_filer_status = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_tax_filer_status.show(numero_cada_uno_diferentes_tax_filer_status.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_tax_filer_status.collect().toSeq,1).saveAsTextFile("tax_filer_status_ordered")
val moda_tax_filer_status = numero_cada_uno_diferentes_tax_filer_status.first()



//COLUMNA REGION OF PREVIOUS RESIDENCE
nombre_columna="region_of_previous_residence"
val numero_region_of_previous_residence_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_region_of_previous_residence_diferentes = census_df.select(nombre_columna).distinct()
valores_region_of_previous_residence_diferentes.show()
val numero_cada_uno_diferentes_region_of_previous_residence = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_region_of_previous_residence.show(numero_cada_uno_diferentes_region_of_previous_residence.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_region_of_previous_residence.collect().toSeq,1).saveAsTextFile("region_of_previous_residence_ordered")
val moda_region_of_previous_residence = numero_cada_uno_diferentes_region_of_previous_residence.first()



//COLUMA STATE OF PREVIOUS RESIDENCE
nombre_columna="state_of_previous_residence"
val numero_state_of_previous_residence_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_state_of_previous_residence_diferentes = census_df.select(nombre_columna).distinct()
valores_state_of_previous_residence_diferentes.show()
val numero_cada_uno_diferentes_state_of_previous_residence = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_state_of_previous_residence.show(numero_cada_uno_diferentes_state_of_previous_residence.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_state_of_previous_residence.collect().toSeq,1).saveAsTextFile("state_of_previous_residence_ordered")
val moda_state_of_previous_residence = numero_cada_uno_diferentes_state_of_previous_residence.first()



//COLUMNA DETAILED HOUSEHOLD AND FAMILY STATUS
nombre_columna="detailed_household_and_family_status"
val numero_detailed_household_and_family_status_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_detailed_household_and_family_status_diferentes = census_df.select(nombre_columna).distinct()
valores_detailed_household_and_family_status_diferentes.show()
val numero_cada_uno_diferentes_detailed_household_and_family_status = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_detailed_household_and_family_status.show(numero_cada_uno_diferentes_detailed_household_and_family_status.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_detailed_household_and_family_status.collect().toSeq,1).saveAsTextFile("detailed_household_and_family_status_ordered")
val moda_detailed_household_and_family_status = numero_cada_uno_diferentes_detailed_household_and_family_status.first()



//COLUMNA DETAILED HOUSEHOLD SUMMARY IN HOUSE INSTANCE WEIGHT
nombre_columna="detailed_household_summary_in_house_instance_weight"
val numero_detailed_household_summary_in_house_instance_weight_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_detailed_household_summary_in_house_instance_weight_diferentes = census_df.select(nombre_columna).distinct()
valores_detailed_household_summary_in_house_instance_weight_diferentes.show()
val numero_cada_uno_diferentes_detailed_household_summary_in_house_instance_weight = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_detailed_household_summary_in_house_instance_weight.show(numero_cada_uno_diferentes_detailed_household_summary_in_house_instance_weight.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_detailed_household_summary_in_house_instance_weight.collect().toSeq,1).saveAsTextFile("detailed_household_summary_in_house_instance_weight_ordered")
val moda_detailed_household_summary_in_house_instance_weight = numero_cada_uno_diferentes_detailed_household_summary_in_house_instance_weight.first()



//COLUMNA MIGRATION CODE CHANGE IN MSA
nombre_columna = "migration_code_change_in_msa"
val numero_migration_code_change_in_msa_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_migration_code_change_in_msa_diferentes = census_df.select(nombre_columna).distinct()
valores_migration_code_change_in_msa_diferentes.show()
val numero_cada_uno_diferentes_migration_code_change_in_msa = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_migration_code_change_in_msa.show(numero_cada_uno_diferentes_migration_code_change_in_msa.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_migration_code_change_in_msa.collect().toSeq,1).saveAsTextFile("migration_code_change_in_msa_ordered")
val moda_migration_code_change_in_msa = numero_cada_uno_diferentes_migration_code_change_in_msa.first()



//COLUMNA MIGRATION CODE CHANGE IN REG
nombre_columna="migration_code_change_within_reg"
val numero_migration_code_change_within_reg_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_migration_code_change_within_reg_diferentes = census_df.select(nombre_columna).distinct()
valores_migration_code_change_within_reg_diferentes.show()
val numero_cada_uno_diferentes_migration_code_change_within_reg = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_migration_code_change_within_reg.show(numero_cada_uno_diferentes_migration_code_change_within_reg.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_migration_code_change_within_reg.collect().toSeq,1).saveAsTextFile("migration_code_change_within_reg_ordered")
val moda_migration_code_change_within_reg = numero_cada_uno_diferentes_migration_code_change_within_reg.first()




//COLUMNA MIGRATION CODE MOVE WITHIN REG
nombre_columna="migration_code_move_within_reg"
val numero_migration_code_move_within_reg_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_migration_code_move_within_reg_diferentes = census_df.select(nombre_columna).distinct()
valores_migration_code_move_within_reg_diferentes.show()
val numero_cada_uno_diferentes_migration_code_move_within_reg = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_migration_code_move_within_reg.show(numero_cada_uno_diferentes_migration_code_move_within_reg.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_migration_code_move_within_reg.collect().toSeq,1).saveAsTextFile("migration_code_move_within_reg_ordered")
val moda_migration_code_move_within_reg = numero_cada_uno_diferentes_migration_code_move_within_reg.first()




//COLUMNA LIVE IN THIS HOUSE ONE YEAR AGO
nombre_columna="live_in_this_house_one_year_ago"
val numero_live_in_this_house_one_year_ago_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_live_in_this_house_one_year_ago_diferentes = census_df.select(nombre_columna).distinct()
valores_live_in_this_house_one_year_ago_diferentes.show()
val numero_cada_uno_diferentes_live_in_this_house_one_year_ago = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_live_in_this_house_one_year_ago.show(numero_cada_uno_diferentes_live_in_this_house_one_year_ago.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_live_in_this_house_one_year_ago.collect().toSeq,1).saveAsTextFile("live_in_this_house_one_year_ago_ordered")
val moda_live_in_this_house_one_year_ago = numero_cada_uno_diferentes_live_in_this_house_one_year_ago.first()





//COLUMNA MIGRATION PREV RES IN SUNBELT
nombre_columna="migration_prev_res_in_sunbelt"
val numero_migration_prev_res_in_sunbelt_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_migration_prev_res_in_sunbelt_diferentes = census_df.select(nombre_columna).distinct()
valores_migration_prev_res_in_sunbelt_diferentes.show()
val numero_cada_uno_diferentes_migration_prev_res_in_sunbelt = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_migration_prev_res_in_sunbelt.show(numero_cada_uno_diferentes_migration_prev_res_in_sunbelt.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_migration_prev_res_in_sunbelt.collect().toSeq,1).saveAsTextFile("migration_prev_res_in_sunbelt_ordered")
val moda_migration_prev_res_in_sunbelt = numero_cada_uno_diferentes_migration_prev_res_in_sunbelt.first()




//COLUMNA NUM PERSONS WORKED FOR EMPLOYER
nombre_columna="num_persons_worked_for_employer"
val numero_num_persons_worked_for_employer_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_num_persons_worked_for_employer_diferentes = census_df.select(nombre_columna).distinct()
valores_num_persons_worked_for_employer_diferentes.show()
val numero_cada_uno_diferentes_num_persons_worked_for_employer = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_num_persons_worked_for_employer.show(numero_cada_uno_diferentes_num_persons_worked_for_employer.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_num_persons_worked_for_employer.collect().toSeq,1).saveAsTextFile("num_persons_worked_for_employer_ordered")
val moda_num_persons_worked_for_employer = numero_cada_uno_diferentes_num_persons_worked_for_employer.first()



//COLUMNA FAMILY MEMBERS UNDER 18
nombre_columna="family_members_under_18"
val numero_family_members_under_18_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_family_members_under_18_diferentes = census_df.select(nombre_columna).distinct()
valores_family_members_under_18_diferentes.show()
val numero_cada_uno_diferentes_family_members_under_18 = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_family_members_under_18.show(numero_cada_uno_diferentes_family_members_under_18.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_family_members_under_18.collect().toSeq,1).saveAsTextFile("family_members_under_18_ordered")
val moda_family_members_under_18 = numero_cada_uno_diferentes_family_members_under_18.first()



//COLUMNA COUNTRY OF BIRTH FATHER
nombre_columna="country_of_birth_father"
val numero_country_of_birth_father_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_country_of_birth_father_diferentes = census_df.select(nombre_columna).distinct()
valores_country_of_birth_father_diferentes.show()
val numero_cada_uno_diferentes_country_of_birth_father = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_country_of_birth_father.show(numero_cada_uno_diferentes_country_of_birth_father.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_country_of_birth_father.collect().toSeq,1).saveAsTextFile("country_of_birth_father_ordered")
val moda_country_of_birth_father = numero_cada_uno_diferentes_country_of_birth_father.first()




//COLUMNA COUNTRY OF BIRTH MOTHER
nombre_columna="country_of_birth_mother"
val numero_country_of_birth_mother_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_country_of_birth_mother_diferentes = census_df.select(nombre_columna).distinct()
valores_country_of_birth_mother_diferentes.show()
val numero_cada_uno_diferentes_country_of_birth_mother = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_country_of_birth_mother.show(numero_cada_uno_diferentes_country_of_birth_mother.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_country_of_birth_mother.collect().toSeq,1).saveAsTextFile("country_of_birth_mother_ordered")
val moda_country_of_birth_mother = numero_cada_uno_diferentes_country_of_birth_mother.first()



//COLUMNA COUNTRY OF BIRTH SELF
nombre_columna="country_of_birth_self"
val numero_country_of_birth_self_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_country_of_birth_self_diferentes = census_df.select(nombre_columna).distinct()
valores_country_of_birth_self_diferentes.show()
val numero_cada_uno_diferentes_country_of_birth_self = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_country_of_birth_self.show(numero_cada_uno_diferentes_country_of_birth_self.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_country_of_birth_self.collect().toSeq,1).saveAsTextFile("country_of_birth_self_ordered")
val moda_country_of_birth_self = numero_cada_uno_diferentes_country_of_birth_self.first()




//COLUMNA CITIZENSHIP
nombre_columna="citizenship"
val numero_citizenship_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_citizenship_diferentes = census_df.select(nombre_columna).distinct()
valores_citizenship_diferentes.show()
val numero_cada_uno_diferentes_citizenship = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_citizenship.show(numero_cada_uno_diferentes_citizenship.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_citizenship.collect().toSeq,1).saveAsTextFile("citizenship_ordered")
val moda_citizenship = numero_cada_uno_diferentes_citizenship.first()




//COLUMNA FILL INC QUESTIONNAIRE FOR VETERANS AD
nombre_columna="fill_inc_questionnaire_for_veterans_ad"
val numero_fill_inc_questionnaire_for_veterans_ad_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_fill_inc_questionnaire_for_veterans_ad_diferentes = census_df.select(nombre_columna).distinct()
valores_fill_inc_questionnaire_for_veterans_ad_diferentes.show()
val numero_cada_uno_diferentes_fill_inc_questionnaire_for_veterans_ad = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_fill_inc_questionnaire_for_veterans_ad.show(numero_cada_uno_diferentes_fill_inc_questionnaire_for_veterans_ad.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_fill_inc_questionnaire_for_veterans_ad.collect().toSeq,1).saveAsTextFile("fill_inc_questionnaire_for_veterans_ad_ordered")
val moda_fill_inc_questionnaire_for_veterans_ad = numero_cada_uno_diferentes_fill_inc_questionnaire_for_veterans_ad.first()



//COLUMNA WEEKS WORKED IN YEAR
nombre_columna="weeks_worked_in_year"
val numero_weeks_worked_in_year_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_weeks_worked_in_year_diferentes = census_df.select(nombre_columna).distinct()
valores_weeks_worked_in_year_diferentes.show()
val numero_cada_uno_diferentes_weeks_worked_in_year = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_weeks_worked_in_year.show(numero_cada_uno_diferentes_weeks_worked_in_year.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_weeks_worked_in_year.collect().toSeq,1).saveAsTextFile("weeks_worked_in_year_ordered")
val moda_weeks_worked_in_year = numero_cada_uno_diferentes_weeks_worked_in_year.first()




//COLUMNA YEAR
nombre_columna="year"
val numero_year_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_year_diferentes = census_df.select(nombre_columna).distinct()
valores_year_diferentes.show()
val numero_cada_uno_diferentes_year = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_year.show(numero_cada_uno_diferentes_year.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_year.collect().toSeq,1).saveAsTextFile("year_ordered")
val moda_year = numero_cada_uno_diferentes_year.first()



//COLUMNA INCOME
nombre_columna="income"
val numero_income_diferentes = census_df.select(nombre_columna).distinct.count()
val valores_income_diferentes = census_df.select(nombre_columna).distinct()
valores_income_diferentes.show()
val numero_cada_uno_diferentes_income = census_df.groupBy(nombre_columna).count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
numero_cada_uno_diferentes_income.show(numero_cada_uno_diferentes_income.count().toInt, false)

sc.parallelize(numero_cada_uno_diferentes_income.collect().toSeq,1).saveAsTextFile("income_ordered")
val moda_income = numero_cada_uno_diferentes_income.first()

*/

// val doubleToDenseVector = udf { (value: Double) =>
//   new DenseVector(Array(value))
// }





// var correlaccion_categoricas=Array[String]()




// val columnas_string = censusSchema.fields.filter(_.dataType == StringType)

// for (i <- 0 until columnas_string.length) {
//   val columna_actual = columnas_string(i)

//   for (j <- (i + 1) until columnas_string.length) {
//     val siguiente_columna = columnas_string(j)
//     val nuevo_dataframe = census_df.select(columna_actual.name, siguiente_columna.name)

//     val corregido_nombre_columna_actual=columna_actual.name+"Index"
//     val corregido_nombre_columna_siguiente=siguiente_columna.name+"Index"

//     val indexer_actual = new StringIndexer().setInputCol(columna_actual.name).setOutputCol(corregido_nombre_columna_actual)
//     val indexer_siguiente = new StringIndexer().setInputCol(siguiente_columna.name).setOutputCol(corregido_nombre_columna_siguiente)

//     val indexers = Array(indexer_actual, indexer_siguiente)
//     val pipeline = new Pipeline().setStages(indexers)
//     val indexed_dataframe = pipeline.fit(nuevo_dataframe).transform(nuevo_dataframe)




//     val nuevo_dataframe_correlaccion: DataFrame = indexed_dataframe.select(corregido_nombre_columna_actual, corregido_nombre_columna_siguiente)



//     val corregido_nombre_columna_siguiente_vector=corregido_nombre_columna_siguiente+"Vector"

//     val nuevo_dataframe_vectors = nuevo_dataframe_correlaccion.withColumn(corregido_nombre_columna_siguiente_vector, doubleToDenseVector(col(corregido_nombre_columna_siguiente)))



//     val chi = ChiSquareTest.test(nuevo_dataframe_vectors, corregido_nombre_columna_siguiente_vector, corregido_nombre_columna_actual).head
//     println(s"pValue para las columnas ${columna_actual.name} con ${siguiente_columna.name} =  ${chi.getAs[Double](0)}")

//     val fila = columna_actual.name+";"+siguiente_columna.name+";"+chi.getAs[Double](0)
//     correlaccion_categoricas = correlaccion_categoricas :+ fila


//   }
// }

//----------------------------------------CORRELACIÓN ATRIBUTOS CONTINUOS-----------------------

val df = census_df.select(listaDeAtributosNumericos.map(col): _*)
val cols = listaDeAtributosNumericos

// def export_corr_matrix(df: DataFrame, cols: List[String], filepath: String): Unit = {
val assembled = new VectorAssembler().setInputCols(df.columns).setOutputCol("correlations").transform(df)
val correlations = Correlation.corr(assembled, column = "correlations", method = "pearson")

correlations.collect()(0)

// val corr_rows = correlations.collect()(0).toString.replaceAll("  \n","\n").trim.replaceAll(" +", " ").replaceAll("\\[","").replaceAll(" \\]","").replaceAll(" ",",").split("\n")

// var corr_export = ""
// val corr_header = ","+cols.mkString(",")
// corr_export = corr_export+corr_header+"\n"

// for (i <- 0 until corr_rows.length) {
//   corr_export = corr_export+cols(i)+","+corr_rows(i)+"\n"
// }

// val file = new File(filepath)
// val pw = new PrintWriter(file)
// pw.write(corr_export)
// pw.close()
// }

// export_corr_matrix(census_continuous_cols_df, listaDeAtributosNumericos, "corr_export.csv")