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
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, RegressionMetrics}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics 
import org.apache.spark.ml.linalg.Vector



val PATH="/home/usuario/Scala/Proyecto4/"
val FILE_CENSUS="census-income.data"
val FILE_CENSUS_TEST="census-income.test"



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






val census_df = spark.read.format("csv").
option("delimiter", ",").option("ignoreLeadingWhiteSpace","true").
schema(censusSchema).load(PATH + FILE_CENSUS)



val census_df_test = spark.read.format("csv").
option("delimiter", ",").option("ignoreLeadingWhiteSpace","true").
schema(censusSchema).load(PATH + FILE_CENSUS_TEST)










//cargamos datasets------------------------
import TransformDataframe._
import CleanDataframe._
val census_df_limpio=cleanDataframe(census_df)
val trainCensusDFProcesado = transformDataFrame(census_df_limpio)


val census_df_limpio=cleanDataframe(census_df_test)
val testCensusDF = transformDataFrame(census_df_limpio)













//validacion cruzada y parametros-----------------
val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features")
val paramGrid = new ParamGridBuilder().addGrid(rf.numTrees, Array(10, 20)).addGrid(rf.maxDepth, Array(5, 10)).build()
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val cv = new CrossValidator().setEstimator(rf).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(2)  
val cvModel = cv.fit(trainCensusDFProcesado)
import org.apache.spark.ml.classification.RandomForestClassificationModel
val bestModel = cvModel.bestModel.asInstanceOf[RandomForestClassificationModel]
println("Best Model:")
println(s" - Number of Trees: ${bestModel.getNumTrees}")
println(s" - Max Depth: ${bestModel.getMaxDepth}")









//utilizamos mejores parametros------------------------------
val RFcar = new RandomForestClassifier().setFeaturesCol("features").
 setLabelCol("label").
 setNumTrees(bestModel.getNumTrees).
 setMaxDepth(bestModel.getMaxDepth).
 setMaxBins(10).
 setMinInstancesPerNode(1).
 setMinInfoGain(0.0).
 setCacheNodeIds(false).
 setCheckpointInterval(10)

val RFcarModel_D =RFcar.fit(trainCensusDFProcesado)
RFcarModel_D.toDebugString
val predictionsAndLabelsDF_RF = RFcarModel_D.transform(testCensusDF).select("prediction", "label","rawPrediction", "probability")

val predictions = RFcarModel_D.transform(testCensusDF).select("prediction").rdd.map(_.getDouble(0))
val labels = RFcarModel_D.transform(testCensusDF).select("label").rdd.map(_.getDouble(0))













//metricas---------------------------------
val metrics = new MulticlassMetrics(predictions.zip(labels))

println("Confusion matrix:")
println(metrics.confusionMatrix)

val accuracy = metrics.accuracy
println("Summary Statistics")
println(f"Accuracy = $accuracy%1.4f")

val labels = metrics.labels
labels.foreach {l => val pl = metrics.precision(l) 
        println(f"PrecisionByLabel($l) = $pl%1.4f")}

labels.foreach {l => val fpl = metrics.falsePositiveRate(l)
        println(f"falsePositiveRate($l) = $fpl%1.4f")}

labels.foreach {l => val fpl = metrics.truePositiveRate(l)
        println(f"truePositiveRate($l) = $fpl%1.4f")}
























//curva roc---------------------------------------------
val probabilitiesAndLabelsRDD = predictionsAndLabelsDF_RF.select("label", "probability").rdd.map{row => (row.getAs[Vector](1).toArray, row.getDouble(0))}.map{r => ( r._1(1), r._2)}

val MLlib_binarymetrics = new BinaryClassificationMetrics(probabilitiesAndLabelsRDD,15)

val MLlib_auROC = MLlib_binarymetrics.areaUnderROC
println(f"%nAUC de la curva ROC para la clase income")
println(f"con MLlib, métrica binaria, probabilitiesAndLAbelsRDD, 15 bins: $MLlib_auROC%1.4f%n")

val MLlib_auPR = MLlib_binarymetrics.areaUnderPR
println(f"%nAUC de la curva PR para la clase income")
println(f"con MLlib, métrica binaria, probabilitiesAndLAbelsRDD, 15 bins: $MLlib_auPR%1.4f%n")

val MLlib_curvaROC =MLlib_binarymetrics.roc
println("Puntos para construir curva ROC con MLlib, probabilitiesAndLabelsRDD, 15 bins:")
MLlib_curvaROC.take(17).foreach(x => println(x))