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


object CleanDataframe {

def cleanDataframe(census_df: DataFrame): DataFrame = {

    /*val eliminarColumnas = Seq("fill_inc_questionnaire_for_veterans_ad", "enrolled_in_edu_last_wk", "member_of_labor_union","reason_for_unemployment")
    val nuevoDataFrame = census_df.drop(eliminarColumnas: _*)


    nuevoDataFrame*/

    val nuevoDataFrame = census_df.select("age","class_of_worker","education","marital_status","citizenship","income")
    nuevoDataFrame
}
}