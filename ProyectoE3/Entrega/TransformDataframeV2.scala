/* Master en Ingeniería Informática - Universidad de Valladolid
*
*  TECNICAS ESCLABLES DE ANÁLISIS DE DATOS EN ENTORNOS BIG DATA: CLASIFICADORES
*  Proyecto de clasificación. Tercera etapa: Creación, selección y evaluación de modelos
*
*  Script para la transformar el DataFrame
*  En esta versión (V2) omitimos el uso de OneHotEncoder
*
*  Grupo 2: Sergio Agudelo Bernal
*           Miguel Ángel Collado Alonso
*           José María Lozano Olmedo.
*/


import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler


object TransformDataframeV2 {

	def transformDataFrame(census_df: DataFrame): DataFrame = {

		val attributeColumns = census_df.columns.toSeq.filter(_ != "income").toArray
		
		val outputColumns = attributeColumns.map(_ + "-num").toArray
		
		val siColumns= new StringIndexer().setInputCols(attributeColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")
		
		val simColumns = siColumns.fit(census_df)
		
		val censusDFnumeric = simColumns.transform(census_df).drop(attributeColumns:_*)
		
		val va = new VectorAssembler().setOutputCol("features").setInputCols(outputColumns)

		val censusFeaturesClaseDF = va.transform(censusDFnumeric).select("features", "income")

		val indiceClase= new StringIndexer().setInputCol("income").setOutputCol("label").setStringOrderType("alphabetDesc")

		val censusFeaturesLabelDF = indiceClase.fit(censusFeaturesClaseDF).transform(censusFeaturesClaseDF).drop("income")

		censusFeaturesLabelDF
	}

}