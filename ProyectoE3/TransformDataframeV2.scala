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