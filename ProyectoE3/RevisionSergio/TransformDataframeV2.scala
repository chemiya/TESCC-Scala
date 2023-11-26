object TransformDataframeV2 {
  def transformDataFrame(census_df: DataFrame): DataFrame = {
    // Seleccionar todas las columnas, excluyendo la clase
    val attributeColumns = census_df.columns.toSeq.filter(_ != "income").toArray
    val outputColumns = attributeColumns.map(_ + "-num").toArray



    // StringIndexer de los atributos
    val siColumns= new StringIndexer().setInputCols(attributeColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")
    val simColumns = siColumns.fit(census_df)
    val censusDFnumeric = simColumns.transform(census_df).drop(attributeColumns:_*)


    // VectorAssembler de los atributos y la clase
    val va = new VectorAssembler().setOutputCol("features").setInputCols(outputColumns)
    val censusFeaturesClaseDF = va.transform(censusDFnumeric).select("features", "income")


    // Agregando ahora StringIndexer de la clase
    val indiceClase= new StringIndexer().setInputCol("income").setOutputCol("label").setStringOrderType("alphabetDesc")
    val censusFeaturesLabelDF = indiceClase.fit(censusFeaturesClaseDF).transform(censusFeaturesClaseDF).drop("income")

    censusFeaturesLabelDF
  }
}