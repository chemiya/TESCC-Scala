object TransformDataframe {
    def transformDataFrame(census_df: DataFrame): DataFrame = {
        var attributeColumns = census_df.columns.toSeq.filter(_ != "income").toArray
        var outputColumns = attributeColumns.map(_ + "-num").toArray

        var siColumns= new StringIndexer().setInputCols(attributeColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

        var simColumns = siColumns.fit(census_df)

        var census_df_numeric = simColumns.transform(census_df).drop(attributeColumns:_*)

        //census_df_numeric.show(10)

        // Generamos los nombres de las nuevas columnas
        var inputColumns = outputColumns
        outputColumns = attributeColumns.map(_ + "-hot").toArray

        // Creamos OneHotEncoder para transformar todos los atributos, salvo la clase
        var hotColumns = new OneHotEncoder().setInputCols(inputColumns).setOutputCols(outputColumns)

        //Creamos el OneHotEncoderModel
        var hotmColumns = hotColumns.fit(census_df_numeric)

        var censusDFhot = hotmColumns.transform(census_df_numeric).drop(inputColumns:_*)

        // Lo examinamos
        //censusDFhot.show(10)

        //  Definimos columna Features con todos los atributos, menos la clase, con VectorAssempler
        //var va = new VectorAssembler().setOutputCol("features").setInputCols(carDFhot.columns.diff(Array("clase")))
        var va = new VectorAssembler().setOutputCol("features").setInputCols(outputColumns)

        // Creamos el DataFrame carFeaturesClaseDF con columnas features y clase
        var censusFeaturesClaseDF = va.transform(censusDFhot).select("features", "income")

        // Lo examinamos
        //censusFeaturesClaseDF.show(10)

        // creamos el StringIndexer para la clase
        var indiceClase= new StringIndexer().setInputCol("income").setOutputCol("label").setStringOrderType("alphabetDesc")

        // Creamos el DataFrame carFeaturesLabelDF con columnas features y label
        var censusFeaturesLabelDF = indiceClase.fit(censusFeaturesClaseDF).transform(censusFeaturesClaseDF).drop("income")
        //censusFeaturesLabelDF.show(10)

        censusFeaturesLabelDF
    }

    def transformDataFrame2(census_df: DataFrame, continuous_cols: Array[String]): DataFrame = {
        // Filtra atributos continuos:
        var attributeColumns = census_df.columns.toSeq.filter(_ != "income").filter(!continuous_cols.contains(_)).toArray
        var outputColumns = attributeColumns.map(_ + "-num").toArray

        var siColumns= new StringIndexer().setInputCols(attributeColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")
        var simColumns = siColumns.fit(census_df)
        var census_df_numeric = simColumns.transform(census_df).drop(attributeColumns:_*)

        // census_df_numeric.show(10)

        // Generamos los nombres de las nuevas columnas
        var inputColumns = outputColumns
        outputColumns = attributeColumns.map(_ + "-hot").toArray

        // Creamos OneHotEncoder para transformar todos los atributos, salvo la clase
        var hotColumns = new OneHotEncoder().setInputCols(inputColumns).setOutputCols(outputColumns)

        //Creamos el OneHotEncoderModel
        var hotmColumns = hotColumns.fit(census_df_numeric)

        var censusDFhot = hotmColumns.transform(census_df_numeric).drop(inputColumns:_*)

        // Lo examinamos
        // censusDFhot.show(10)

        //  Definimos columna Features con todos los atributos, menos la clase, con VectorAssempler
        var va = new VectorAssembler().setOutputCol("features").setInputCols(censusDFhot.columns.diff(Array("income")))

        // Creamos el DataFrame carFeaturesClaseDF con columnas features y clase
        var censusFeaturesClaseDF = va.transform(censusDFhot).select("features", "income")

        // Lo examinamos
        // censusFeaturesClaseDF.show(10)

        // creamos el StringIndexer para la clase
        var indiceClase= new StringIndexer().setInputCol("income").setOutputCol("label").setStringOrderType("alphabetDesc")

        // Creamos el DataFrame carFeaturesLabelDF con columnas features y label
        var censusFeaturesLabelDF = indiceClase.fit(censusFeaturesClaseDF).transform(censusFeaturesClaseDF).drop("income")
        // censusFeaturesLabelDF.show(10)

        censusFeaturesLabelDF
    }
}

