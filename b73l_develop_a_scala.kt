package com.b73l.develop.scala

import org.jetbrains.kotlinx.dataframe.Dataframe
import org.jetbrains.kotlinx.dataframe.api.*
import org.jetbrains.kotlinx.linear.*
import org.jetbrains.kotlinx.ml.api.*
import org.jetbrains.kotlinx.ml.core.Distance
import org.jetbrains.kotlinx.ml.core.OnlineLearning
import org.jetbrains.kotlinx.ml.dsl.*

data class ModelConfig(
    val epochs: Int,
    val learningRate: Double,
    val batchSize: Int,
    val hiddenLayers: List<Int>,
    val activationFunctions: List<ActivationFunction>,
    val optimizer: Optimizer
)

data class TrainingData(
    val features: Dataframe,
    val labels: Dataframe
)

data class ModelController(
    val config: ModelConfig,
    val trainingData: TrainingData
) {
    private val model: OnlineLearning.Model = createModel(config)

    fun train() {
        println("Training model...")
        model.train(trainingData.features, trainingData.labels)
        println("Model trained successfully!")
    }

    fun predict(features: Dataframe): Dataframe {
        println("Making predictions...")
        val predictions = model.predict(features)
        println("Predictions made successfully!")
        return predictions
    }

    private fun createModel(config: ModelConfig): OnlineLearning.Model {
        val layers = config.hiddenLayers.mapIndexed { index, size ->
            LinearLayer(size, activation = config.activationFunctions[index])
        }
        return OnlineLearning.Model(layers, config.epochs, config.learningRate, config.batchSize, config.optimizer)
    }
}