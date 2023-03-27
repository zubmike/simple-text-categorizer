package com.github.zubmike.nn;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Sgd;

import java.io.IOException;

public class TextNeuralNetwork extends MultiLayerNetwork {

	public TextNeuralNetwork(MultiLayerConfiguration configuration) {
		super(configuration);
	}

	public static MultiLayerConfiguration createConfiguration(double learningRate, int inputSize, int hiddenLayerSize, int outputSize) {
		return new NeuralNetConfiguration.Builder()
				.updater(new Sgd(learningRate))
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(inputSize)
						.nOut(hiddenLayerSize)
						.activation(Activation.SIGMOID)
						.build())
				.layer(new OutputLayer.Builder()
						.nOut(outputSize)
						.activation(Activation.SOFTMAX)
						.build())
				.build();
	}

	public void learn(DataSet dataSet, int epochs) {
		for (int i = 0; i < epochs; i++) {
			fit(dataSet);
		}
	}

	public void save(String fileName) {
		try {
			ModelSerializer.writeModel(this, fileName, true);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public static MultiLayerNetwork load(String fileName) {
		try {
			return ModelSerializer.restoreMultiLayerNetwork(fileName);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

}
