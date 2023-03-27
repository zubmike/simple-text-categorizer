package com.github.zubmike.nn;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.shade.guava.base.Strings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Properties;

public class App {

	private static final Logger LOGGER = LoggerFactory.getLogger(App.class);

	private static final String PROPERTIES_FILE_NAME = "app.properties";

	public static void main(String[] args) {
		var properties = loadProperties();
		var dictionary = createTextVectorizer(properties);
		var recreate = Boolean.parseBoolean(properties.getProperty("config.recreate", "false"));
		var network = getNeuralNetwork(properties, dictionary, recreate);
		inputConsole(dictionary, network);
	}

	private static Properties loadProperties() {
		var file = new File(PROPERTIES_FILE_NAME);
		if (!file.exists()) {
			throw new RuntimeException("properties are not found");
		}
		try (var inputStream = new FileInputStream(file)) {
			var properties = new Properties();
			properties.load(inputStream);
			LOGGER.info("properties:\n{}", properties);
			return properties;
		} catch (Exception e) {
			throw new RuntimeException("can't load properties", e);
		}
	}

	private static TextVectorizer createTextVectorizer(Properties properties) {
		var textVectorizer = new TextVectorizer(
				TextVectorMethod.valueOf(properties.getProperty("config.textVectorMethod")),
				List.of(new ReplacingRule("\\d(,)\\d", ".", 1),
						new ReplacingRule("[\\[\\]\\(\\):;,\\/\"\\-*%#&$]", " ")));

		textVectorizer.fill(properties.getProperty("file.corpusName"));

		// optional saving (for analysing)
		var vocabularyFileName = properties.getProperty("file.vocabularyName");
		if (!Strings.isNullOrEmpty(vocabularyFileName)) {
			textVectorizer.save(vocabularyFileName);
		}

		return textVectorizer;
	}

	private static MultiLayerNetwork getNeuralNetwork(Properties properties, TextVectorizer dictionary, boolean recreate) {
		var neuralNetworkModelFileName = properties.getProperty("file.neuralNetworkModelName");
		if (recreate) {
			var configuration = TextNeuralNetwork.createConfiguration(
					Double.parseDouble(properties.getProperty("config.learningRate", "0.1")),
					dictionary.getVocabularySize(),
					Integer.parseInt(properties.getProperty("config.hiddenLayerSize", "10")),
					dictionary.getCategorySize());
			var neuralNetwork = new TextNeuralNetwork(configuration);
			neuralNetwork.init();
			neuralNetwork.setListeners(new ScoreIterationListener(100));
			LOGGER.info(neuralNetwork.summary());

			LOGGER.info("start learning");
			neuralNetwork.learn(
					dictionary.createDataSet(),
					Integer.parseInt(properties.getProperty("config.epochs", "1000")));
			LOGGER.info("finish learning");
			neuralNetwork.save(neuralNetworkModelFileName);

			return neuralNetwork;
		} else {
			var network = TextNeuralNetwork.load(neuralNetworkModelFileName);
			LOGGER.info(network.summary());
			return network;
		}
	}

	private static void inputConsole(TextVectorizer dictionary, MultiLayerNetwork network) {
		try (var reader = new BufferedReader(new InputStreamReader(System.in))) {
			while (true) {
				System.out.print("Input: ");
				var command = reader.readLine();
				if (command.equals("exit")) {
					System.exit(0);
				} else {
					var output = network.output(dictionary.parse(command));
					System.out.println(dictionary.parseCategory(output));
				}
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

}
