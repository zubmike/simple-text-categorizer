package com.github.zubmike.nn;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.io.Files;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class TextVectorizer {

	private static final Logger LOGGER = LoggerFactory.getLogger(TextVectorizer.class);

	private static final char CSV_DELIMITER = '\t';
	private static final String SPLITTER = " ";

	private final TextVectorMethod textVectorMethod;
	private final List<ReplacingRule> replacingRules;

	private final Map<String, List<String>> categoryTextMap = new LinkedHashMap<>();
	private final List<String> categories = new ArrayList<>();
	private final Map<String, Integer> wordIndexMap = new LinkedHashMap<>();
	private final Map<String, Integer> wordCountMap = new LinkedHashMap<>();
	private int textCount;

	public TextVectorizer() {
		this(TextVectorMethod.BOW, Collections.emptyList());
	}

	public TextVectorizer(TextVectorMethod textVectorMethod, List<ReplacingRule> replacingRules) {
		this.textVectorMethod = textVectorMethod;
		this.replacingRules = replacingRules;
	}

	public void fill(String corpusFileName) {
		var csvFormat = CSVFormat.DEFAULT.builder()
				.setDelimiter(CSV_DELIMITER)
				.build();
		try (var reader = new FileReader(corpusFileName, StandardCharsets.UTF_8);
			 var parser = csvFormat.parse(reader)) {
			fill(parser);
			fillCategories();
			fillVocabulary();
		} catch (Exception e) {
			throw new RuntimeException("can't parse corpus", e);
		}
	}

	private void fill(CSVParser parser) {
		for (var record : parser) {
			var type = record.get(0);
			var text = prepareText(record.get(1));
			categoryTextMap.computeIfAbsent(type, key -> new ArrayList<>()).add(text);
			textCount++;
		}
	}

	private void fillCategories() {
		categories.addAll(categoryTextMap.keySet());
		LOGGER.info("filled categories: {}", categories.size());
	}

	private void fillVocabulary() {
		var vocabularySet = new LinkedHashSet<String>();
		categoryTextMap.forEach((category, texts) -> {
			var categoryAllWords = new ArrayList<String>();
			texts.forEach(text -> {
				var words = Arrays.asList(text.split(SPLITTER));
				vocabularySet.addAll(words);
				categoryAllWords.addAll(words);
			});
			categoryAllWords
					.stream()
					.collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
					.forEach((word, count) -> {
						var prevCount = wordCountMap.getOrDefault(word, 0);
						wordCountMap.put(word, prevCount + count.intValue());
					});
		});
		int i = 0;
		for (var word : vocabularySet) {
			wordIndexMap.put(word, i);
			i++;
		}
		LOGGER.info("filled vocabulary: {}", wordIndexMap.size());
	}

	public DataSet createDataSet() {
		var textCount = categoryTextMap.values().stream().mapToInt(List::size).sum();
		var inputArray = new double [textCount][wordIndexMap.size()];
		var targetArray = new double [textCount][categories.size()];

		int rowIndex = 0;
		for (Map.Entry<String, List<String>> entry : categoryTextMap.entrySet()) {
			var category = entry.getKey();
			var categoryIndex = categories.indexOf(category);
			for (var text : entry.getValue()) {
				switch (textVectorMethod) {
					case BOW -> fillInputArrayByBow(inputArray, text, rowIndex);
					case TF_IDF -> fillInputArrayByTfIdf(inputArray, text, textCount, rowIndex);
				}
				targetArray[rowIndex][categoryIndex] = 1;
				rowIndex++;
			}
		}

		return new DataSet(Nd4j.createFromArray(inputArray), Nd4j.createFromArray(targetArray));
	}

	private void fillInputArrayByBow(double[][] inputArray, String text, int rowIndex) {
		for (var word : text.split(SPLITTER)) {
			var columnIndex = wordIndexMap.get(word);
			if (columnIndex != null) {
				inputArray[rowIndex][columnIndex] += 1;
			}
		}
	}

	private void fillInputArrayByTfIdf(double[][] inputArray, String text, int textCount, int rowIndex) {
		var wordArray = text.split(SPLITTER);
		var textWordCountMap = Arrays.stream(wordArray)
				.collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
		for (var word : wordArray) {
			var columnIndex = wordIndexMap.get(word);
			if (columnIndex != null) {
				double tf = textWordCountMap.get(word).doubleValue() / wordArray.length;
				double idf = Math.log(textCount / wordCountMap.get(word).doubleValue());
				inputArray[rowIndex][columnIndex] = tf * idf;
			}
		}
	}

	public INDArray parse(String text) {
		var preparedText = prepareText(text);
		var inputArray = new double[1][wordIndexMap.size()];
		switch (textVectorMethod) {
			case BOW -> fillInputArrayByBow(inputArray, preparedText, 0);
			case TF_IDF -> fillInputArrayByTfIdf(inputArray, preparedText, textCount, 0);
		}
		return Nd4j.createFromArray(inputArray);
	}

	private String prepareText(String origin) {
		var text = origin.toUpperCase();
		for (var rule : replacingRules) {
			if (rule.getReplacedGroup() == 0) {
				text = text.replaceAll(rule.getRegex(), rule.getReplacement());
			} else {
				var builder = new StringBuilder();
				var matcher = Pattern.compile(rule.getRegex()).matcher(text);
				while (matcher.find()) {
					var group0 = matcher.group(0);
					var group = matcher.group(rule.getReplacedGroup());
					var quote = Pattern.quote(group);
					var replacement = group0.replaceAll(quote, rule.getReplacement());
					matcher.appendReplacement(builder, replacement);
				}
				matcher.appendTail(builder);
				text = builder.toString();
			}
		}
		return text.replaceAll("\\s+", " ").trim();
	}

	public String parseCategory(INDArray output) {
		int maxIndex = 0;
		double maxValue = 0;
		for (int i = 0; i < output.length(); i++) {
			var value = output.getColumn(i).getDouble();
			if (value > maxValue) {
				maxValue = value;
				maxIndex = i;
			}
		}
		if (maxValue >= 0.9) {
			return categories.get(maxIndex);
		} else {
			return "Unknown";
		}
	}

	public void save(String vocabularyFileName) {
		try {
			Files.write(String.join(System.lineSeparator(), wordIndexMap.keySet()).getBytes(StandardCharsets.UTF_8), new File(vocabularyFileName));
		} catch (IOException e) {
			throw new RuntimeException("can't save vocabulary", e);
		}
	}

	public int getVocabularySize() {
		return wordIndexMap.size();
	}

	public int getCategorySize() {
		return categories.size();
	}

}
