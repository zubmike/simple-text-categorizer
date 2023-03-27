package com.github.zubmike.nn;

public class ReplacingRule {

	private final String regex;
	private final String replacement;
	private final int replacedGroup;

	public ReplacingRule(String regex, String replacement) {
		this(regex, replacement, 0);
	}

	public ReplacingRule(String regex, String replacement, int replacedGroupIndex) {
		this.regex = regex;
		this.replacement = replacement;
		this.replacedGroup = replacedGroupIndex;
	}

	public String getRegex() {
		return regex;
	}

	public String getReplacement() {
		return replacement;
	}

	public int getReplacedGroup() {
		return replacedGroup;
	}
}
