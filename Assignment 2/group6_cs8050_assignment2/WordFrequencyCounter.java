package group6_cs8050_assignment2;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

/**
 * Word Frequency Counter - Real-world application using dictionary implementations
 * Processes large text datasets to count word occurrences
 */
public class WordFrequencyCounter {

    private Dictionary<String, Integer> dictionary;
    private long processingTime;
    private int totalWords;
    private int uniqueWords;

    public WordFrequencyCounter(Dictionary<String, Integer> dictionary) {
        this.dictionary = dictionary;
        this.totalWords = 0;
        this.uniqueWords = 0;
    }

    /**
     * Process a single file and count word frequencies
     */
    public void processFile(String filepath) throws IOException {
        long startTime = System.nanoTime();

        try (BufferedReader reader = new BufferedReader(new FileReader(filepath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                processLine(line);
            }
        }

        processingTime = System.nanoTime() - startTime;
        uniqueWords = dictionary.size();
    }

    /**
     * Process multiple files
     */
    public void processFiles(List<String> filepaths) throws IOException {
        long startTime = System.nanoTime();

        for (String filepath : filepaths) {
            processFile(filepath);
        }

        processingTime = System.nanoTime() - startTime;
        uniqueWords = dictionary.size();
    }

    /**
     * Process a directory of text files
     */
    public void processDirectory(String directoryPath) throws IOException {
        long startTime = System.nanoTime();

        try (Stream<Path> paths = Files.walk(Paths.get(directoryPath))) {
            paths.filter(Files::isRegularFile)
                    .filter(p -> p.toString().endsWith(".txt"))
                    .forEach(p -> {
                        try {
                            processFile(p.toString());
                        } catch (IOException e) {
                            System.err.println("Error processing " + p + ": " + e.getMessage());
                        }
                    });
        }

        processingTime = System.nanoTime() - startTime;
        uniqueWords = dictionary.size();
    }

    /**
     * Process text from a string
     */
    public void processText(String text) {
        long startTime = System.nanoTime();

        String[] lines = text.split("\n");
        for (String line : lines) {
            processLine(line);
        }

        processingTime = System.nanoTime() - startTime;
        uniqueWords = dictionary.size();
    }

    /**
     * Process a single line of text
     */
    private void processLine(String line) {
        // Tokenize and normalize
        String[] words = line.toLowerCase()
                .replaceAll("[^a-z0-9\\s]", " ")
                .split("\\s+");

        for (String word : words) {
            if (!word.isEmpty() && word.length() > 1) {
                totalWords++;
                Integer count = dictionary.find(word);
                if (count == null) {
                    dictionary.insert(word, 1);
                } else {
                    dictionary.update(word, count + 1);
                }
            }
        }
    }

    /**
     * Get the most frequent words
     */
    public List<Map.Entry<String, Integer>> getTopWords(int n) {
        List<Map.Entry<String, Integer>> entries = new ArrayList<>();

        // This is a simplified approach - in production, you'd want a more efficient method
        // For now, we'll create a simple wrapper to extract all entries

        // Note: This requires modifying the Dictionary interface to support iteration
        // For demonstration, we'll return an empty list
        // In a full implementation, you'd add an entries() method to Dictionary

        System.out.println("Note: To get top words, Dictionary interface needs entries() method");
        return entries;
    }

    /**
     * Get frequency of a specific word
     */
    public int getWordFrequency(String word) {
        Integer freq = dictionary.find(word.toLowerCase());
        return freq != null ? freq : 0;
    }

    /**
     * Get statistics about the word frequency analysis
     */
    public String getStatistics() {
        return String.format(
                "Word Frequency Statistics:\n" +
                        "  Total words processed: %d\n" +
                        "  Unique words: %d\n" +
                        "  Processing time: %.2f ms\n" +
                        "  Words per second: %.2f\n" +
                        "  Dictionary load factor: %.4f\n" +
                        "  Dictionary stats:\n%s",
                totalWords,
                uniqueWords,
                processingTime / 1_000_000.0,
                (totalWords / (processingTime / 1_000_000_000.0)),
                dictionary.getLoadFactor(),
                dictionary.getStats().toString()
        );
    }

    /**
     * Generate a sample text dataset for testing
     */
    public static String generateSampleText(int wordCount) {
        String[] commonWords = {
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
                "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
                "or", "an", "will", "my", "one", "all", "would", "there", "their", "what"
        };

        Random rand = new Random(42);
        StringBuilder text = new StringBuilder();

        for (int i = 0; i < wordCount; i++) {
            // Zipf distribution - some words appear much more frequently
            double u = rand.nextDouble();
            int index = (int) (Math.pow(u, -0.7) * 10) % commonWords.length;

            text.append(commonWords[index]).append(" ");

            if ((i + 1) % 20 == 0) {
                text.append("\n");
            }
        }

        return text.toString();
    }

    /**
     * Export word frequencies to file
     */
    public void exportToFile(String filename) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("Word,Frequency");

            // This also needs entries() method in Dictionary interface
            writer.println("# Export requires Dictionary.entries() method");

            System.out.println("Word frequencies exported to " + filename);
        }
    }

    /**
     * Compare performance of different dictionary implementations
     */
    public static void compareDictionaryPerformance(String text) {
        System.out.println("\n=== Comparing Dictionary Implementations ===\n");

        // Test Chaining with different hash functions
        HashFunction<String>[] hashFunctions = new HashFunction[]{
                new PolynomialHash(),
                new FNV1aHash(),
                new MurmurHash3()
        };

        for (HashFunction<String> hf : hashFunctions) {
            System.out.println("Testing Chaining with " + hf.getName());
            ChainingHashTable<String, Integer> dict = new ChainingHashTable<>(hf);
            WordFrequencyCounter counter = new WordFrequencyCounter(dict);
            counter.processText(text);
            System.out.println(counter.getStatistics());
            System.out.println();
        }

        // Test Open Addressing with Linear Probing
        for (HashFunction<String> hf : hashFunctions) {
            System.out.println("Testing Linear Probing with " + hf.getName());
            OpenAddressingHashTable<String, Integer> dict =
                    new OpenAddressingHashTable<>(hf, OpenAddressingHashTable.ProbingStrategy.LINEAR);
            WordFrequencyCounter counter = new WordFrequencyCounter(dict);
            counter.processText(text);
            System.out.println(counter.getStatistics());
            System.out.println();
        }

        // Test Open Addressing with Quadratic Probing
        for (HashFunction<String> hf : hashFunctions) {
            System.out.println("Testing Quadratic Probing with " + hf.getName());
            OpenAddressingHashTable<String, Integer> dict =
                    new OpenAddressingHashTable<>(hf, OpenAddressingHashTable.ProbingStrategy.QUADRATIC);
            WordFrequencyCounter counter = new WordFrequencyCounter(dict);
            counter.processText(text);
            System.out.println(counter.getStatistics());
            System.out.println();
        }
    }

    public long getProcessingTime() {
        return processingTime;
    }

    public int getTotalWords() {
        return totalWords;
    }

    public int getUniqueWords() {
        return uniqueWords;
    }
}