package group6_cs8050_assignment2;

import java.util.*;

/**
 * Main Driver Class for Assignment 2
 * Demonstrates all dictionary implementations and applications
 */
public class MainDriver {

    public static void main(String[] args) {
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║   CS 8050 - Assignment 2: Advanced Dictionaries & Hashing ║");
        System.out.println("║   Dictionary Implementation & Analysis                     ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝\n");

        // Run all demonstrations
        demonstrateBasicOperations();
        demonstrateHashFunctions();
        demonstrateProbingStrategies();
        demonstrateWordFrequencyCounter();
        runBenchmarkSuite();
        demonstrateLoadFactorImpact();

        System.out.println("\n╔════════════════════════════════════════════════════════════╗");
        System.out.println("║   All demonstrations completed successfully!               ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
    }

    /**
     * Demonstrate basic dictionary operations
     */
    private static void demonstrateBasicOperations() {
        System.out.println("\n=== 1. Basic Dictionary Operations ===\n");

        // Create a chaining hash table
        ChainingHashTable<String, Integer> dict = new ChainingHashTable<>(new PolynomialHash());

        System.out.println("Inserting key-value pairs...");
        dict.insert("apple", 5);
        dict.insert("banana", 3);
        dict.insert("cherry", 8);
        dict.insert("date", 2);
        dict.insert("elderberry", 12);

        System.out.println("Dictionary size: " + dict.size());
        System.out.println("Load factor: " + String.format("%.4f", dict.getLoadFactor()));

        System.out.println("\nFinding values...");
        System.out.println("apple: " + dict.find("apple"));
        System.out.println("banana: " + dict.find("banana"));
        System.out.println("grape (not exists): " + dict.find("grape"));

        System.out.println("\nUpdating values...");
        dict.update("apple", 10);
        System.out.println("apple (after update): " + dict.find("apple"));

        System.out.println("\nDeleting entries...");
        Integer deletedValue = dict.delete("banana");
        System.out.println("Deleted banana with value: " + deletedValue);
        System.out.println("Dictionary size after deletion: " + dict.size());

        System.out.println("\n" + dict.getStats());
    }

    /**
     * Demonstrate different hash functions
     */
    private static void demonstrateHashFunctions() {
        System.out.println("\n=== 2. Hash Function Comparison ===\n");

        String[] testKeys = {"algorithm", "data", "structure", "hash", "collision",
                "probing", "chaining", "dictionary", "performance", "analysis"};

        HashFunction<String>[] hashFunctions = new HashFunction[]{
                new PolynomialHash(),
                new FNV1aHash(),
                new MurmurHash3(),
                new SHA256Hash()
        };

        int tableSize = 17; // Prime number

        for (HashFunction<String> hf : hashFunctions) {
            System.out.println("Testing " + hf.getName() + ":");
            Map<Integer, List<String>> buckets = new HashMap<>();

            for (String key : testKeys) {
                int hash = hf.hash(key, tableSize);
                buckets.computeIfAbsent(hash, k -> new ArrayList<>()).add(key);
            }

            int collisions = 0;
            for (List<String> bucket : buckets.values()) {
                if (bucket.size() > 1) {
                    collisions += bucket.size() - 1;
                }
            }

            System.out.println("  Buckets used: " + buckets.size() + "/" + tableSize);
            System.out.println("  Collisions: " + collisions);
            System.out.println("  Distribution: " + buckets);
            System.out.println();
        }
    }

    /**
     * Demonstrate different probing strategies
     */
    private static void demonstrateProbingStrategies() {
        System.out.println("\n=== 3. Probing Strategy Comparison ===\n");

        String[] keys = new String[50];
        for (int i = 0; i < keys.length; i++) {
            keys[i] = "key_" + i;
        }

        HashFunction<String> hf = new MurmurHash3();

        // Linear Probing
        System.out.println("Testing Linear Probing:");
        OpenAddressingHashTable<String, Integer> linearDict =
                new OpenAddressingHashTable<>(16, 0.7, hf,
                        OpenAddressingHashTable.ProbingStrategy.LINEAR);

        for (int i = 0; i < keys.length; i++) {
            linearDict.insert(keys[i], i);
        }

        System.out.println(linearDict.getStats());

        // Quadratic Probing
        System.out.println("\nTesting Quadratic Probing:");
        OpenAddressingHashTable<String, Integer> quadraticDict =
                new OpenAddressingHashTable<>(16, 0.7, hf,
                        OpenAddressingHashTable.ProbingStrategy.QUADRATIC);

        for (int i = 0; i < keys.length; i++) {
            quadraticDict.insert(keys[i], i);
        }

        System.out.println(quadraticDict.getStats());
    }

    /**
     * Demonstrate Word Frequency Counter application
     */
    private static void demonstrateWordFrequencyCounter() {
        System.out.println("\n=== 4. Word Frequency Counter Application ===\n");

        // Generate sample text
        System.out.println("Generating sample text dataset...");
        String sampleText = WordFrequencyCounter.generateSampleText(100000);

        System.out.println("Sample text size: " + sampleText.length() + " characters\n");

        // Test with Chaining Hash Table
        System.out.println("Testing with Chaining Hash Table (MurmurHash3):");
        ChainingHashTable<String, Integer> chainingDict = new ChainingHashTable<>(new MurmurHash3());
        WordFrequencyCounter counter1 = new WordFrequencyCounter(chainingDict);
        counter1.processText(sampleText);
        System.out.println(counter1.getStatistics());

        System.out.println("\nTesting with Open Addressing (Linear Probing, FNV1a):");
        OpenAddressingHashTable<String, Integer> openDict =
                new OpenAddressingHashTable<>(new FNV1aHash(),
                        OpenAddressingHashTable.ProbingStrategy.LINEAR);
        WordFrequencyCounter counter2 = new WordFrequencyCounter(openDict);
        counter2.processText(sampleText);
        System.out.println(counter2.getStatistics());

        // Test some specific words
        System.out.println("\n--- Sample Word Frequencies ---");
        String[] testWords = {"the", "be", "to", "of", "and"};
        for (String word : testWords) {
            System.out.println(word + ": " + counter1.getWordFrequency(word));
        }
    }

    /**
     * Run comprehensive benchmark suite
     */
    private static void runBenchmarkSuite() {
        System.out.println("\n=== 5. Comprehensive Benchmark Suite ===\n");
        System.out.println("Running limited benchmark suite (use BenchmarkSuite.runFullSuite() for complete tests)...\n");

        // Run a smaller version for demonstration
        List<String> keys = new ArrayList<>();
        Random rand = new Random(42);
        for (int i = 0; i < 10000; i++) {
            keys.add("key_" + rand.nextInt(50000));
        }

        double[] loadFactors = {0.5, 0.75};
        HashFunction<String>[] hashFunctions = new HashFunction[]{
                new PolynomialHash(),
                new MurmurHash3()
        };

        System.out.println("Testing with 10,000 keys...\n");

        for (double lf : loadFactors) {
            for (HashFunction<String> hf : hashFunctions) {
                // Test Chaining
                ChainingHashTable<String, Integer> chainingDict =
                        new ChainingHashTable<>(16, lf, hf);

                long startTime = System.nanoTime();
                for (int i = 0; i < keys.size(); i++) {
                    chainingDict.insert(keys.get(i), i);
                }
                long insertTime = System.nanoTime() - startTime;

                startTime = System.nanoTime();
                for (String key : keys) {
                    chainingDict.find(key);
                }
                long findTime = System.nanoTime() - startTime;

                System.out.printf("Chaining | %s | LF: %.2f | Insert: %.2f ms | Find: %.2f ms | Collisions: %.4f%n",
                        hf.getName(), lf, insertTime / 1_000_000.0, findTime / 1_000_000.0,
                        chainingDict.getStats().getCollisionRate());

                // Test Linear Probing
                OpenAddressingHashTable<String, Integer> linearDict =
                        new OpenAddressingHashTable<>(16, lf, hf,
                                OpenAddressingHashTable.ProbingStrategy.LINEAR);

                startTime = System.nanoTime();
                for (int i = 0; i < keys.size(); i++) {
                    linearDict.insert(keys.get(i), i);
                }
                insertTime = System.nanoTime() - startTime;

                startTime = System.nanoTime();
                for (String key : keys) {
                    linearDict.find(key);
                }
                findTime = System.nanoTime() - startTime;

                System.out.printf("Linear   | %s | LF: %.2f | Insert: %.2f ms | Find: %.2f ms | Avg Probes: %.2f%n",
                        hf.getName(), lf, insertTime / 1_000_000.0, findTime / 1_000_000.0,
                        linearDict.getStats().getAverageProbesPerOperation());

                System.out.println();
            }
        }
    }

    /**
     * Demonstrate load factor impact
     */
    private static void demonstrateLoadFactorImpact() {
        System.out.println("\n=== 6. Load Factor Impact Analysis ===\n");

        double[] loadFactors = {0.25, 0.5, 0.75, 0.9, 0.95};
        HashFunction<String> hf = new MurmurHash3();

        List<String> keys = new ArrayList<>();
        for (int i = 0; i < 5000; i++) {
            keys.add("test_key_" + i);
        }

        System.out.println("Testing Chaining Hash Table:");
        System.out.printf("%-12s | %-15s | %-15s | %-15s | %-10s%n",
                "Load Factor", "Avg Insert (ns)", "Avg Find (ns)", "Max Chain Len", "Resizes");
        System.out.println("─".repeat(80));

        for (double lf : loadFactors) {
            ChainingHashTable<String, Integer> dict = new ChainingHashTable<>(16, lf, hf);

            for (int i = 0; i < keys.size(); i++) {
                dict.insert(keys.get(i), i);
            }

            for (String key : keys) {
                dict.find(key);
            }

            DictionaryStats stats = dict.getStats();
            System.out.printf("%-12.2f | %-15.2f | %-15.2f | %-15d | %-10d%n",
                    lf, stats.getAverageInsertTime(), stats.getAverageFindTime(),
                    stats.getMaxChainLength(), stats.getTotalResizes());
        }

        System.out.println("\nTesting Open Addressing (Linear Probing):");
        System.out.printf("%-12s | %-15s | %-15s | %-15s | %-10s%n",
                "Load Factor", "Avg Insert (ns)", "Avg Find (ns)", "Max Probe Len", "Resizes");
        System.out.println("─".repeat(80));

        for (double lf : loadFactors) {
            if (lf >= 0.9) continue; // Skip very high load factors for open addressing

            OpenAddressingHashTable<String, Integer> dict =
                    new OpenAddressingHashTable<>(16, lf, hf,
                            OpenAddressingHashTable.ProbingStrategy.LINEAR);

            for (int i = 0; i < keys.size(); i++) {
                dict.insert(keys.get(i), i);
            }

            for (String key : keys) {
                dict.find(key);
            }

            DictionaryStats stats = dict.getStats();
            System.out.printf("%-12.2f | %-15.2f | %-15.2f | %-15d | %-10d%n",
                    lf, stats.getAverageInsertTime(), stats.getAverageFindTime(),
                    stats.getMaxProbeLength(), stats.getTotalResizes());
        }
    }

    /**
     * Additional utility: Test with real-world text
     */
    public static void testWithRealText(String filename) {
        System.out.println("\n=== Testing with Real Text File ===\n");

        try {
            ChainingHashTable<String, Integer> dict = new ChainingHashTable<>(new MurmurHash3());
            WordFrequencyCounter counter = new WordFrequencyCounter(dict);
            counter.processFile(filename);
            System.out.println(counter.getStatistics());
        } catch (Exception e) {
            System.err.println("Error processing file: " + e.getMessage());
        }
    }

    /**
     * Helper method to create sample test scenarios
     */
    public static void runCustomTest(int numKeys, double loadFactor, String hashFunctionName) {
        System.out.println("\n=== Custom Test Configuration ===");
        System.out.println("Number of keys: " + numKeys);
        System.out.println("Load factor: " + loadFactor);
        System.out.println("Hash function: " + hashFunctionName);
        System.out.println();

        HashFunction<String> hf;
        switch (hashFunctionName.toLowerCase()) {
            case "polynomial":
                hf = new PolynomialHash();
                break;
            case "fnv1a":
                hf = new FNV1aHash();
                break;
            case "murmur":
                hf = new MurmurHash3();
                break;
            case "sha256":
                hf = new SHA256Hash();
                break;
            default:
                System.out.println("Unknown hash function, using MurmurHash3");
                hf = new MurmurHash3();
        }

        List<String> keys = new ArrayList<>();
        for (int i = 0; i < numKeys; i++) {
            keys.add("key_" + i);
        }

        // Test Chaining
        System.out.println("Chaining Hash Table:");
        ChainingHashTable<String, Integer> chainingDict =
                new ChainingHashTable<>(16, loadFactor, hf);

        long startTime = System.nanoTime();
        for (int i = 0; i < keys.size(); i++) {
            chainingDict.insert(keys.get(i), i);
        }
        long totalTime = System.nanoTime() - startTime;

        System.out.println("  Time: " + (totalTime / 1_000_000.0) + " ms");
        System.out.println(chainingDict.getStats());

        // Test Open Addressing
        System.out.println("\nOpen Addressing (Linear Probing):");
        OpenAddressingHashTable<String, Integer> openDict =
                new OpenAddressingHashTable<>(16, loadFactor, hf,
                        OpenAddressingHashTable.ProbingStrategy.LINEAR);

        startTime = System.nanoTime();
        for (int i = 0; i < keys.size(); i++) {
            openDict.insert(keys.get(i), i);
        }
        totalTime = System.nanoTime() - startTime;

        System.out.println("  Time: " + (totalTime / 1_000_000.0) + " ms");
        System.out.println(openDict.getStats());
    }
}