package group6_cs8050_assignment2;

import java.util.*;
import java.io.*;

/**
 * Comprehensive Benchmark Suite for Dictionary Implementations
 * Tests performance under various conditions and generates data for analysis
 */
public class BenchmarkSuite {

    public static class BenchmarkResult {
        String testName;
        String implementation;
        String hashFunction;
        double loadFactor;
        long avgInsertTime;
        long avgFindTime;
        long avgDeleteTime;
        double avgProbes;
        double collisionRate;
        int maxChainOrProbeLength;
        int totalResizes;

        @Override
        public String toString() {
            return String.format(
                    "%-40s | %-25s | %-20s | LF: %.2f | Insert: %6d ns | Find: %6d ns | " +
                            "Delete: %6d ns | Probes: %.2f | Collisions: %.4f | Max: %d | Resizes: %d",
                    testName, implementation, hashFunction, loadFactor,
                    avgInsertTime, avgFindTime, avgDeleteTime, avgProbes,
                    collisionRate, maxChainOrProbeLength, totalResizes
            );
        }

        public String toCSV() {
            return String.format("%s,%s,%s,%.2f,%d,%d,%d,%.2f,%.4f,%d,%d",
                    testName, implementation, hashFunction, loadFactor,
                    avgInsertTime, avgFindTime, avgDeleteTime, avgProbes,
                    collisionRate, maxChainOrProbeLength, totalResizes
            );
        }
    }

    /**
     * Run comprehensive benchmark suite
     */
    public static List<BenchmarkResult> runFullSuite() {
        List<BenchmarkResult> results = new ArrayList<>();

        System.out.println("=== Starting Comprehensive Benchmark Suite ===\n");

        // Test different load factors
        double[] loadFactors = {0.25, 0.5, 0.75, 0.9, 0.95};

        // Test different data sizes
        int[] dataSizes = {1000, 10000, 100000};

        // Test different hash functions
        HashFunction<String>[] hashFunctions = new HashFunction[]{
                new PolynomialHash(),
                new SHA256Hash()
        };

        // Test different key distributions
        for (int dataSize : dataSizes) {
            System.out.println("Testing with data size: " + dataSize);

            // Uniform random distribution
            results.addAll(testUniformDistribution(dataSize, loadFactors, hashFunctions));

            // Skewed distribution (power-law)
            results.addAll(testPowerLawDistribution(dataSize, loadFactors, hashFunctions));

            // Adversarial (many collisions)
            results.addAll(testAdversarialDistribution(dataSize, loadFactors, hashFunctions));
        }

        System.out.println("\n Benchmark Suite Complete ");
        return results;
    }

    /**
     * Test with uniform random key distribution
     */
    private static List<BenchmarkResult> testUniformDistribution(
            int size, double[] loadFactors, HashFunction<String>[] hashFunctions) {

        List<BenchmarkResult> results = new ArrayList<>();
        List<String> keys = generateUniformRandomKeys(size);

        for (double lf : loadFactors) {
            for (HashFunction<String> hf : hashFunctions) {
                // Test Chaining
                results.add(testChainingHashTable(keys, lf, hf, "Uniform-Random"));

                // Test Linear Probing
                results.add(testOpenAddressing(keys, lf, hf,
                        OpenAddressingHashTable.ProbingStrategy.LINEAR, "Uniform-Random"));

                // Test Quadratic Probing
                results.add(testOpenAddressing(keys, lf, hf,
                        OpenAddressingHashTable.ProbingStrategy.QUADRATIC, "Uniform-Random"));
            }
        }

        return results;
    }

    /**
     * Test with power-law (skewed) distribution
     */
    private static List<BenchmarkResult> testPowerLawDistribution(
            int size, double[] loadFactors, HashFunction<String>[] hashFunctions) {

        List<BenchmarkResult> results = new ArrayList<>();
        List<String> keys = generatePowerLawKeys(size);

        for (double lf : loadFactors) {
            for (HashFunction<String> hf : hashFunctions) {
                results.add(testChainingHashTable(keys, lf, hf, "Power-Law"));
                results.add(testOpenAddressing(keys, lf, hf,
                        OpenAddressingHashTable.ProbingStrategy.LINEAR, "Power-Law"));
                results.add(testOpenAddressing(keys, lf, hf,
                        OpenAddressingHashTable.ProbingStrategy.QUADRATIC, "Power-Law"));
            }
        }

        return results;
    }

    /**
     * Test with adversarial distribution (designed to cause collisions)
     */
    private static List<BenchmarkResult> testAdversarialDistribution(
            int size, double[] loadFactors, HashFunction<String>[] hashFunctions) {

        List<BenchmarkResult> results = new ArrayList<>();
        List<String> keys = generateAdversarialKeys(size);

        for (double lf : loadFactors) {
            for (HashFunction<String> hf : hashFunctions) {
                results.add(testChainingHashTable(keys, lf, hf, "Adversarial"));
                results.add(testOpenAddressing(keys, lf, hf,
                        OpenAddressingHashTable.ProbingStrategy.LINEAR, "Adversarial"));
                results.add(testOpenAddressing(keys, lf, hf,
                        OpenAddressingHashTable.ProbingStrategy.QUADRATIC, "Adversarial"));
            }
        }

        return results;
    }

    /**
     * Test Chaining Hash Table
     */
    private static BenchmarkResult testChainingHashTable(
            List<String> keys, double loadFactor, HashFunction<String> hashFunc, String distribution) {

        ChainingHashTable<String, Integer> dict =
                new ChainingHashTable<>(16, loadFactor, hashFunc);

        // Insert phase
        for (int i = 0; i < keys.size(); i++) {
            dict.insert(keys.get(i), i);
        }

        // Find phase (search for all keys)
        for (String key : keys) {
            dict.find(key);
        }

        // Delete phase (delete half the keys)
        for (int i = 0; i < keys.size() / 2; i++) {
            dict.delete(keys.get(i));
        }

        DictionaryStats stats = dict.getStats();

        BenchmarkResult result = new BenchmarkResult();
        result.testName = distribution + " (n=" + keys.size() + ")";
        result.implementation = "Chaining";
        result.hashFunction = hashFunc.getName();
        result.loadFactor = loadFactor;
        result.avgInsertTime = (long) stats.getAverageInsertTime();
        result.avgFindTime = (long) stats.getAverageFindTime();
        result.avgDeleteTime = (long) stats.getAverageDeleteTime();
        result.avgProbes = stats.getAverageProbesPerOperation();
        result.collisionRate = stats.getCollisionRate();
        result.maxChainOrProbeLength = stats.getMaxChainLength();
        result.totalResizes = (int) stats.getTotalResizes();

        return result;
    }

    /**
     * Test Open Addressing Hash Table
     */
    private static BenchmarkResult testOpenAddressing(
            List<String> keys, double loadFactor, HashFunction<String> hashFunc,
            OpenAddressingHashTable.ProbingStrategy strategy, String distribution) {

        OpenAddressingHashTable<String, Integer> dict =
                new OpenAddressingHashTable<>(16, loadFactor, hashFunc, strategy);

        // Insert phase
        for (int i = 0; i < keys.size(); i++) {
            dict.insert(keys.get(i), i);
        }

        // Find phase
        for (String key : keys) {
            dict.find(key);
        }

        // Delete phase
        for (int i = 0; i < keys.size() / 2; i++) {
            dict.delete(keys.get(i));
        }

        DictionaryStats stats = dict.getStats();

        BenchmarkResult result = new BenchmarkResult();
        result.testName = distribution + " (n=" + keys.size() + ")";
        result.implementation = strategy.toString() + " Probing";
        result.hashFunction = hashFunc.getName();
        result.loadFactor = loadFactor;
        result.avgInsertTime = (long) stats.getAverageInsertTime();
        result.avgFindTime = (long) stats.getAverageFindTime();
        result.avgDeleteTime = (long) stats.getAverageDeleteTime();
        result.avgProbes = stats.getAverageProbesPerOperation();
        result.collisionRate = stats.getCollisionRate();
        result.maxChainOrProbeLength = stats.getMaxProbeLength();
        result.totalResizes = (int) stats.getTotalResizes();

        return result;
    }

    /**
     * Generate uniform random keys
     */
    private static List<String> generateUniformRandomKeys(int count) {
        List<String> keys = new ArrayList<>();
        Random rand = new Random(42); // Fixed seed for reproducibility

        for (int i = 0; i < count; i++) {
            keys.add("key_" + rand.nextInt(count * 10));
        }

        return keys;
    }

    /**
     * Generate power-law distributed keys
     */
    private static List<String> generatePowerLawKeys(int count) {
        List<String> keys = new ArrayList<>();
        Random rand = new Random(42);

        // Zipf-like distribution
        for (int i = 0; i < count; i++) {
            double u = rand.nextDouble();
            int rank = (int) Math.pow(u, -0.5) % (count / 10);
            keys.add("popular_" + rank);
        }

        return keys;
    }

    /**
     * Generate adversarial keys (designed to cause many collisions)
     */
    private static List<String> generateAdversarialKeys(int count) {
        List<String> keys = new ArrayList<>();

        // Create keys that are likely to collide with simple hash functions
        for (int i = 0; i < count; i++) {
            // Keys with similar patterns
            keys.add("collision_" + (i % 100) + "_" + i);
        }

        return keys;
    }

    /**
     * Export results to CSV file
     */
    public static void exportToCSV(List<BenchmarkResult> results, String filename) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("TestName,Implementation,HashFunction,LoadFactor," +
                    "AvgInsertTime,AvgFindTime,AvgDeleteTime,AvgProbes," +
                    "CollisionRate,MaxLength,TotalResizes");

            for (BenchmarkResult result : results) {
                writer.println(result.toCSV());
            }

            System.out.println("Results exported to " + filename);
        } catch (IOException e) {
            System.err.println("Error exporting results: " + e.getMessage());
        }
    }

    /**
     * Print summary statistics
     */
    public static void printSummary(List<BenchmarkResult> results) {
        System.out.println("\n Benchmark Results Summary \n");

        for (BenchmarkResult result : results) {
            System.out.println(result);
        }
    }
}