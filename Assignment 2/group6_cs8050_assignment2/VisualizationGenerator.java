package group6_cs8050_assignment2;

import java.io.*;
import java.util.*;

/**
 * Generates data files for visualization and analysis
 * Exports CSV and plotting scripts for graphs
 */
public class VisualizationGenerator {

    /**
     * Generate load factor vs performance data
     */
    public static void generateLoadFactorAnalysis(String outputFile) throws IOException {
        System.out.println("Generating load factor analysis data...");

        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            writer.println("LoadFactor,Implementation,HashFunction,AvgInsertTime,AvgFindTime,AvgDeleteTime,AvgProbes,MaxLength");

            double[] loadFactors = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95};
            HashFunction<String>[] hashFunctions = new HashFunction[]{
                    new PolynomialHash(),
                    new SHA256Hash()
            };

            List<String> keys = generateTestKeys(5000);

            for (double lf : loadFactors) {
                for (HashFunction<String> hf : hashFunctions) {
                    // Chaining
                    ChainingHashTable<String, Integer> chainingDict =
                            new ChainingHashTable<>(16, lf, hf);
                    testDictionary(chainingDict, keys);
                    DictionaryStats stats = chainingDict.getStats();

                    writer.printf("%.2f,Chaining,%s,%.2f,%.2f,%.2f,%.2f,%d%n",
                            lf, hf.getName(),
                            stats.getAverageInsertTime(),
                            stats.getAverageFindTime(),
                            stats.getAverageDeleteTime(),
                            stats.getAverageProbesPerOperation(),
                            stats.getMaxChainLength());

                    // Linear Probing (skip high load factors)
                    if (lf < 0.9) {
                        OpenAddressingHashTable<String, Integer> linearDict =
                                new OpenAddressingHashTable<>(16, lf, hf,
                                        OpenAddressingHashTable.ProbingStrategy.LINEAR);
                        testDictionary(linearDict, keys);
                        stats = linearDict.getStats();

                        writer.printf("%.2f,Linear Probing,%s,%.2f,%.2f,%.2f,%.2f,%d%n",
                                lf, hf.getName(),
                                stats.getAverageInsertTime(),
                                stats.getAverageFindTime(),
                                stats.getAverageDeleteTime(),
                                stats.getAverageProbesPerOperation(),
                                stats.getMaxProbeLength());
                    }

                    // Quadratic Probing
                    if (lf < 0.9) {
                        OpenAddressingHashTable<String, Integer> quadraticDict =
                                new OpenAddressingHashTable<>(16, lf, hf,
                                        OpenAddressingHashTable.ProbingStrategy.QUADRATIC);
                        testDictionary(quadraticDict, keys);
                        stats = quadraticDict.getStats();

                        writer.printf("%.2f,Quadratic Probing,%s,%.2f,%.2f,%.2f,%.2f,%d%n",
                                lf, hf.getName(),
                                stats.getAverageInsertTime(),
                                stats.getAverageFindTime(),
                                stats.getAverageDeleteTime(),
                                stats.getAverageProbesPerOperation(),
                                stats.getMaxProbeLength());
                    }
                }
            }
        }

        System.out.println("Load factor analysis data saved to " + outputFile);
    }

    /**
     * Generate collision analysis data
     */
    public static void generateCollisionAnalysis(String outputFile) throws IOException {
        System.out.println("Generating collision analysis data...");

        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            writer.println("DataSize,HashFunction,CollisionRate,AvgChainLength,MaxChainLength");

            int[] dataSizes = {100, 500, 1000, 5000, 10000, 50000};
            HashFunction<String>[] hashFunctions = new HashFunction[]{
                    new PolynomialHash(),
                    new SHA256Hash()
            };

            for (int dataSize : dataSizes) {
                List<String> keys = generateTestKeys(dataSize);

                for (HashFunction<String> hf : hashFunctions) {
                    ChainingHashTable<String, Integer> dict =
                            new ChainingHashTable<>(16, 0.75, hf);

                    for (int i = 0; i < keys.size(); i++) {
                        dict.insert(keys.get(i), i);
                    }

                    DictionaryStats stats = dict.getStats();
                    double avgChainLength = stats.getAverageProbesPerOperation();

                    writer.printf("%d,%s,%.4f,%.2f,%d%n",
                            dataSize, hf.getName(),
                            stats.getCollisionRate(),
                            avgChainLength,
                            stats.getMaxChainLength());
                }
            }
        }

        System.out.println("Collision analysis data saved to " + outputFile);
    }

    /**
     * Generate scalability analysis data
     */
    public static void generateScalabilityAnalysis(String outputFile) throws IOException {
        System.out.println("Generating scalability analysis data...");

        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            writer.println("DataSize,Implementation,TotalInsertTime,TotalFindTime,AvgInsertTime,AvgFindTime");

            int[] dataSizes = {1000, 5000, 10000, 25000, 50000, 100000};
            HashFunction<String> hf = new PolynomialHash();

            for (int dataSize : dataSizes) {
                List<String> keys = generateTestKeys(dataSize);

                // Chaining
                ChainingHashTable<String, Integer> chainingDict =
                        new ChainingHashTable<>(hf);

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

                writer.printf("%d,Chaining,%.2f,%.2f,%.2f,%.2f%n",
                        dataSize,
                        insertTime / 1_000_000.0,
                        findTime / 1_000_000.0,
                        insertTime / (double) dataSize,
                        findTime / (double) dataSize);

                // Linear Probing
                OpenAddressingHashTable<String, Integer> linearDict =
                        new OpenAddressingHashTable<>(hf,
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

                writer.printf("%d,Linear Probing,%.2f,%.2f,%.2f,%.2f%n",
                        dataSize,
                        insertTime / 1_000_000.0,
                        findTime / 1_000_000.0,
                        insertTime / (double) dataSize,
                        findTime / (double) dataSize);

                // Quadratic Probing
                OpenAddressingHashTable<String, Integer> quadraticDict =
                        new OpenAddressingHashTable<>(hf,
                                OpenAddressingHashTable.ProbingStrategy.QUADRATIC);

                startTime = System.nanoTime();
                for (int i = 0; i < keys.size(); i++) {
                    quadraticDict.insert(keys.get(i), i);
                }
                insertTime = System.nanoTime() - startTime;

                startTime = System.nanoTime();
                for (String key : keys) {
                    quadraticDict.find(key);
                }
                findTime = System.nanoTime() - startTime;

                writer.printf("%d,Quadratic Probing,%.2f,%.2f,%.2f,%.2f%n",
                        dataSize,
                        insertTime / 1_000_000.0,
                        findTime / 1_000_000.0,
                        insertTime / (double) dataSize,
                        findTime / (double) dataSize);
            }
        }

        System.out.println("Scalability analysis data saved to " + outputFile);
    }

    /**
     * Generate distribution comparison data
     */
    public static void generateDistributionAnalysis(String outputFile) throws IOException {
        System.out.println("Generating distribution analysis data...");

        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            writer.println("Distribution,Implementation,CollisionRate,AvgProbes,MaxLength,InsertTime");

            int dataSize = 10000;
            HashFunction<String> hf = new PolynomialHash();

            String[] distributions = {"Uniform", "PowerLaw", "Adversarial"};

            for (String distribution : distributions) {
                List<String> keys;
                switch (distribution) {
                    case "Uniform":
                        keys = generateUniformKeys(dataSize);
                        break;
                    case "PowerLaw":
                        keys = generatePowerLawKeys(dataSize);
                        break;
                    case "Adversarial":
                        keys = generateAdversarialKeys(dataSize);
                        break;
                    default:
                        keys = generateUniformKeys(dataSize);
                }

                // Chaining
                ChainingHashTable<String, Integer> chainingDict =
                        new ChainingHashTable<>(hf);

                long startTime = System.nanoTime();
                for (int i = 0; i < keys.size(); i++) {
                    chainingDict.insert(keys.get(i), i);
                }
                long insertTime = System.nanoTime() - startTime;

                DictionaryStats stats = chainingDict.getStats();
                writer.printf("%s,Chaining,%.4f,%.2f,%d,%.2f%n",
                        distribution,
                        stats.getCollisionRate(),
                        stats.getAverageProbesPerOperation(),
                        stats.getMaxChainLength(),
                        insertTime / 1_000_000.0);

                // Linear Probing
                OpenAddressingHashTable<String, Integer> linearDict =
                        new OpenAddressingHashTable<>(hf,
                                OpenAddressingHashTable.ProbingStrategy.LINEAR);

                startTime = System.nanoTime();
                for (int i = 0; i < keys.size(); i++) {
                    linearDict.insert(keys.get(i), i);
                }
                insertTime = System.nanoTime() - startTime;

                stats = linearDict.getStats();
                writer.printf("%s,Linear Probing,%.4f,%.2f,%d,%.2f%n",
                        distribution,
                        stats.getCollisionRate(),
                        stats.getAverageProbesPerOperation(),
                        stats.getMaxProbeLength(),
                        insertTime / 1_000_000.0);
            }
        }

        System.out.println("Distribution analysis data saved to " + outputFile);
    }

    /**
     * Generate Python plotting script
     */
    public static void generatePlotScript(String scriptFile) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(scriptFile))) {
            writer.println("import pandas as pd");
            writer.println("import matplotlib.pyplot as plt");
            writer.println("import seaborn as sns");
            writer.println();
            writer.println("sns.set_style('whitegrid')");
            writer.println();
            writer.println("# Load Factor Analysis");
            writer.println("df_lf = pd.read_csv('load_factor_analysis.csv')");
            writer.println();
            writer.println("fig, axes = plt.subplots(2, 2, figsize=(14, 10))");
            writer.println();
            writer.println("# Insert Time vs Load Factor");
            writer.println("for impl in df_lf['Implementation'].unique():");
            writer.println("    data = df_lf[df_lf['Implementation'] == impl]");
            writer.println("    axes[0, 0].plot(data['LoadFactor'], data['AvgInsertTime'], marker='o', label=impl)");
            writer.println("axes[0, 0].set_xlabel('Load Factor')");
            writer.println("axes[0, 0].set_ylabel('Avg Insert Time (ns)')");
            writer.println("axes[0, 0].set_title('Insert Time vs Load Factor')");
            writer.println("axes[0, 0].legend()");
            writer.println("axes[0, 0].grid(True)");
            writer.println();
            writer.println("# Find Time vs Load Factor");
            writer.println("for impl in df_lf['Implementation'].unique():");
            writer.println("    data = df_lf[df_lf['Implementation'] == impl]");
            writer.println("    axes[0, 1].plot(data['LoadFactor'], data['AvgFindTime'], marker='o', label=impl)");
            writer.println("axes[0, 1].set_xlabel('Load Factor')");
            writer.println("axes[0, 1].set_ylabel('Avg Find Time (ns)')");
            writer.println("axes[0, 1].set_title('Find Time vs Load Factor')");
            writer.println("axes[0, 1].legend()");
            writer.println("axes[0, 1].grid(True)");
            writer.println();
            writer.println("# Average Probes vs Load Factor");
            writer.println("for impl in df_lf['Implementation'].unique():");
            writer.println("    data = df_lf[df_lf['Implementation'] == impl]");
            writer.println("    axes[1, 0].plot(data['LoadFactor'], data['AvgProbes'], marker='o', label=impl)");
            writer.println("axes[1, 0].set_xlabel('Load Factor')");
            writer.println("axes[1, 0].set_ylabel('Average Probes')");
            writer.println("axes[1, 0].set_title('Average Probes vs Load Factor')");
            writer.println("axes[1, 0].legend()");
            writer.println("axes[1, 0].grid(True)");
            writer.println();
            writer.println("# Max Length vs Load Factor");
            writer.println("for impl in df_lf['Implementation'].unique():");
            writer.println("    data = df_lf[df_lf['Implementation'] == impl]");
            writer.println("    axes[1, 1].plot(data['LoadFactor'], data['MaxLength'], marker='o', label=impl)");
            writer.println("axes[1, 1].set_xlabel('Load Factor')");
            writer.println("axes[1, 1].set_ylabel('Max Chain/Probe Length')");
            writer.println("axes[1, 1].set_title('Max Length vs Load Factor')");
            writer.println("axes[1, 1].legend()");
            writer.println("axes[1, 1].grid(True)");
            writer.println();
            writer.println("plt.tight_layout()");
            writer.println("plt.savefig('load_factor_analysis.png', dpi=300)");
            writer.println("plt.show()");
            writer.println();
            writer.println("# Scalability Analysis");
            writer.println("df_scale = pd.read_csv('scalability_analysis.csv')");
            writer.println();
            writer.println("fig, axes = plt.subplots(1, 2, figsize=(14, 5))");
            writer.println();
            writer.println("for impl in df_scale['Implementation'].unique():");
            writer.println("    data = df_scale[df_scale['Implementation'] == impl]");
            writer.println("    axes[0].plot(data['DataSize'], data['TotalInsertTime'], marker='o', label=impl)");
            writer.println("axes[0].set_xlabel('Data Size')");
            writer.println("axes[0].set_ylabel('Total Insert Time (ms)')");
            writer.println("axes[0].set_title('Scalability: Total Insert Time')");
            writer.println("axes[0].legend()");
            writer.println("axes[0].grid(True)");
            writer.println();
            writer.println("for impl in df_scale['Implementation'].unique():");
            writer.println("    data = df_scale[df_scale['Implementation'] == impl]");
            writer.println("    axes[1].plot(data['DataSize'], data['AvgInsertTime'], marker='o', label=impl)");
            writer.println("axes[1].set_xlabel('Data Size')");
            writer.println("axes[1].set_ylabel('Avg Insert Time (ns)')");
            writer.println("axes[1].set_title('Scalability: Average Insert Time')");
            writer.println("axes[1].legend()");
            writer.println("axes[1].grid(True)");
            writer.println();
            writer.println("plt.tight_layout()");
            writer.println("plt.savefig('scalability_analysis.png', dpi=300)");
            writer.println("plt.show()");
        }

        System.out.println("Python plotting script saved to " + scriptFile);
    }

    /**
     * Helper method to test a dictionary
     */
    private static void testDictionary(Dictionary<String, Integer> dict, List<String> keys) {
        // Insert all keys
        for (int i = 0; i < keys.size(); i++) {
            dict.insert(keys.get(i), i);
        }

        // Find all keys
        for (String key : keys) {
            dict.find(key);
        }

        // Delete half the keys
        for (int i = 0; i < keys.size() / 2; i++) {
            dict.delete(keys.get(i));
        }
    }

    /**
     * Generate test keys with uniform distribution
     */
    private static List<String> generateTestKeys(int count) {
        List<String> keys = new ArrayList<>();
        Random rand = new Random(42);

        for (int i = 0; i < count; i++) {
            keys.add("key_" + rand.nextInt(count * 10));
        }

        return keys;
    }

    /**
     * Generate uniform random keys
     */
    private static List<String> generateUniformKeys(int count) {
        List<String> keys = new ArrayList<>();
        Random rand = new Random(42);

        for (int i = 0; i < count; i++) {
            keys.add("uniform_" + rand.nextInt(Integer.MAX_VALUE));
        }

        return keys;
    }

    /**
     * Generate power-law distributed keys
     */
    private static List<String> generatePowerLawKeys(int count) {
        List<String> keys = new ArrayList<>();
        Random rand = new Random(42);

        for (int i = 0; i < count; i++) {
            double u = rand.nextDouble();
            int rank = (int) Math.pow(u, -0.5) % (count / 10);
            keys.add("popular_" + rank);
        }

        return keys;
    }

    /**
     * Generate adversarial keys
     */
    private static List<String> generateAdversarialKeys(int count) {
        List<String> keys = new ArrayList<>();

        for (int i = 0; i < count; i++) {
            keys.add("collision_" + (i % 100) + "_" + i);
        }

        return keys;
    }

    /**
     * Generate all visualization data files
     */
    public static void generateAll() {
        try {
            System.out.println("\n=== Generating All Visualization Data ===\n");

            generateLoadFactorAnalysis("load_factor_analysis.csv");
            generateCollisionAnalysis("collision_analysis.csv");
            generateScalabilityAnalysis("scalability_analysis.csv");
            generateDistributionAnalysis("distribution_analysis.csv");
            generatePlotScript("plot_results.py");

            System.out.println("\nAll visualization data generated successfully!");
            System.out.println("Run 'python plot_results.py' to generate graphs.");
        } catch (IOException e) {
            System.err.println("Error generating visualization data: " + e.getMessage());
        }
    }

    /**
     * Main method to run visualization generation standalone
     */
    public static void main(String[] args) {
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║   Visualization Data Generator                             ║");
        System.out.println("║   CS 8050 - Assignment 2                                   ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝\n");

        if (args.length > 0) {
            // Generate specific analysis based on command line argument
            try {
                switch (args[0].toLowerCase()) {
                    case "loadfactor":
                    case "lf":
                        System.out.println("Generating load factor analysis only...");
                        generateLoadFactorAnalysis("load_factor_analysis.csv");
                        break;
                    case "collision":
                    case "col":
                        System.out.println("Generating collision analysis only...");
                        generateCollisionAnalysis("collision_analysis.csv");
                        break;
                    case "scalability":
                    case "scale":
                        System.out.println("Generating scalability analysis only...");
                        generateScalabilityAnalysis("scalability_analysis.csv");
                        break;
                    case "distribution":
                    case "dist":
                        System.out.println("Generating distribution analysis only...");
                        generateDistributionAnalysis("distribution_analysis.csv");
                        break;
                    case "plot":
                        System.out.println("Generating Python plotting script only...");
                        generatePlotScript("plot_results.py");
                        break;
                    case "all":
                        generateAll();
                        break;
                    default:
                        System.out.println("Unknown option: " + args[0]);
                        printUsage();
                }
            } catch (IOException e) {
                System.err.println("Error: " + e.getMessage());
                e.printStackTrace();
            }
        } else {
            // No arguments - generate everything
            generateAll();
        }

        System.out.println("\n✓ Done!");
    }

    private static void printUsage() {
        System.out.println("\nUsage: java group1_cs8050_assignment2.VisualizationGenerator [option]");
        System.out.println("\nOptions:");
        System.out.println("  all          - Generate all visualization data (default)");
        System.out.println("  loadfactor   - Generate load factor analysis only");
        System.out.println("  collision    - Generate collision analysis only");
        System.out.println("  scalability  - Generate scalability analysis only");
        System.out.println("  distribution - Generate distribution analysis only");
        System.out.println("  plot         - Generate Python plotting script only");
        System.out.println("\nExamples:");
        System.out.println("  java group1_cs8050_assignment2.VisualizationGenerator");
        System.out.println("  java group1_cs8050_assignment2.VisualizationGenerator all");
        System.out.println("  java group1_cs8050_assignment2.VisualizationGenerator loadfactor");
    }
}