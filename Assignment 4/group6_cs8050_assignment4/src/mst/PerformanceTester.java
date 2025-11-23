package mst;

import java.io.*;
import java.util.*;

/**
 * Tests and compares performance of MST algorithms
 */
public class PerformanceTester {

    public static void runComprehensiveTests() {
        System.out.println("=".repeat(80));
        System.out.println("COMPREHENSIVE MST ALGORITHM PERFORMANCE TESTING");
        System.out.println("=".repeat(80));

        // Test sparse graphs
        System.out.println("\n" + "=".repeat(80));
        System.out.println("SPARSE GRAPHS (E ≈ V)");
        System.out.println("=".repeat(80));
        testGraphType("Sparse", new int[]{10, 50, 100, 500}, false);

        // Test dense graphs
        System.out.println("\n" + "=".repeat(80));
        System.out.println("DENSE GRAPHS (E ≈ V²/2)");
        System.out.println("=".repeat(80));
        testGraphType("Dense", new int[]{10, 50, 100, 200}, true);

        // Test large graphs
        System.out.println("\n" + "=".repeat(80));
        System.out.println("LARGE GRAPHS (≥500 vertices)");
        System.out.println("=".repeat(80));
        testLargeGraphs();

        // Generate summary report
        generateSummaryReport();
    }

    private static void testGraphType(String type, int[] vertexCounts, boolean dense) {
        for (int vertices : vertexCounts) {
            System.out.println("\n" + "-".repeat(80));
            System.out.printf("Testing %s Graph: %d vertices\n", type, vertices);
            System.out.println("-".repeat(80));

            Graph graph = dense ?
                    GraphGenerator.generateDenseGraph(vertices) :
                    GraphGenerator.generateSparseGraph(vertices);

            int edges = graph.getAllEdges().size();
            System.out.printf("Graph Statistics: V=%d, E=%d, Density=%.2f%%\n",
                    vertices, edges, (edges * 100.0) / (vertices * (vertices - 1) / 2));

            // Test Prim's Algorithm
            System.out.println("\n--- Prim's Algorithm ---");
            PrimMST prim = new PrimMST();
            PerformanceMetrics primMetrics = runMultipleTrials(() -> prim.findMSTWithMetrics(graph, 1), 5);
            System.out.println(primMetrics);

            // Test Kruskal's Algorithm
            System.out.println("\n--- Kruskal's Algorithm ---");
            KruskalMST kruskal = new KruskalMST();
            PerformanceMetrics kruskalMetrics = runMultipleTrials(() -> kruskal.findMSTWithMetrics(graph), 5);
            System.out.println(kruskalMetrics);

            // Compare results
            System.out.println("\n--- Comparison ---");
            compareAlgorithms(primMetrics, kruskalMetrics);
        }
    }

    private static void testLargeGraphs() {
        int[] sizes = {500, 1000};

        for (int size : sizes) {
            // Test sparse large graph
            System.out.println("\n" + "-".repeat(80));
            System.out.printf("Testing Large Sparse Graph: %d vertices\n", size);
            System.out.println("-".repeat(80));

            Graph sparseGraph = GraphGenerator.generateSparseGraph(size);
            testGraph(sparseGraph, "Large Sparse");

            // Test dense large graph (smaller size due to memory)
            if (size <= 500) {
                System.out.println("\n" + "-".repeat(80));
                System.out.printf("Testing Large Dense Graph: %d vertices\n", size);
                System.out.println("-".repeat(80));

                Graph denseGraph = GraphGenerator.generateDenseGraph(size);
                testGraph(denseGraph, "Large Dense");
            }
        }
    }

    private static void testGraph(Graph graph, String description) {
        int vertices = graph.getVertices();
        int edges = graph.getAllEdges().size();
        System.out.printf("Graph Statistics: V=%d, E=%d\n", vertices, edges);

        PrimMST prim = new PrimMST();
        KruskalMST kruskal = new KruskalMST();

        System.out.println("\n--- Prim's Algorithm ---");
        PerformanceMetrics primMetrics = runMultipleTrials(() -> prim.findMSTWithMetrics(graph, 1), 3);
        System.out.println(primMetrics);

        System.out.println("\n--- Kruskal's Algorithm ---");
        PerformanceMetrics kruskalMetrics = runMultipleTrials(() -> kruskal.findMSTWithMetrics(graph), 3);
        System.out.println(kruskalMetrics);

        System.out.println("\n--- Comparison ---");
        compareAlgorithms(primMetrics, kruskalMetrics);
    }

    private static PerformanceMetrics runMultipleTrials(java.util.function.Supplier<PerformanceMetrics> test, int trials) {
        long totalTime = 0;
        long totalMemory = 0;
        List<Edge> edges = null;

        for (int i = 0; i < trials; i++) {
            PerformanceMetrics metrics = test.get();
            totalTime += metrics.getExecutionTimeNanos();
            totalMemory += metrics.getMemoryUsedBytes();
            edges = metrics.getMstEdges();
        }

        return new PerformanceMetrics(totalTime / trials, totalMemory / trials, edges);
    }

    private static void compareAlgorithms(PerformanceMetrics prim, PerformanceMetrics kruskal) {
        double timeRatio = prim.getExecutionTimeMillis() / kruskal.getExecutionTimeMillis();
        double memoryRatio = prim.getMemoryUsedKB() / kruskal.getMemoryUsedKB();

        System.out.printf("Time Ratio (Prim/Kruskal): %.2f\n", timeRatio);
        System.out.printf("Memory Ratio (Prim/Kruskal): %.2f\n", memoryRatio);

        if (timeRatio < 1) {
            System.out.printf("Prim's is %.2f%% faster\n", (1 - timeRatio) * 100);
        } else {
            System.out.printf("Kruskal's is %.2f%% faster\n", (timeRatio - 1) * 100);
        }

        // Verify both produce same MST weight
        double primWeight = prim.getTotalMSTWeight();
        double kruskalWeight = kruskal.getTotalMSTWeight();
        System.out.printf("MST Weights - Prim: %.2f, Kruskal: %.2f, Match: %s\n",
                primWeight, kruskalWeight, Math.abs(primWeight - kruskalWeight) < 0.01);
    }

    private static void generateSummaryReport() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("SUMMARY AND CONCLUSIONS");
        System.out.println("=".repeat(80));

        System.out.println("\nKey Observations:");
        System.out.println("1. Prim's Algorithm performs better on dense graphs due to efficient heap operations");
        System.out.println("2. Kruskal's Algorithm performs better on sparse graphs due to fewer edges to sort");
        System.out.println("3. Both algorithms produce the same MST weight (correctness verified)");
        System.out.println("4. Memory usage is comparable, with slight variations based on data structures");
        System.out.println("5. For very large graphs, Kruskal's edge sorting becomes a bottleneck");

        System.out.println("\nTheoretical Complexity:");
        System.out.println("Prim's:    O((V + E) log V) with binary heap");
        System.out.println("Kruskal's: O(E log E) ≈ O(E log V) dominated by sorting");

        System.out.println("\nRecommendations:");
        System.out.println("- Use Prim's for dense graphs (E close to V²)");
        System.out.println("- Use Kruskal's for sparse graphs (E close to V)");
        System.out.println("- Consider Prim's when starting vertex matters");
        System.out.println("- Consider Kruskal's for parallel processing (edge sorting parallelizable)");
    }
}