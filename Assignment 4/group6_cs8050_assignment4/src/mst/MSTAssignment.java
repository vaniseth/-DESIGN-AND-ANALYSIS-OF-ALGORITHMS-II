package mst;

import java.io.IOException;
import java.util.List;
import java.util.Scanner;

/**
 * Main class for MST Assignment
 * CS 8050 - Design and Analysis of Algorithms II
 * Assignment 4 - Minimum Spanning Tree Algorithms
 */
public class MSTAssignment {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.println("=".repeat(80));
        System.out.println("CS 8050 - ASSIGNMENT 4: MINIMUM SPANNING TREE ALGORITHMS");
        System.out.println("=".repeat(80));
        System.out.println("\nSelect an option:");
        System.out.println("1. Run with input file (graph.txt)");
        System.out.println("2. Generate and test sample graphs");
        System.out.println("3. Run comprehensive performance tests");
        System.out.println("4. Create custom graph");
        System.out.print("\nEnter choice (1-4): ");

        int choice = scanner.nextInt();
        scanner.nextLine(); // Consume newline

        switch (choice) {
            case 1:
                runWithInputFile();
                break;
            case 2:
                runSampleTests();
                break;
            case 3:
                runPerformanceTests();
                break;
            case 4:
                createCustomGraph(scanner);
                break;
            default:
                System.out.println("Invalid choice. Running with input file...");
                runWithInputFile();
        }

        scanner.close();
    }

    private static void runWithInputFile() {
        try {
            System.out.println("\n" + "=".repeat(80));
            System.out.println("READING GRAPH FROM FILE: graph.txt");
            System.out.println("=".repeat(80));

            Graph graph = GraphReader.readGraphFromFile("graph.txt");

            // Print input graph
            System.out.println();
            graph.printGraph();

            // Run Prim's Algorithm
            System.out.println("\n" + "=".repeat(80));
            System.out.println("PRIM'S ALGORITHM");
            System.out.println("=".repeat(80));
            PrimMST prim = new PrimMST();
            List<Edge> primMST = prim.findMST(graph, 1);
            graph.printMST(primMST, "Prim's Algorithm");

            // Run Kruskal's Algorithm
            System.out.println("\n" + "=".repeat(80));
            System.out.println("KRUSKAL'S ALGORITHM");
            System.out.println("=".repeat(80));
            KruskalMST kruskal = new KruskalMST();
            List<Edge> kruskalMST = kruskal.findMST(graph);
            graph.printMST(kruskalMST, "Kruskal's Algorithm");

            // Compare results
            System.out.println("\n" + "=".repeat(80));
            System.out.println("ALGORITHM COMPARISON");
            System.out.println("=".repeat(80));
            compareEdgeOrder(primMST, kruskalMST);

        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
            System.err.println("\nMake sure 'graph.txt' exists in the project directory.");
            System.err.println("Creating a sample graph.txt file...");
            createSampleGraphFile();
        }
    }

    private static void runSampleTests() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("GENERATING AND TESTING SAMPLE GRAPHS");
        System.out.println("=".repeat(80));

        // Test small sparse graph
        System.out.println("\n--- Test 1: Small Sparse Graph (10 vertices) ---");
        Graph sparseGraph = GraphGenerator.generateSparseGraph(10);
        testGraph(sparseGraph);

        // Test small dense graph
        System.out.println("\n--- Test 2: Small Dense Graph (10 vertices) ---");
        Graph denseGraph = GraphGenerator.generateDenseGraph(10);
        testGraph(denseGraph);

        // Test medium graph
        System.out.println("\n--- Test 3: Medium Graph (50 vertices) ---");
        Graph mediumGraph = GraphGenerator.generateSparseGraph(50);
        testGraph(mediumGraph);
    }

    private static void runPerformanceTests() {
        PerformanceTester.runComprehensiveTests();
    }

    private static void createCustomGraph(Scanner scanner) {
        System.out.print("\nEnter number of vertices: ");
        int vertices = scanner.nextInt();

        Graph graph = new Graph(vertices);

        System.out.println("Enter edges in format: source destination weight");
        System.out.println("Enter 0 0 0 to finish");

        while (true) {
            int source = scanner.nextInt();
            int dest = scanner.nextInt();
            double weight = scanner.nextDouble();

            if (source == 0 && dest == 0 && weight == 0) {
                break;
            }

            graph.addEdge(source, dest, weight);
        }

        testGraph(graph);

        // Optionally save to file
        System.out.print("\nSave graph to file? (y/n): ");
        String save = scanner.next();
        if (save.equalsIgnoreCase("y")) {
            System.out.print("Enter filename: ");
            String filename = scanner.next();
            try {
                GraphReader.saveGraphToFile(graph, filename);
                System.out.println("Graph saved to " + filename);
            } catch (IOException e) {
                System.err.println("Error saving file: " + e.getMessage());
            }
        }
    }

    private static void testGraph(Graph graph) {
        System.out.println("\nGraph Statistics:");
        System.out.println("Vertices: " + graph.getVertices());
        System.out.println("Edges: " + graph.getAllEdges().size());

        // Run both algorithms
        PrimMST prim = new PrimMST();
        KruskalMST kruskal = new KruskalMST();

        System.out.println("\n--- Prim's Algorithm ---");
        PerformanceMetrics primMetrics = prim.findMSTWithMetrics(graph, 1);
        System.out.println(primMetrics);

        System.out.println("\n--- Kruskal's Algorithm ---");
        PerformanceMetrics kruskalMetrics = kruskal.findMSTWithMetrics(graph);
        System.out.println(kruskalMetrics);

        // Verify same MST weight
        double primWeight = primMetrics.getTotalMSTWeight();
        double kruskalWeight = kruskalMetrics.getTotalMSTWeight();
        System.out.printf("\nMST Weight Verification: Prim=%.2f, Kruskal=%.2f, Match=%s\n",
                primWeight, kruskalWeight, Math.abs(primWeight - kruskalWeight) < 0.01);
    }

    private static void compareEdgeOrder(List<Edge> primEdges, List<Edge> kruskalEdges) {
        System.out.println("\nEdge Selection Order Comparison:");
        System.out.println("-".repeat(80));
        System.out.printf("%-40s %-40s\n", "Prim's Order", "Kruskal's Order");
        System.out.println("-".repeat(80));

        int maxSize = Math.max(primEdges.size(), kruskalEdges.size());
        for (int i = 0; i < maxSize; i++) {
            String primStr = i < primEdges.size() ? primEdges.get(i).toString() : "";
            String kruskalStr = i < kruskalEdges.size() ? kruskalEdges.get(i).toString() : "";
            System.out.printf("%-40s %-40s\n", primStr, kruskalStr);
        }

        System.out.println("\nObservation:");
        System.out.println("- Both algorithms produce MSTs with the same total weight");
        System.out.println("- Edge selection order differs based on algorithm strategy");
        System.out.println("- Prim's: Expands from starting vertex, selecting minimum edges");
        System.out.println("- Kruskal's: Processes all edges globally by weight, avoiding cycles");
    }

    private static void createSampleGraphFile() {
        try {
            Graph sampleGraph = new Graph(20);

            // Create the same graph structure from your example
            sampleGraph.addEdge(1, 2, 0.5);
            sampleGraph.addEdge(2, 4, 1.0);
            sampleGraph.addEdge(3, 6, 1.5);
            sampleGraph.addEdge(4, 8, 2.0);
            sampleGraph.addEdge(5, 10, 2.5);
            sampleGraph.addEdge(6, 12, 3.0);
            sampleGraph.addEdge(7, 14, 3.5);
            sampleGraph.addEdge(8, 16, 4.0);
            sampleGraph.addEdge(9, 18, 4.5);
            sampleGraph.addEdge(10, 20, 5.0);
            sampleGraph.addEdge(3, 1, 0.53);
            sampleGraph.addEdge(5, 2, 0.86);
            sampleGraph.addEdge(7, 3, 1.2);
            sampleGraph.addEdge(9, 4, 1.53);
            sampleGraph.addEdge(11, 5, 1.86);
            sampleGraph.addEdge(13, 6, 2.2);
            sampleGraph.addEdge(15, 7, 2.53);
            sampleGraph.addEdge(17, 8, 2.86);
            sampleGraph.addEdge(19, 9, 3.2);
            sampleGraph.addEdge(12, 11, 0.25);
            sampleGraph.addEdge(14, 13, 0.25);
            sampleGraph.addEdge(18, 17, 0.25);
            sampleGraph.addEdge(20, 19, 0.25);

            GraphReader.saveGraphToFile(sampleGraph, "graph.txt");
            System.out.println("Sample graph.txt created successfully!");
            System.out.println("You can now re-run the program with option 1.");

        } catch (IOException e) {
            System.err.println("Error creating sample file: " + e.getMessage());
        }
    }
}