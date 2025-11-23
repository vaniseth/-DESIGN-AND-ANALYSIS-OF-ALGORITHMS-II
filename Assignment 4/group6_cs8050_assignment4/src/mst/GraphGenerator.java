package mst;

import java.util.*;

/**
 * Generates different types of graphs for testing
 */
public class GraphGenerator {
    private static Random random = new Random();

    /**
     * Generate a sparse graph (E ≈ V)
     */
    public static Graph generateSparseGraph(int vertices) {
        Graph graph = new Graph(vertices);
        int numEdges = vertices; // Approximately V edges

        Set<String> addedEdges = new HashSet<>();

        // First, create a connected graph by making a spanning tree
        for (int i = 2; i <= vertices; i++) {
            int parent = random.nextInt(i - 1) + 1;
            double weight = random.nextDouble() * 10;
            graph.addEdge(parent, i, weight);
            addEdge(addedEdges, parent, i);
        }

        // Add remaining edges randomly
        int edgesAdded = vertices - 1;
        int attempts = 0;
        while (edgesAdded < numEdges && attempts < numEdges * 3) {
            int v1 = random.nextInt(vertices) + 1;
            int v2 = random.nextInt(vertices) + 1;

            if (v1 != v2 && !hasEdge(addedEdges, v1, v2)) {
                double weight = random.nextDouble() * 10;
                graph.addEdge(v1, v2, weight);
                addEdge(addedEdges, v1, v2);
                edgesAdded++;
            }
            attempts++;
        }

        return graph;
    }

    /**
     * Generate a dense graph (E ≈ V²/2)
     */
    public static Graph generateDenseGraph(int vertices) {
        Graph graph = new Graph(vertices);
        int maxEdges = vertices * (vertices - 1) / 2;
        int numEdges = (int)(maxEdges * 0.7); // 70% of maximum edges

        Set<String> addedEdges = new HashSet<>();

        // Add edges randomly
        int edgesAdded = 0;
        int attempts = 0;
        while (edgesAdded < numEdges && attempts < numEdges * 2) {
            int v1 = random.nextInt(vertices) + 1;
            int v2 = random.nextInt(vertices) + 1;

            if (v1 != v2 && !hasEdge(addedEdges, v1, v2)) {
                double weight = random.nextDouble() * 10;
                graph.addEdge(v1, v2, weight);
                addEdge(addedEdges, v1, v2);
                edgesAdded++;
            }
            attempts++;
        }

        return graph;
    }

    /**
     * Generate a large graph (≥500 vertices)
     */
    public static Graph generateLargeGraph(int vertices, boolean dense) {
        if (dense) {
            return generateDenseGraph(vertices);
        } else {
            return generateSparseGraph(vertices);
        }
    }

    /**
     * Generate a complete graph (every vertex connected to every other)
     */
    public static Graph generateCompleteGraph(int vertices) {
        Graph graph = new Graph(vertices);

        for (int i = 1; i <= vertices; i++) {
            for (int j = i + 1; j <= vertices; j++) {
                double weight = random.nextDouble() * 10;
                graph.addEdge(i, j, weight);
            }
        }

        return graph;
    }

    private static void addEdge(Set<String> edges, int v1, int v2) {
        String key = Math.min(v1, v2) + "-" + Math.max(v1, v2);
        edges.add(key);
    }

    private static boolean hasEdge(Set<String> edges, int v1, int v2) {
        String key = Math.min(v1, v2) + "-" + Math.max(v1, v2);
        return edges.contains(key);
    }
}