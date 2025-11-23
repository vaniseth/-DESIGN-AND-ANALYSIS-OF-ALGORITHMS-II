package mst;

import java.util.*;

/**
 * Kruskal's Algorithm for Minimum Spanning Tree using Union-Find
 */
public class KruskalMST {

    public List<Edge> findMST(Graph graph) {
        List<Edge> mstEdges = new ArrayList<>();
        List<Edge> allEdges = graph.getAllEdges();

        // Sort all edges by weight
        Collections.sort(allEdges);

        // Initialize Union-Find
        UnionFind uf = new UnionFind(graph.getVertices());

        System.out.println("\nKruskal's Algorithm Edge Selection Order:");
        int edgeCount = 1;

        // Process edges in sorted order
        for (Edge edge : allEdges) {
            int source = edge.getSource();
            int destination = edge.getDestination();

            // Check if adding this edge creates a cycle
            if (!uf.connected(source, destination)) {
                // No cycle - add edge to MST
                mstEdges.add(edge);
                uf.union(source, destination);

                System.out.printf("%d. (%d,%d): %.2f\n", edgeCount++,
                        source, destination, edge.getWeight());

                // Stop when we have V-1 edges
                if (mstEdges.size() == graph.getVertices() - 1) {
                    break;
                }
            }
        }

        return mstEdges;
    }

    public PerformanceMetrics findMSTWithMetrics(Graph graph) {
        long startTime = System.nanoTime();
        Runtime runtime = Runtime.getRuntime();

        // Force garbage collection for more accurate memory measurement
        runtime.gc();
        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();

        List<Edge> mstEdges = findMSTQuiet(graph);

        long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
        long endTime = System.nanoTime();

        long executionTime = endTime - startTime;
        long memoryUsed = memoryAfter - memoryBefore;

        return new PerformanceMetrics(executionTime, memoryUsed, mstEdges);
    }

    private List<Edge> findMSTQuiet(Graph graph) {
        List<Edge> mstEdges = new ArrayList<>();
        List<Edge> allEdges = graph.getAllEdges();

        Collections.sort(allEdges);
        UnionFind uf = new UnionFind(graph.getVertices());

        for (Edge edge : allEdges) {
            int source = edge.getSource();
            int destination = edge.getDestination();

            if (!uf.connected(source, destination)) {
                mstEdges.add(edge);
                uf.union(source, destination);

                if (mstEdges.size() == graph.getVertices() - 1) {
                    break;
                }
            }
        }

        return mstEdges;
    }
}