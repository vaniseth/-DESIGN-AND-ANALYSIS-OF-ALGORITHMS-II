package mst;

import java.util.*;

/**
 * Prim's Algorithm for Minimum Spanning Tree using a Min-Heap
 */
public class PrimMST {

    public List<Edge> findMST(Graph graph, int startVertex) {
        List<Edge> mstEdges = new ArrayList<>();
        Set<Integer> inMST = new HashSet<>();
        PriorityQueue<Edge> minHeap = new PriorityQueue<>();

        // Start with the starting vertex
        inMST.add(startVertex);

        // Add all edges from starting vertex to heap
        for (Edge edge : graph.getEdges(startVertex)) {
            minHeap.offer(edge);
        }

        System.out.println("\nPrim's Algorithm Edge Selection Order:");
        int edgeCount = 1;

        // Continue until we have V-1 edges or heap is empty
        while (!minHeap.isEmpty() && mstEdges.size() < graph.getVertices() - 1) {
            Edge minEdge = minHeap.poll();

            // Skip if destination is already in MST (would create cycle)
            if (inMST.contains(minEdge.getDestination())) {
                continue;
            }

            // Add edge to MST
            mstEdges.add(minEdge);
            inMST.add(minEdge.getDestination());

            System.out.printf("%d. (%d,%d): %.2f\n", edgeCount++,
                    minEdge.getSource(), minEdge.getDestination(), minEdge.getWeight());

            // Add all edges from newly added vertex to heap
            for (Edge edge : graph.getEdges(minEdge.getDestination())) {
                if (!inMST.contains(edge.getDestination())) {
                    minHeap.offer(edge);
                }
            }
        }

        return mstEdges;
    }

    public PerformanceMetrics findMSTWithMetrics(Graph graph, int startVertex) {
        long startTime = System.nanoTime();
        Runtime runtime = Runtime.getRuntime();

        // Force garbage collection for more accurate memory measurement
        runtime.gc();
        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();

        List<Edge> mstEdges = findMSTQuiet(graph, startVertex);

        long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
        long endTime = System.nanoTime();

        long executionTime = endTime - startTime;
        long memoryUsed = memoryAfter - memoryBefore;

        return new PerformanceMetrics(executionTime, memoryUsed, mstEdges);
    }

    private List<Edge> findMSTQuiet(Graph graph, int startVertex) {
        List<Edge> mstEdges = new ArrayList<>();
        Set<Integer> inMST = new HashSet<>();
        PriorityQueue<Edge> minHeap = new PriorityQueue<>();

        inMST.add(startVertex);

        for (Edge edge : graph.getEdges(startVertex)) {
            minHeap.offer(edge);
        }

        while (!minHeap.isEmpty() && mstEdges.size() < graph.getVertices() - 1) {
            Edge minEdge = minHeap.poll();

            if (inMST.contains(minEdge.getDestination())) {
                continue;
            }

            mstEdges.add(minEdge);
            inMST.add(minEdge.getDestination());

            for (Edge edge : graph.getEdges(minEdge.getDestination())) {
                if (!inMST.contains(edge.getDestination())) {
                    minHeap.offer(edge);
                }
            }
        }

        return mstEdges;
    }
}