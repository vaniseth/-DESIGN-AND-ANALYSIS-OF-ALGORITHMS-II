package mst;

import java.util.List;

/**
 * Stores performance metrics for MST algorithms
 */
public class PerformanceMetrics {
    private long executionTimeNanos;
    private long memoryUsedBytes;
    private List<Edge> mstEdges;

    public PerformanceMetrics(long executionTimeNanos, long memoryUsedBytes, List<Edge> mstEdges) {
        this.executionTimeNanos = executionTimeNanos;
        this.memoryUsedBytes = memoryUsedBytes;
        this.mstEdges = mstEdges;
    }

    public long getExecutionTimeNanos() {
        return executionTimeNanos;
    }

    public double getExecutionTimeMillis() {
        return executionTimeNanos / 1_000_000.0;
    }

    public long getMemoryUsedBytes() {
        return memoryUsedBytes;
    }

    public double getMemoryUsedKB() {
        return memoryUsedBytes / 1024.0;
    }

    public List<Edge> getMstEdges() {
        return mstEdges;
    }

    public double getTotalMSTWeight() {
        double totalWeight = 0;
        for (Edge edge : mstEdges) {
            totalWeight += edge.getWeight();
        }
        return totalWeight;
    }

    @Override
    public String toString() {
        return String.format("Execution Time: %.3f ms, Memory Used: %.2f KB, MST Weight: %.2f",
                getExecutionTimeMillis(), getMemoryUsedKB(), getTotalMSTWeight());
    }
}