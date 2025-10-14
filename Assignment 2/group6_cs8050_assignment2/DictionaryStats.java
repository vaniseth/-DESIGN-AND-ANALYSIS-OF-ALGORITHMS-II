package group6_cs8050_assignment2;

public class DictionaryStats {
    private long totalInsertions;
    private long totalFinds;
    private long totalDeletions;
    private long totalCollisions;
    private long totalProbes;
    private long totalResizes;
    private long insertTime;
    private long findTime;
    private long deleteTime;
    private int maxChainLength;
    private int maxProbeLength;

    public DictionaryStats() {
        reset();
    }

    public void reset() {
        totalInsertions = 0;
        totalFinds = 0;
        totalDeletions = 0;
        totalCollisions = 0;
        totalProbes = 0;
        totalResizes = 0;
        insertTime = 0;
        findTime = 0;
        deleteTime = 0;
        maxChainLength = 0;
        maxProbeLength = 0;
    }

    public void recordInsertion(long timeNanos, int collisions, int probes) {
        totalInsertions++;
        insertTime += timeNanos;
        totalCollisions += collisions;
        totalProbes += probes;
    }

    public void recordFind(long timeNanos, int probes) {
        totalFinds++;
        findTime += timeNanos;
        totalProbes += probes;
    }

    public void recordDeletion(long timeNanos, int probes) {
        totalDeletions++;
        deleteTime += timeNanos;
        totalProbes += probes;
    }

    public void recordResize() {
        totalResizes++;
    }

    public void updateMaxChainLength(int length) {
        if (length > maxChainLength) {
            maxChainLength = length;
        }
    }

    public void updateMaxProbeLength(int length) {
        if (length > maxProbeLength) {
            maxProbeLength = length;
        }
    }

    public long getTotalInsertions() { return totalInsertions; }
    public long getTotalFinds() { return totalFinds; }
    public long getTotalDeletions() { return totalDeletions; }
    public long getTotalCollisions() { return totalCollisions; }
    public long getTotalProbes() { return totalProbes; }
    public long getTotalResizes() { return totalResizes; }
    public int getMaxChainLength() { return maxChainLength; }
    public int getMaxProbeLength() { return maxProbeLength; }

    public double getAverageInsertTime() {
        return totalInsertions > 0 ? (double) insertTime / totalInsertions : 0;
    }

    public double getAverageFindTime() {
        return totalFinds > 0 ? (double) findTime / totalFinds : 0;
    }

    public double getAverageDeleteTime() {
        return totalDeletions > 0 ? (double) deleteTime / totalDeletions : 0;
    }

    public double getAverageProbesPerOperation() {
        long totalOps = totalInsertions + totalFinds + totalDeletions;
        return totalOps > 0 ? (double) totalProbes / totalOps : 0;
    }

    public double getCollisionRate() {
        return totalInsertions > 0 ? (double) totalCollisions / totalInsertions : 0;
    }

    @Override
    public String toString() {
        return String.format(
                "DictionaryStats{\n" +
                        "  Insertions: %d (avg time: %.2f ns)\n" +
                        "  Finds: %d (avg time: %.2f ns)\n" +
                        "  Deletions: %d (avg time: %.2f ns)\n" +
                        "  Collisions: %d (rate: %.4f)\n" +
                        "  Total Probes: %d (avg: %.2f per op)\n" +
                        "  Resizes: %d\n" +
                        "  Max Chain Length: %d\n" +
                        "  Max Probe Length: %d\n" +
                        "}",
                totalInsertions, getAverageInsertTime(),
                totalFinds, getAverageFindTime(),
                totalDeletions, getAverageDeleteTime(),
                totalCollisions, getCollisionRate(),
                totalProbes, getAverageProbesPerOperation(),
                totalResizes,
                maxChainLength,
                maxProbeLength
        );
    }
}