package mst;

/**
 * Represents an edge in a weighted graph
 */
public class Edge implements Comparable<Edge> {
    private int source;
    private int destination;
    private double weight;

    public Edge(int source, int destination, double weight) {
        this.source = source;
        this.destination = destination;
        this.weight = weight;
    }

    public int getSource() {
        return source;
    }

    public int getDestination() {
        return destination;
    }

    public double getWeight() {
        return weight;
    }

    @Override
    public int compareTo(Edge other) {
        return Double.compare(this.weight, other.weight);
    }

    @Override
    public String toString() {
        return "(" + source + ", " + destination + ", " + weight + ")";
    }
}