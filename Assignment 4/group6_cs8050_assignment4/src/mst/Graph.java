package mst;

import java.util.*;

/**
 * Represents a weighted undirected graph using adjacency list
 */
public class Graph {
    private int vertices;
    private List<List<Edge>> adjacencyList;

    public Graph(int vertices) {
        this.vertices = vertices;
        this.adjacencyList = new ArrayList<>(vertices + 1);

        // Initialize adjacency list (vertices numbered 1 to n)
        for (int i = 0; i <= vertices; i++) {
            adjacencyList.add(new ArrayList<>());
        }
    }

    public void addEdge(int source, int destination, double weight) {
        // Undirected graph - add edge in both directions
        adjacencyList.get(source).add(new Edge(source, destination, weight));
        adjacencyList.get(destination).add(new Edge(destination, source, weight));
    }

    public int getVertices() {
        return vertices;
    }

    public List<Edge> getEdges(int vertex) {
        return adjacencyList.get(vertex);
    }

    public List<Edge> getAllEdges() {
        List<Edge> allEdges = new ArrayList<>();
        Set<String> added = new HashSet<>();

        for (int i = 1; i <= vertices; i++) {
            for (Edge edge : adjacencyList.get(i)) {
                // Avoid duplicate edges (since graph is undirected)
                String edgeKey = Math.min(edge.getSource(), edge.getDestination()) + "-" +
                        Math.max(edge.getSource(), edge.getDestination());
                if (!added.contains(edgeKey)) {
                    allEdges.add(edge);
                    added.add(edgeKey);
                }
            }
        }

        return allEdges;
    }

    public void printGraph() {
        System.out.println("The input graph is represented in an adjacent list as:");
        for (int i = 1; i <= vertices; i++) {
            System.out.print(i + " --> ");
            List<Edge> edges = adjacencyList.get(i);
            for (int j = 0; j < edges.size(); j++) {
                Edge edge = edges.get(j);
                System.out.print("(" + edge.getDestination() + ", " + edge.getWeight() + ")");
                if (j < edges.size() - 1) {
                    System.out.print(" --> ");
                }
            }
            System.out.println();
        }
    }

    public void printMST(List<Edge> mstEdges, String algorithmName) {
        System.out.println("\nThe minimum cost spanning tree is represented in an adjacent list as (" + algorithmName + "):");

        // Build adjacency list for MST
        List<List<Edge>> mstAdjList = new ArrayList<>(vertices + 1);
        for (int i = 0; i <= vertices; i++) {
            mstAdjList.add(new ArrayList<>());
        }

        // Add MST edges
        for (Edge edge : mstEdges) {
            mstAdjList.get(edge.getSource()).add(edge);
            mstAdjList.get(edge.getDestination()).add(new Edge(edge.getDestination(), edge.getSource(), edge.getWeight()));
        }

        // Print MST adjacency list
        for (int i = 1; i <= vertices; i++) {
            System.out.print(i + " --> ");
            List<Edge> edges = mstAdjList.get(i);
            for (int j = 0; j < edges.size(); j++) {
                Edge edge = edges.get(j);
                System.out.print("(" + edge.getDestination() + ", " + edge.getWeight() + ")");
                if (j < edges.size() - 1) {
                    System.out.print(" --> ");
                }
            }
            System.out.println();
        }

        // Calculate and print total weight
        double totalWeight = 0;
        for (Edge edge : mstEdges) {
            totalWeight += edge.getWeight();
        }
        System.out.printf("\nTotal MST Weight: %.2f\n", totalWeight);
    }
}