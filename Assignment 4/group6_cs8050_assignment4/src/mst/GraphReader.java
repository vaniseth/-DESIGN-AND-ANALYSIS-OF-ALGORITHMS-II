package mst;

import java.io.*;
import java.util.*;

/**
 * Reads graphs from input files
 */
public class GraphReader {

    public static Graph readGraphFromFile(String filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));

        // Read number of vertices
        String firstLine = reader.readLine();
        int vertices = Integer.parseInt(firstLine.trim());

        Graph graph = new Graph(vertices);

        // Read edges
        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) {
                continue;
            }

            String[] parts = line.split("\\s+");
            if (parts.length == 3) {
                int source = Integer.parseInt(parts[0]);
                int destination = Integer.parseInt(parts[1]);
                double weight = Double.parseDouble(parts[2]);

                graph.addEdge(source, destination, weight);
            }
        }

        reader.close();
        return graph;
    }

    public static void saveGraphToFile(Graph graph, String filename) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(filename));

        // Write number of vertices
        writer.write(graph.getVertices() + "\n");

        // Write edges (avoid duplicates)
        List<Edge> allEdges = graph.getAllEdges();
        for (Edge edge : allEdges) {
            writer.write(edge.getSource() + " " + edge.getDestination() + " " + edge.getWeight() + "\n");
        }

        writer.close();
    }
}