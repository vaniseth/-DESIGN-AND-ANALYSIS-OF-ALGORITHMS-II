package mst;

/**
 * Union-Find (Disjoint Set) data structure with path compression and union by rank
 */
public class UnionFind {
    private int[] parent;
    private int[] rank;

    public UnionFind(int size) {
        parent = new int[size + 1];
        rank = new int[size + 1];

        // Initialize: each element is its own parent
        for (int i = 0; i <= size; i++) {
            parent[i] = i;
            rank[i] = 0;
        }
    }

    /**
     * Find the root of the set containing x with path compression
     */
    public int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]); // Path compression
        }
        return parent[x];
    }

    /**
     * Union two sets containing x and y using union by rank
     */
    public void union(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);

        if (rootX == rootY) {
            return; // Already in same set
        }

        // Union by rank
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
    }

    /**
     * Check if x and y are in the same set
     */
    public boolean connected(int x, int y) {
        return find(x) == find(y);
    }
}