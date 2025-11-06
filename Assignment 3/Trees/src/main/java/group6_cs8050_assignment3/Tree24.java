package group6_cs8050_assignment3;

import javafx.scene.paint.Color;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * 2-4 Tree implementation - a B-tree of order 4
 * Each node can have 1-3 keys and 2-4 children
 * Always balanced - all leaves at same level
 * Author: Group 6
 */
public class Tree24<T extends Comparable<T>> implements Tree<T>, Serializable {
    public Node root; // Made public for visualization access
    private int size;

    /**
     * Node class for 2-4 Tree
     * Can contain 1-3 values and 2-4 children
     */
    public class Node implements TreeNode<T>, Serializable {
        public List<T> values;
        public List<Node> children;
        Node parent;

        public Node() {
            values = new ArrayList<>();
            children = new ArrayList<>();
            parent = null;
        }

        boolean isLeaf() {
            return children.isEmpty();
        }

        boolean isFull() {
            return values.size() == 3;
        }

        @Override
        public T getValue() {
            return values.isEmpty() ? null : values.get(0);
        }

        @Override
        public TreeNode<T> getLeft() {
            return children.isEmpty() ? null : children.get(0);
        }

        @Override
        public TreeNode<T> getRight() {
            return children.size() > 1 ? children.get(children.size() - 1) : null;
        }

        @Override
        public String getColor() {
            return null;
        }

        /**
         * Get all values as string for visualization
         */
        public String getAllValues() {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < values.size(); i++) {
                sb.append(values.get(i));
                if (i < values.size() - 1) sb.append(",");
            }
            return sb.toString();
        }
    }

    /**
     * Returns the color used for visualization (Purple for 2-4 Tree)
     */
    @Override
    public Color color() {
        return Color.PURPLE;
    }

    /**
     * Returns the tree type identifier
     */
    @Override
    public String type() {
        return "2-4";
    }

    /**
     * Inserts a value into the 2-4 Tree
     * Splits nodes when they become full
     */
    @Override
    public void insert(T value) {
        if (root == null) {
            root = new Node();
            root.values.add(value);
            size++;
            return;
        }

        if (root.isFull()) {
            Node newRoot = new Node();
            newRoot.children.add(root);
            root.parent = newRoot;
            splitChild(newRoot, 0);
            root = newRoot;
        }

        insertNonFull(root, value);
        size++;
    }

    /**
     * Inserts value into a non-full node
     */
    private void insertNonFull(Node node, T value) {
        if (node.isLeaf()) {
            // Insert into sorted position
            int pos = 0;
            while (pos < node.values.size() && value.compareTo(node.values.get(pos)) > 0) {
                pos++;
            }
            node.values.add(pos, value);
        } else {
            // Find child to insert into
            int pos = 0;
            while (pos < node.values.size() && value.compareTo(node.values.get(pos)) > 0) {
                pos++;
            }

            Node child = node.children.get(pos);
            if (child.isFull()) {
                splitChild(node, pos);
                if (value.compareTo(node.values.get(pos)) > 0) {
                    pos++;
                }
            }
            insertNonFull(node.children.get(pos), value);
        }
    }

    /**
     * Splits a full child node
     */
    private void splitChild(Node parent, int index) {
        Node fullChild = parent.children.get(index);
        Node newChild = new Node();
        newChild.parent = parent;

        T middleValue = fullChild.values.get(1);

        // Move right value to new node
        newChild.values.add(fullChild.values.get(2));
        fullChild.values.remove(2);
        fullChild.values.remove(1);

        // Move right children if not leaf
        if (!fullChild.isLeaf()) {
            newChild.children.add(fullChild.children.get(2));
            newChild.children.add(fullChild.children.get(3));
            fullChild.children.get(2).parent = newChild;
            fullChild.children.get(3).parent = newChild;
            fullChild.children.remove(3);
            fullChild.children.remove(2);
        }

        // Insert middle value into parent
        parent.values.add(index, middleValue);
        parent.children.add(index + 1, newChild);
    }

    /**
     * Deletes a value from the 2-4 Tree
     */
    @Override
    public boolean delete(T value) {
        if (root == null) {
            return false;
        }

        boolean deleted = deleteValue(root, value);

        // If root is empty after deletion, make its only child the new root
        if (root.values.isEmpty() && !root.children.isEmpty()) {
            root = root.children.get(0);
            root.parent = null;
        }

        if (deleted) {
            size--;
        }
        return deleted;
    }

    /**
     * Deletes a value from the tree
     */
    private boolean deleteValue(Node node, T value) {
        int index = findIndexInNode(node, value);

        if (index < node.values.size() && node.values.get(index).compareTo(value) == 0) {
            // Value found in current node
            if (node.isLeaf()) {
                node.values.remove(index);
                return true;
            } else {
                return deleteFromInternal(node, index);
            }
        } else if (!node.isLeaf()) {
            // Value not in current node, continue search in children
            boolean isInSubtree = (index == node.values.size()) ||
                    value.compareTo(node.values.get(index)) < 0;

            Node child = node.children.get(index);

            if (child.values.size() == 1) {
                fill(node, index);
            }

            if (index < node.children.size()) {
                return deleteValue(node.children.get(index), value);
            }
        }

        return false;
    }

    /**
     * Deletes value from internal node
     */
    private boolean deleteFromInternal(Node node, int index) {
        T value = node.values.get(index);

        if (node.children.get(index).values.size() >= 2) {
            T pred = getPredecessor(node, index);
            node.values.set(index, pred);
            return deleteValue(node.children.get(index), pred);
        } else if (node.children.get(index + 1).values.size() >= 2) {
            T succ = getSuccessor(node, index);
            node.values.set(index, succ);
            return deleteValue(node.children.get(index + 1), succ);
        } else {
            merge(node, index);
            return deleteValue(node.children.get(index), value);
        }
    }

    /**
     * Gets predecessor value
     */
    private T getPredecessor(Node node, int index) {
        Node current = node.children.get(index);
        while (!current.isLeaf()) {
            current = current.children.get(current.children.size() - 1);
        }
        return current.values.get(current.values.size() - 1);
    }

    /**
     * Gets successor value
     */
    private T getSuccessor(Node node, int index) {
        Node current = node.children.get(index + 1);
        while (!current.isLeaf()) {
            current = current.children.get(0);
        }
        return current.values.get(0);
    }

    /**
     * Fills child node with minimum keys
     */
    private void fill(Node node, int index) {
        if (index != 0 && node.children.get(index - 1).values.size() >= 2) {
            borrowFromPrev(node, index);
        } else if (index != node.children.size() - 1 &&
                node.children.get(index + 1).values.size() >= 2) {
            borrowFromNext(node, index);
        } else {
            if (index != node.children.size() - 1) {
                merge(node, index);
            } else {
                merge(node, index - 1);
            }
        }
    }

    /**
     * Borrows a key from previous sibling
     */
    private void borrowFromPrev(Node node, int index) {
        Node child = node.children.get(index);
        Node sibling = node.children.get(index - 1);

        child.values.add(0, node.values.get(index - 1));
        node.values.set(index - 1, sibling.values.remove(sibling.values.size() - 1));

        if (!child.isLeaf()) {
            child.children.add(0, sibling.children.remove(sibling.children.size() - 1));
            child.children.get(0).parent = child;
        }
    }

    /**
     * Borrows a key from next sibling
     */
    private void borrowFromNext(Node node, int index) {
        Node child = node.children.get(index);
        Node sibling = node.children.get(index + 1);

        child.values.add(node.values.get(index));
        node.values.set(index, sibling.values.remove(0));

        if (!child.isLeaf()) {
            child.children.add(sibling.children.remove(0));
            child.children.get(child.children.size() - 1).parent = child;
        }
    }

    /**
     * Merges child with sibling
     */
    private void merge(Node node, int index) {
        Node child = node.children.get(index);
        Node sibling = node.children.get(index + 1);

        child.values.add(node.values.remove(index));
        child.values.addAll(sibling.values);

        if (!child.isLeaf()) {
            for (Node grandchild : sibling.children) {
                grandchild.parent = child;
                child.children.add(grandchild);
            }
        }

        node.children.remove(index + 1);
    }

    /**
     * Finds index for value in node
     */
    private int findIndexInNode(Node node, T value) {
        int index = 0;
        while (index < node.values.size() && value.compareTo(node.values.get(index)) > 0) {
            index++;
        }
        return index;
    }

    /**
     * Searches for a value in the tree
     */
    @Override
    public boolean contains(T value) {
        return search(root, value);
    }

    /**
     * Searches for value in tree
     */
    private boolean search(Node node, T value) {
        if (node == null) {
            return false;
        }

        int i = 0;
        while (i < node.values.size() && value.compareTo(node.values.get(i)) > 0) {
            i++;
        }

        if (i < node.values.size() && value.compareTo(node.values.get(i)) == 0) {
            return true;
        }

        if (node.isLeaf()) {
            return false;
        }

        return search(node.children.get(i), value);
    }

    /**
     * Clears all nodes from the tree
     */
    @Override
    public void clear() {
        root = null;
        size = 0;
    }

    /**
     * Returns the size of the tree
     */
    @Override
    public int size() {
        return size;
    }

    /**
     * Performs in-order traversal of the tree
     */
    @Override
    public List<T> inorderTraversal() {
        List<T> result = new ArrayList<>();
        inorderTraversal(root, result);
        return result;
    }

    /**
     * Performs in-order traversal helper
     */
    private void inorderTraversal(Node node, List<T> result) {
        if (node != null) {
            for (int i = 0; i < node.values.size(); i++) {
                if (i < node.children.size()) {
                    inorderTraversal(node.children.get(i), result);
                }
                result.add(node.values.get(i));
            }
            if (!node.children.isEmpty()) {
                inorderTraversal(node.children.get(node.children.size() - 1), result);
            }
        }
    }

    /**
     * Returns the root of the tree
     */
    @Override
    public TreeNode<T> getRoot() {
        return root;
    }
}