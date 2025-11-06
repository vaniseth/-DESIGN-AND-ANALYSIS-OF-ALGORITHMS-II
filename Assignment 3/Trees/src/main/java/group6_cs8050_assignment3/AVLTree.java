package group6_cs8050_assignment3;

import javafx.scene.paint.Color;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * AVL Tree implementation - a self-balancing binary search tree
 * Maintains balance factor of -1, 0, or 1 for each node
 * Author: Group 6
 */
public class AVLTree<T extends Comparable<T>> implements Tree<T>, Serializable {
    private Node root;
    private int size;

    /**
     * Node class representing each element in the AVL tree
     */
    private class Node implements TreeNode<T>, Serializable {
        T value;
        Node left, right;
        int height;

        Node(T value) {
            this.value = value;
            this.height = 1;
        }

        @Override
        public T getValue() { return value; }

        @Override
        public TreeNode<T> getLeft() { return left; }

        @Override
        public TreeNode<T> getRight() { return right; }

        @Override
        public String getColor() { return null; }
    }

    @Override
    public Color color() { return Color.BLUE; }

    @Override
    public String type() { return "AVL"; }

    /**
     * Gets the height of a node
     */
    private int height(Node node) {
        return node == null ? 0 : node.height;
    }

    /**
     * Calculates balance factor of a node
     */
    private int getBalance(Node node) {
        return node == null ? 0 : height(node.left) - height(node.right);
    }

    /**
     * Updates the height of a node based on children heights
     */
    private void updateHeight(Node node) {
        if (node != null) {
            node.height = 1 + Math.max(height(node.left), height(node.right));
        }
    }

    /**
     * Right rotation for balancing
     */
    private Node rotateRight(Node y) {
        Node x = y.left;
        Node T2 = x.right;

        x.right = y;
        y.left = T2;

        updateHeight(y);
        updateHeight(x);

        return x;
    }

    /**
     * Left rotation for balancing
     */
    private Node rotateLeft(Node x) {
        Node y = x.right;
        Node T2 = y.left;

        y.left = x;
        x.right = T2;

        updateHeight(x);
        updateHeight(y);

        return y;
    }

    /**
     * Inserts a value into the AVL tree
     */
    @Override
    public void insert(T value) {
        root = insert(root, value);
    }

    /**
     * Recursive insert with balancing
     */
    private Node insert(Node node, T value) {
        // Standard BST insertion
        if (node == null) {
            size++;
            return new Node(value);
        }

        if (value.compareTo(node.value) < 0) {
            node.left = insert(node.left, value);
        } else if (value.compareTo(node.value) > 0) {
            node.right = insert(node.right, value);
        } else {
            return node; // Duplicate values not allowed
        }

        // Update height of current node
        updateHeight(node);

        // Get balance factor
        int balance = getBalance(node);

        // Left Left Case
        if (balance > 1 && value.compareTo(node.left.value) < 0) {
            return rotateRight(node);
        }

        // Right Right Case
        if (balance < -1 && value.compareTo(node.right.value) > 0) {
            return rotateLeft(node);
        }

        // Left Right Case
        if (balance > 1 && value.compareTo(node.left.value) > 0) {
            node.left = rotateLeft(node.left);
            return rotateRight(node);
        }

        // Right Left Case
        if (balance < -1 && value.compareTo(node.right.value) < 0) {
            node.right = rotateRight(node.right);
            return rotateLeft(node);
        }

        return node;
    }

    /**
     * Deletes a value from the AVL tree
     */
    @Override
    public boolean delete(T value) {
        int originalSize = size;
        root = delete(root, value);
        return size < originalSize;
    }

    /**
     * Recursive delete with balancing
     */
    private Node delete(Node node, T value) {
        if (node == null) {
            return null;
        }

        // Standard BST deletion
        if (value.compareTo(node.value) < 0) {
            node.left = delete(node.left, value);
        } else if (value.compareTo(node.value) > 0) {
            node.right = delete(node.right, value);
        } else {
            // Node to be deleted found
            if (node.left == null || node.right == null) {
                Node temp = (node.left != null) ? node.left : node.right;
                if (temp == null) {
                    node = null;
                } else {
                    node = temp;
                }
                size--;
            } else {
                // Node with two children
                Node temp = findMin(node.right);
                node.value = temp.value;
                node.right = delete(node.right, temp.value);
            }
        }

        if (node == null) {
            return null;
        }

        // Update height
        updateHeight(node);

        // Balance the node
        int balance = getBalance(node);

        // Left Left Case
        if (balance > 1 && getBalance(node.left) >= 0) {
            return rotateRight(node);
        }

        // Left Right Case
        if (balance > 1 && getBalance(node.left) < 0) {
            node.left = rotateLeft(node.left);
            return rotateRight(node);
        }

        // Right Right Case
        if (balance < -1 && getBalance(node.right) <= 0) {
            return rotateLeft(node);
        }

        // Right Left Case
        if (balance < -1 && getBalance(node.right) > 0) {
            node.right = rotateRight(node.right);
            return rotateLeft(node);
        }

        return node;
    }

    /**
     * Finds the minimum value node in a subtree
     */
    private Node findMin(Node node) {
        while (node.left != null) {
            node = node.left;
        }
        return node;
    }

    /**
     * Searches for a value in the tree
     */
    @Override
    public boolean contains(T value) {
        return contains(root, value);
    }

    /**
     * Recursive search for a value
     */
    private boolean contains(Node node, T value) {
        if (node == null) {
            return false;
        }
        if (value.compareTo(node.value) == 0) {
            return true;
        } else if (value.compareTo(node.value) < 0) {
            return contains(node.left, value);
        } else {
            return contains(node.right, value);
        }
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
     * Recursive in-order traversal helper
     */
    private void inorderTraversal(Node node, List<T> result) {
        if (node != null) {
            inorderTraversal(node.left, result);
            result.add(node.value);
            inorderTraversal(node.right, result);
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