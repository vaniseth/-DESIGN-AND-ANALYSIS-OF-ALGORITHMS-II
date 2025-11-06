package group6_cs8050_assignment3;

import javafx.scene.paint.Color;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Red-Black Tree implementation - a self-balancing binary search tree
 * Properties: 1) Every node is red or black, 2) Root is black,
 * 3) All leaves (NIL) are black, 4) Red nodes have black children,
 * 5) All paths from node to leaves have same number of black nodes
 * Author: Group 6
 */
public class RedBlackTree<T extends Comparable<T>> implements Tree<T>, Serializable {
    private static final boolean RED = true;
    private static final boolean BLACK = false;

    private Node root;
    private int size;

    /**
     * Node class for Red-Black Tree with color property
     */
    private class Node implements TreeNode<T>, Serializable {
        T value;
        Node left, right, parent;
        boolean color;

        Node(T value) {
            this.value = value;
            this.color = RED;
        }

        @Override
        public T getValue() { return value; }

        @Override
        public TreeNode<T> getLeft() { return left; }

        @Override
        public TreeNode<T> getRight() { return right; }

        @Override
        public String getColor() { return color == RED ? "RED" : "BLACK"; }
    }

    @Override
    public Color color() { return Color.BLACK; }

    @Override
    public String type() { return "RBT"; }

    /**
     * Inserts a value into the Red-Black Tree
     * Maintains Red-Black properties through rotations and recoloring
     */
    @Override
    public void insert(T value) {
        Node newNode = new Node(value);
        if (root == null) {
            root = newNode;
            root.color = BLACK;
            size++;
            return;
        }

        Node parent = null;
        Node current = root;

        // Find insertion position
        while (current != null) {
            parent = current;
            if (value.compareTo(current.value) < 0) {
                current = current.left;
            } else if (value.compareTo(current.value) > 0) {
                current = current.right;
            } else {
                return; // Duplicate value
            }
        }

        // Insert node
        newNode.parent = parent;
        if (value.compareTo(parent.value) < 0) {
            parent.left = newNode;
        } else {
            parent.right = newNode;
        }

        size++;
        fixInsert(newNode);
    }

    /**
     * Fixes Red-Black Tree properties after insertion
     */
    private void fixInsert(Node node) {
        while (node != root && node.parent.color == RED) {
            if (node.parent == node.parent.parent.left) {
                Node uncle = node.parent.parent.right;

                if (uncle != null && uncle.color == RED) {
                    // Case 1: Uncle is red
                    node.parent.color = BLACK;
                    uncle.color = BLACK;
                    node.parent.parent.color = RED;
                    node = node.parent.parent;
                } else {
                    if (node == node.parent.right) {
                        // Case 2: Node is right child
                        node = node.parent;
                        rotateLeft(node);
                    }
                    // Case 3: Node is left child
                    node.parent.color = BLACK;
                    node.parent.parent.color = RED;
                    rotateRight(node.parent.parent);
                }
            } else {
                Node uncle = node.parent.parent.left;

                if (uncle != null && uncle.color == RED) {
                    node.parent.color = BLACK;
                    uncle.color = BLACK;
                    node.parent.parent.color = RED;
                    node = node.parent.parent;
                } else {
                    if (node == node.parent.left) {
                        node = node.parent;
                        rotateRight(node);
                    }
                    node.parent.color = BLACK;
                    node.parent.parent.color = RED;
                    rotateLeft(node.parent.parent);
                }
            }
        }
        root.color = BLACK;
    }

    /**
     * Left rotation operation
     */
    private void rotateLeft(Node x) {
        Node y = x.right;
        x.right = y.left;

        if (y.left != null) {
            y.left.parent = x;
        }

        y.parent = x.parent;

        if (x.parent == null) {
            root = y;
        } else if (x == x.parent.left) {
            x.parent.left = y;
        } else {
            x.parent.right = y;
        }

        y.left = x;
        x.parent = y;
    }

    /**
     * Right rotation operation
     */
    private void rotateRight(Node y) {
        Node x = y.left;
        y.left = x.right;

        if (x.right != null) {
            x.right.parent = y;
        }

        x.parent = y.parent;

        if (y.parent == null) {
            root = x;
        } else if (y == y.parent.right) {
            y.parent.right = x;
        } else {
            y.parent.left = x;
        }

        x.right = y;
        y.parent = x;
    }

    /**
     * Deletes a value from the Red-Black Tree
     */
    @Override
    public boolean delete(T value) {
        Node node = findNode(root, value);
        if (node == null) {
            return false;
        }

        deleteNode(node);
        size--;
        return true;
    }

    /**
     * Finds a node with given value
     */
    private Node findNode(Node node, T value) {
        while (node != null) {
            if (value.compareTo(node.value) == 0) {
                return node;
            } else if (value.compareTo(node.value) < 0) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        return null;
    }

    /**
     * Deletes a node and fixes Red-Black properties
     */
    private void deleteNode(Node node) {
        Node replacement;
        Node child;

        // Find replacement node
        if (node.left == null || node.right == null) {
            replacement = node;
        } else {
            replacement = findMin(node.right);
        }

        // Get child of replacement
        if (replacement.left != null) {
            child = replacement.left;
        } else {
            child = replacement.right;
        }

        // Remove replacement from tree
        if (child != null) {
            child.parent = replacement.parent;
        }

        if (replacement.parent == null) {
            root = child;
        } else if (replacement == replacement.parent.left) {
            replacement.parent.left = child;
        } else {
            replacement.parent.right = child;
        }

        // Copy replacement value to node
        if (replacement != node) {
            node.value = replacement.value;
        }

        // Fix Red-Black properties if needed
        if (replacement.color == BLACK && child != null) {
            fixDelete(child);
        }
    }

    /**
     * Fixes Red-Black Tree properties after deletion
     */
    private void fixDelete(Node node) {
        while (node != root && (node == null || node.color == BLACK)) {
            if (node == node.parent.left) {
                Node sibling = node.parent.right;

                if (sibling != null && sibling.color == RED) {
                    sibling.color = BLACK;
                    node.parent.color = RED;
                    rotateLeft(node.parent);
                    sibling = node.parent.right;
                }

                if (sibling != null &&
                        (sibling.left == null || sibling.left.color == BLACK) &&
                        (sibling.right == null || sibling.right.color == BLACK)) {
                    sibling.color = RED;
                    node = node.parent;
                } else {
                    if (sibling != null && (sibling.right == null || sibling.right.color == BLACK)) {
                        if (sibling.left != null) {
                            sibling.left.color = BLACK;
                        }
                        sibling.color = RED;
                        rotateRight(sibling);
                        sibling = node.parent.right;
                    }

                    if (sibling != null) {
                        sibling.color = node.parent.color;
                        if (sibling.right != null) {
                            sibling.right.color = BLACK;
                        }
                    }
                    node.parent.color = BLACK;
                    rotateLeft(node.parent);
                    node = root;
                }
            } else {
                Node sibling = node.parent.left;

                if (sibling != null && sibling.color == RED) {
                    sibling.color = BLACK;
                    node.parent.color = RED;
                    rotateRight(node.parent);
                    sibling = node.parent.left;
                }

                if (sibling != null &&
                        (sibling.right == null || sibling.right.color == BLACK) &&
                        (sibling.left == null || sibling.left.color == BLACK)) {
                    sibling.color = RED;
                    node = node.parent;
                } else {
                    if (sibling != null && (sibling.left == null || sibling.left.color == BLACK)) {
                        if (sibling.right != null) {
                            sibling.right.color = BLACK;
                        }
                        sibling.color = RED;
                        rotateLeft(sibling);
                        sibling = node.parent.left;
                    }

                    if (sibling != null) {
                        sibling.color = node.parent.color;
                        if (sibling.left != null) {
                            sibling.left.color = BLACK;
                        }
                    }
                    node.parent.color = BLACK;
                    rotateRight(node.parent);
                    node = root;
                }
            }
        }

        if (node != null) {
            node.color = BLACK;
        }
    }

    /**
     * Finds minimum value node in subtree
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
        return findNode(root, value) != null;
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