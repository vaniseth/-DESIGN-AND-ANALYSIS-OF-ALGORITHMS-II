package org.example.Trees;
//Auther: Abdelnasser Ouda
import javafx.scene.paint.Color;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class BinarySearchTree<T extends Comparable<T>> implements Tree<T> , Serializable {
    private Node root;
    private int size;

    private class Node implements TreeNode<T> , Serializable{
        T value;
        Node left, right;

        Node(T value) {
            this.value = value;
        }

        @Override
        public T getValue() { return value; }

        @Override
        public TreeNode<T> getLeft() { return left; }

        @Override
        public TreeNode<T> getRight() { return right; }

        public String getColor() { return null; }
    }


    @Override
    public Color color() { return Color.BLACK; }

    @Override
    public void insert(T value) {
        root = insert(root, value);
        size++;
    }

    private Node insert(Node node, T value) {
        if (node == null) {
            return new Node(value);
        }
        if (value.compareTo(node.value) < 0) {
            node.left = insert(node.left, value);
        } else if (value.compareTo(node.value) > 0) {
            node.right = insert(node.right, value);
        }
        return node;
    }

    @Override
    public String type() {
         return "BST";
    }

    @Override
    public boolean delete(T value) {
        int originalSize = size;
        root = delete(root, value);
        return size < originalSize;
    }

    private Node delete(Node node, T value) {
        if (node == null) {
            return null;
        }
        if (value.compareTo(node.value) < 0) {
            node.left = delete(node.left, value);
        } else if (value.compareTo(node.value) > 0) {
            node.right = delete(node.right, value);
        } else {
            if (node.left == null) {
                size--;
                return node.right;
            } else if (node.right == null) {
                size--;
                return node.left;
            }
            Node minRight = findMin(node.right);
            node.value = minRight.value;
            node.right = delete(node.right, minRight.value);
        }
        return node;
    }

    private Node findMin(Node node) {
        while (node.left != null) {
            node = node.left;
        }
        return node;
    }

    @Override
    public boolean contains(T value) {
        return contains(root, value);
    }

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

    @Override
    public void clear() {
        root = null;
        size = 0;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public List<T> inorderTraversal() {
        List<T> result = new ArrayList<>();
        inorderTraversal(root, result);
        return result;
    }

    private void inorderTraversal(Node node, List<T> result) {
        if (node != null) {
            inorderTraversal(node.left, result);
            result.add(node.value);
            inorderTraversal(node.right, result);
        }
    }

    @Override
    public TreeNode<T> getRoot() {
        return root;
    }
}