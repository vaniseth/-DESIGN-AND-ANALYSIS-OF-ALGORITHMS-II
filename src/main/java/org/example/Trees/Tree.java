package org.example.Trees;
//Auther: Abdelnasser Ouda
import javafx.scene.paint.Color;

import java.io.Serializable;
import java.util.List;

public interface Tree<T extends Comparable<T>> extends Serializable {
    void insert(T value);
    boolean delete(T value);
    boolean contains(T value);
    void clear();
    int size();
    List<T> inorderTraversal();
    String type();
    Color color();
    TreeNode<T> getRoot();
}