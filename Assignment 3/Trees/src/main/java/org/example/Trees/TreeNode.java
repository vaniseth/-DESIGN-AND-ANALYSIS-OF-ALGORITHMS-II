package org.example.Trees;
//Auther: Abdelnasser Ouda
public interface TreeNode<T> {
    T getValue();
    TreeNode<T> getLeft();
    TreeNode<T> getRight();
    String getColor();
}
