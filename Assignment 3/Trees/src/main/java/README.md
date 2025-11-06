## Group 6 CS8050 Assignment 3

### Assignment Instructions:
In this assignment, you will develop a JavaFX application that allows users to interact with and visualize
various tree data structures. The objective is to help you understand the implementation and behavior of
different tree types while gaining hands-on experience in creating graphical user interfaces.
By the end of this assignment, you should be able to:

- Write Java code to implement a simple tree system with functionalities to insert, delete, search, store
and retrieve different tree types.
- Understand how to implement the Tree ADT (Abstract Data Type) using the Node ADT.
- Use JavaFX and Scene Builder to create a graphical user interface (GUI) for the application.

### Project Description
This JavaFX application provides an interactive visualization tool for six different tree data structures: Binary Search Tree, AVL Tree, Red-Black Tree, Min Heap, Max Heap, and 2-4 Tree. Users can perform insert, delete, search, clear, and in-order traversal operations through a graphical interface, with real-time visual representation of the tree structure on a canvas using color-coded nodes for different tree types. The application also supports saving and loading trees to/from files using Java serialization, allowing users to save and restore tree structures across sessions.

### Implementations
We have implemented the following Tree-based implementations:
- AVL Tree
- RedBlack Tree
- MinHeap
- MaxHeap
- 2-4  Tree

Each tree structure can be visualized using different colours. We have also implemented special handling for 2-4 Tree as it has multi-value nodes.


#### Supported Operations
The code supports an interactive UI with the following features:
- Insert
- Delete
- Search
- Clear
- In-order Traversal
- Save and Load the tree file

All these operations can be accessed through different buttons that are there on the UI

```commandline
src/
   └── main/
       └── java/
           └── group6_cs8050_assignment3/
               ├── BinarySearchTree.java
               ├── Heap.java
               ├── Tree.java
               ├── TreeNode.java
               ├── AVLTree.java
               ├── RedBlackTree.java
               ├── MinHeap.java
               ├── MaxHeap.java
               ├── Tree24.java
               ├── TreeVisualizerApp.java
               └── TreeVisualizerController.java
           └── module-info.java
           └── README.md
```

### THANK YOU!!