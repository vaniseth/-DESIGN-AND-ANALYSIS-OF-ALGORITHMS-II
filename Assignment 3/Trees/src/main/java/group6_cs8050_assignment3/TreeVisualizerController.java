package group6_cs8050_assignment3;

import javafx.scene.canvas.Canvas;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.*;
import javafx.scene.control.*;
import javafx.scene.canvas.*;
import javafx.scene.paint.Color;
import javafx.geometry.Insets;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.scene.text.Font;

import java.io.*;
import java.util.*;
import java.util.List;

/**
 * Controller for Tree Visualizer Application
 * Handles user interactions and coordinates between UI and tree implementations
 * Author: Group 6
 */
public class TreeVisualizerController {
    private VBox view;
    private ComboBox<String> treeTypeComboBox;
    private TextField inputField;
    private Button insertButton, deleteButton, searchButton, clearButton, inOrderButton;
    private Canvas treeCanvas;
    private TextArea outputArea;

    private Tree<Integer> currentTree;
    private Map<String, Tree<Integer>> trees;
    private Stage stage;

    // For search highlighting
    private Integer highlightedValue = null;

    /**
     * Sets the stage for file dialogs
     */
    public void setStage(Stage stage) {
        this.stage = stage;
    }

    /**
     * Constructor - initializes controller
     */
    public TreeVisualizerController() {
        initializeTrees();
        initializeView();
        setupEventHandlers();
    }

    /**
     * Initializes all tree data structures
     */
    private void initializeTrees() {
        trees = new HashMap<>();
        trees.put("Binary Search Tree", new BinarySearchTree<>());
        trees.put("AVL Tree", new AVLTree<>());
        trees.put("Red-Black Tree", new RedBlackTree<>());
        trees.put("Min Heap", new MinHeap<>());
        trees.put("Max Heap", new MaxHeap<>());
        trees.put("2-4 Tree", new Tree24<>());
        currentTree = trees.get("Binary Search Tree");
    }

    /**
     * Initializes the user interface components
     */
    private void initializeView() {
        view = new VBox(10);
        view.setPadding(new Insets(10));

        // Tree type selection dropdown
        treeTypeComboBox = new ComboBox<>();
        treeTypeComboBox.getItems().addAll(trees.keySet());
        treeTypeComboBox.setValue("Binary Search Tree");

        // Input field for values
        inputField = new TextField();
        inputField.setPromptText("Enter integer value");
        inputField.setPrefWidth(200);

        // Action buttons
        insertButton = new Button("Insert");
        deleteButton = new Button("Delete");
        searchButton = new Button("Search");
        clearButton = new Button("Clear");
        inOrderButton = new Button("In-Order Traversal");

        // Style buttons
        insertButton.setStyle("-fx-background-color: lightgray; -fx-text-fill: black;");
        deleteButton.setStyle("-fx-background-color: lightgray; -fx-text-fill: black;");
        searchButton.setStyle("-fx-background-color: lightgray; -fx-text-fill: black;");
        clearButton.setStyle("-fx-background-color: lightgray; -fx-text-fill: black;");
        inOrderButton.setStyle("-fx-background-color: lightgray; -fx-text-fill: black;");

        HBox buttonBox = new HBox(10, insertButton, deleteButton, searchButton, clearButton, inOrderButton);

        // Canvas for tree visualization
        treeCanvas = new Canvas(1000, 675);

        // Output area for messages - IMPROVED
        outputArea = new TextArea();
        outputArea.setEditable(false);
        outputArea.setWrapText(true);
        outputArea.setPrefHeight(150);
        outputArea.setFont(Font.font("Monospaced", 12));
        outputArea.setStyle("-fx-control-inner-background: #f5f5f5; " +
                "-fx-font-family: 'Monospaced'; " +
                "-fx-border-color: #333; " +
                "-fx-border-width: 2;");

        // Add all components to view
        view.getChildren().addAll(
                new HBox(10, new Label("Tree Type:"), treeTypeComboBox),
                new HBox(10, new Label("Value:"), inputField),
                buttonBox,
                treeCanvas,
                new Label("Output:"),
                outputArea
        );
    }

    /**
     * Sets up event handlers for all UI components
     */
    private void setupEventHandlers() {
        insertButton.setOnAction(e -> handleInsert());
        deleteButton.setOnAction(e -> handleDelete());
        searchButton.setOnAction(e -> handleSearch());
        clearButton.setOnAction(e -> handleClear());
        inOrderButton.setOnAction(e -> handleInOrderTraversal());
        treeTypeComboBox.setOnAction(e -> handleTreeTypeChange());
    }

    /**
     * Handles insert button click
     */
    private void handleInsert() {
        try {
            int value = Integer.parseInt(inputField.getText());
            if (!currentTree.contains(value)) {
                currentTree.insert(value);
                highlightedValue = null; // Clear any previous highlight
                updateTreeVisualization();
                outputArea.appendText("✓ Inserted: " + value + "\n");
                inputField.clear();
            } else {
                outputArea.appendText("✗ Value " + value + " already exists in tree\n");
            }
        } catch (NumberFormatException ex) {
            outputArea.appendText("✗ Invalid input - Please enter an integer\n");
        }
    }

    /**
     * Handles delete button click
     */
    private void handleDelete() {
        try {
            int value = Integer.parseInt(inputField.getText());
            boolean deleted = currentTree.delete(value);
            highlightedValue = null; // Clear any previous highlight
            updateTreeVisualization();
            if (deleted) {
                outputArea.appendText("✓ Deleted: " + value + "\n");
            } else {
                outputArea.appendText("✗ Value " + value + " not found in tree\n");
            }
            inputField.clear();
        } catch (NumberFormatException ex) {
            outputArea.appendText("✗ Invalid input - Please enter an integer\n");
        }
    }

    /**
     * Handles search button click with visual highlighting
     */
    private void handleSearch() {
        try {
            int value = Integer.parseInt(inputField.getText());
            boolean found = currentTree.contains(value);

            if (found) {
                highlightedValue = value;
                updateTreeVisualization();
                outputArea.appendText("✓ FOUND: " + value + " (highlighted in YELLOW)\n");
            } else {
                highlightedValue = null;
                updateTreeVisualization();
                outputArea.appendText("✗ NOT FOUND: " + value + "\n");
            }
        } catch (NumberFormatException ex) {
            outputArea.appendText("✗ Invalid input - Please enter an integer\n");
        }
    }

    /**
     * Handles clear button click
     */
    private void handleClear() {
        currentTree.clear();
        highlightedValue = null;
        updateTreeVisualization();
        outputArea.appendText("✓ Tree cleared successfully\n");
    }

    /**
     * Handles in-order traversal button click
     * Shows sorted order of all elements in the tree
     */
    private void handleInOrderTraversal() {
        List<Integer> traversal = currentTree.inorderTraversal();

        outputArea.appendText("\n" + "=".repeat(60) + "\n");
        outputArea.appendText("IN-ORDER TRAVERSAL (Sorted Order)\n");
        outputArea.appendText("=".repeat(60) + "\n");

        if (traversal.isEmpty()) {
            outputArea.appendText("Tree is empty - no elements to traverse\n");
        } else {
            // Format output nicely with line wrapping
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < traversal.size(); i++) {
                sb.append(String.format("%4d", traversal.get(i)));

                if (i < traversal.size() - 1) {
                    sb.append(" -> ");
                }

                // Add line break every 8 elements for readability
                if ((i + 1) % 8 == 0 && i < traversal.size() - 1) {
                    sb.append("\n");
                }
            }
            outputArea.appendText(sb.toString() + "\n");
            outputArea.appendText("-".repeat(60) + "\n");
            outputArea.appendText("Total elements: " + traversal.size() + "\n");
        }
        outputArea.appendText("=".repeat(60) + "\n\n");
    }

    /**
     * Handles tree type change in dropdown
     */
    private void handleTreeTypeChange() {
        String selectedType = treeTypeComboBox.getValue();
        currentTree = trees.get(selectedType);
        highlightedValue = null; // Clear highlight when switching trees
        updateTreeVisualization();
        outputArea.appendText("➜ Switched to " + selectedType + "\n");
    }

    /**
     * Updates the tree visualization on canvas
     */
    private void updateTreeVisualization() {
        GraphicsContext gc = treeCanvas.getGraphicsContext2D();
        gc.clearRect(0, 0, treeCanvas.getWidth(), treeCanvas.getHeight());

        if (currentTree.getRoot() != null) {
            int depth = getTreeDepth(currentTree.getRoot());
            int width = getTreeWidth(currentTree.getRoot());

            double verticalSpacing = (treeCanvas.getHeight()) / (depth + currentTree.size() / 3);
            double horizontalSpacing = treeCanvas.getWidth() / (width + currentTree.size() / 2);

            // Special handling for 2-4 Tree
            if (currentTree.type().equals("2-4")) {
                draw24Tree(gc, (Tree24<Integer>.Node) currentTree.getRoot(), treeCanvas.getWidth() / 2, 40, horizontalSpacing, verticalSpacing, width);
            } else {
                drawNormalTree(gc, currentTree.getRoot(), treeCanvas.getWidth() / 2, 40, horizontalSpacing, verticalSpacing, width);
            }
        }
    }

    /**
     * Draws regular binary trees (BST, AVL, RBT, Heaps) with search highlighting
     */
    private void drawNormalTree(GraphicsContext gc, TreeNode<Integer> node, double x, double y,
                                double hSpacing, double vSpacing, int width) {
        if (node == null) return;

        // Check if this node should be highlighted (search result)
        boolean isHighlighted = highlightedValue != null && node.getValue().equals(highlightedValue);

        // Set node color based on tree type
        if (isHighlighted) {
            // Highlight searched node in YELLOW
            gc.setFill(Color.YELLOW);
        } else if (currentTree.type().equals("RBT") && node.getColor() != null) {
            if (node.getColor().equals("RED")) {
                gc.setFill(Color.RED);
            } else {
                gc.setFill(Color.BLACK);
            }
        } else {
            gc.setFill(currentTree.color());
        }

        // Draw node circle
        gc.fillOval(x - 15, y - 15, 40, 40);

        // Draw border for highlighted node
        if (isHighlighted) {
            gc.setStroke(Color.DARKGREEN);
            gc.setLineWidth(4);
            gc.strokeOval(x - 15, y - 15, 40, 40);
            gc.setLineWidth(1);
        }

        // Draw node value
        if (isHighlighted) {
            gc.setFill(Color.BLACK); // Black text on yellow background
            gc.setFont(Font.font("System", javafx.scene.text.FontWeight.BOLD, 14));
        } else {
            gc.setFill(Color.WHITE); // White text on colored background
            gc.setFont(Font.font("System", 12));
        }

        String valueStr = node.getValue().toString();
        double textWidth = valueStr.length() * 7;
        gc.fillText(valueStr, x - textWidth / 2 + 5, y + 5);
        gc.setFont(Font.font("System", 12)); // Reset font

        // Draw left subtree
        if (node.getLeft() != null) {
            int leftWidth = getTreeWidth(node.getLeft());
            double newX = x - (width - leftWidth / 3) * hSpacing / 3;
            double newY = y + vSpacing;
            gc.setStroke(Color.BLACK);
            gc.setLineWidth(1);
            gc.strokeLine(x + 5, y + 24, newX + 5, newY - 15);
            drawNormalTree(gc, node.getLeft(), newX, newY, hSpacing, vSpacing, leftWidth);
        }

        // Draw right subtree
        if (node.getRight() != null) {
            int rightWidth = getTreeWidth(node.getRight());
            double newX = x + (width - rightWidth / 3) * hSpacing / 3;
            double newY = y + vSpacing;
            gc.setStroke(Color.BLACK);
            gc.setLineWidth(1);
            gc.strokeLine(x + 5, y + 24, newX + 5, newY - 15);
            drawNormalTree(gc, node.getRight(), newX, newY, hSpacing, vSpacing, rightWidth);
        }
    }

    /**
     * Draws 2-4 Tree with multi-value node handling and search highlighting
     */
    private void draw24Tree(GraphicsContext gc, Tree24<Integer>.Node node, double x, double y,
                            double hSpacing, double vSpacing, int width) {
        if (node == null) return;

        // Check if this node contains the highlighted value
        boolean containsHighlighted = false;
        if (highlightedValue != null) {
            for (Integer val : node.values) {
                if (val.equals(highlightedValue)) {
                    containsHighlighted = true;
                    break;
                }
            }
        }

        // Calculate node width based on number of values
        int numValues = node.values.size();
        double nodeWidth = 40 + (numValues - 1) * 30;

        // Draw node rectangle with highlight if needed
        if (containsHighlighted) {
            gc.setFill(Color.YELLOW);
        } else {
            gc.setFill(currentTree.color());
        }
        gc.fillRect(x - nodeWidth / 2, y - 15, nodeWidth, 30);

        // Draw border (thicker for highlighted)
        if (containsHighlighted) {
            gc.setStroke(Color.DARKGREEN);
            gc.setLineWidth(4);
        } else {
            gc.setStroke(Color.BLACK);
            gc.setLineWidth(1);
        }
        gc.strokeRect(x - nodeWidth / 2, y - 15, nodeWidth, 30);
        gc.setLineWidth(1);

        // Draw values
        for (int i = 0; i < node.values.size(); i++) {
            String valueStr = node.values.get(i).toString();
            double textX = x - nodeWidth / 2 + 15 + i * 30;

            // Extra highlight for the specific value in multi-value node
            if (highlightedValue != null && node.values.get(i).equals(highlightedValue)) {
                gc.setFill(Color.RED); // Red text for exact match
                gc.setFont(Font.font("System", javafx.scene.text.FontWeight.BOLD, 14));
                gc.fillText(valueStr, textX, y + 5);
                gc.setFont(Font.font("System", 12));
            } else {
                if (containsHighlighted) {
                    gc.setFill(Color.BLACK); // Black text on yellow
                } else {
                    gc.setFill(Color.WHITE); // White text on purple
                }
                gc.fillText(valueStr, textX, y + 5);
            }
        }

        // Draw children
        if (!node.children.isEmpty()) {
            int numChildren = node.children.size();
            double childSpacing = (width * hSpacing) / (numChildren + 1);

            for (int i = 0; i < numChildren; i++) {
                double childX = x - (width * hSpacing / 2) + (i + 1) * childSpacing;
                double childY = y + vSpacing;

                gc.setStroke(Color.BLACK);
                gc.setLineWidth(1);
                gc.strokeLine(x, y + 15, childX, childY - 15);

                Tree24<Integer>.Node child = node.children.get(i);
                int childWidth = getTreeWidth(child);
                draw24Tree(gc, child, childX, childY, hSpacing * 0.8, vSpacing, childWidth);
            }
        }
    }

    /**
     * Calculates the depth of the tree
     */
    private int getTreeDepth(TreeNode<Integer> node) {
        if (node == null) return 0;
        return 1 + Math.max(getTreeDepth(node.getLeft()), getTreeDepth(node.getRight()));
    }

    /**
     * Calculates the width of the tree
     */
    private int getTreeWidth(TreeNode<Integer> node) {
        if (node == null) return 0;
        if (node.getLeft() == null && node.getRight() == null) return 1;
        return getTreeWidth(node.getLeft()) + getTreeWidth(node.getRight());
    }

    /**
     * Returns the main view
     */
    public VBox getView() {
        return view;
    }

    /**
     * Saves the current tree to a file using Java serialization
     */
    public void saveTree() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Save Tree");
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Tree Files", "*.tree")
        );
        File file = fileChooser.showSaveDialog(stage);

        if (file != null) {
            try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(file))) {
                out.writeObject(currentTree);
                outputArea.appendText("✓ Tree saved to: " + file.getName() + "\n");
                outputArea.appendText("  Type: " + currentTree.type() + " | Size: " + currentTree.size() + "\n");
            } catch (IOException e) {
                outputArea.appendText("✗ Error saving tree: " + e.getMessage() + "\n");
                e.printStackTrace();
            }
        }
    }

    /**
     * Loads a tree from a file using Java deserialization
     */
    public void loadTree() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Load Tree");
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Tree Files", "*.tree")
        );
        File file = fileChooser.showOpenDialog(stage);

        if (file != null) {
            try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(file))) {
                Object loadedObject = in.readObject();

                if (!(loadedObject instanceof Tree)) {
                    outputArea.appendText("✗ Error: File is not a valid Tree\n");
                    return;
                }

                @SuppressWarnings("unchecked")
                Tree<Integer> loadedTree = (Tree<Integer>) loadedObject;

                // Determine tree type and update UI
                String treeType = determineTreeType(loadedTree);
                if (treeType == null) {
                    outputArea.appendText("✗ Error: Unknown tree type\n");
                    return;
                }

                // Update current tree and UI
                treeTypeComboBox.setValue(treeType);
                currentTree = loadedTree;
                trees.put(treeType, loadedTree);
                highlightedValue = null;

                outputArea.appendText("✓ Tree loaded from: " + file.getName() + "\n");
                outputArea.appendText("  Type: " + currentTree.type() + " | Size: " + currentTree.size() + "\n");
                outputArea.appendText("  Contents: " + currentTree.inorderTraversal() + "\n");

                updateTreeVisualization();
            } catch (IOException | ClassNotFoundException e) {
                outputArea.appendText("✗ Error loading tree: " + e.getMessage() + "\n");
                e.printStackTrace();
            }
        }
    }

    /**
     * Determines the type of the loaded tree
     */
    private String determineTreeType(Tree<?> tree) {
        String type = tree.type();
        switch (type) {
            case "BST": return "Binary Search Tree";
            case "AVL": return "AVL Tree";
            case "RBT": return "Red-Black Tree";
            case "MinHeap": return "Min Heap";
            case "MaxHeap": return "Max Heap";
            case "2-4": return "2-4 Tree";
            default: return null;
        }
    }
}