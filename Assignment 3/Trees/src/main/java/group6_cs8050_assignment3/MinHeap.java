package group6_cs8050_assignment3;

import javafx.scene.paint.Color;

/**
 * Min Heap implementation - a complete binary tree where parent is smaller than children
 * Stored as an array for efficient operations
 * Author: Group 6
 */
public class MinHeap<T extends Comparable<T>> extends Heap<T> {

    /**
     * Returns the tree type identifier
     */
    @Override
    public String type() {
        return "MinHeap";
    }

    /**
     * Returns the color used for visualization (Green for Min Heap)
     */
    @Override
    public Color color() {
        return Color.GREEN;
    }

    /**
     * Maintains min heap property by bubbling up
     * Called after insertion to restore heap order
     * Time Complexity: O(log n)
     */
    @Override
    protected void heapifyUp(int index) {
        while (index > 0) {
            int parentIndex = getParentIndex(index);

            // If current element is smaller than parent, swap
            if (heap.get(index).compareTo(heap.get(parentIndex)) < 0) {
                swap(index, parentIndex);
                index = parentIndex;
            } else {
                break;
            }
        }
    }

    /**
     * Maintains min heap property by bubbling down
     * Called after deletion to restore heap order
     * Time Complexity: O(log n)
     */
    @Override
    protected void heapifyDown(int index) {
        int size = heap.size();

        while (true) {
            int smallest = index;
            int leftIndex = getLeftChildIndex(index);
            int rightIndex = getRightChildIndex(index);

            // Compare with left child
            if (leftIndex < size && heap.get(leftIndex).compareTo(heap.get(smallest)) < 0) {
                smallest = leftIndex;
            }

            // Compare with right child
            if (rightIndex < size && heap.get(rightIndex).compareTo(heap.get(smallest)) < 0) {
                smallest = rightIndex;
            }

            // If smallest is not current index, swap and continue
            if (smallest != index) {
                swap(index, smallest);
                index = smallest;
            } else {
                break;
            }
        }
    }
}