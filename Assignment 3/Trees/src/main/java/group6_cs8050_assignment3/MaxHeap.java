package group6_cs8050_assignment3;

import javafx.scene.paint.Color;

/**
 * Max Heap implementation - a complete binary tree where parent is larger than children
 * Stored as an array for efficient operations
 * Author: Group 6
 */
public class MaxHeap<T extends Comparable<T>> extends Heap<T> {

    /**
     * Returns the tree type identifier
     */
    @Override
    public String type() {
        return "MaxHeap";
    }

    /**
     * Returns the color used for visualization (Orange for Max Heap)
     */
    @Override
    public Color color() {
        return Color.ORANGE;
    }

    /**
     * Maintains max heap property by bubbling up
     * Called after insertion to restore heap order
     * Time Complexity: O(log n)
     */
    @Override
    protected void heapifyUp(int index) {
        while (index > 0) {
            int parentIndex = getParentIndex(index);

            // If current element is larger than parent, swap
            if (heap.get(index).compareTo(heap.get(parentIndex)) > 0) {
                swap(index, parentIndex);
                index = parentIndex;
            } else {
                break;
            }
        }
    }

    /**
     * Maintains max heap property by bubbling down
     * Called after deletion to restore heap order
     * Time Complexity: O(log n)
     */
    @Override
    protected void heapifyDown(int index) {
        int size = heap.size();

        while (true) {
            int largest = index;
            int leftIndex = getLeftChildIndex(index);
            int rightIndex = getRightChildIndex(index);

            // Compare with left child
            if (leftIndex < size && heap.get(leftIndex).compareTo(heap.get(largest)) > 0) {
                largest = leftIndex;
            }

            // Compare with right child
            if (rightIndex < size && heap.get(rightIndex).compareTo(heap.get(largest)) > 0) {
                largest = rightIndex;
            }

            // If largest is not current index, swap and continue
            if (largest != index) {
                swap(index, largest);
                index = largest;
            } else {
                break;
            }
        }
    }
}