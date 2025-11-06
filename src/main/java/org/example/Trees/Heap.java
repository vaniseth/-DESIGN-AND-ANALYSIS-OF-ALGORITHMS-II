package org.example.Trees;
//Auther: Abdelnasser Ouda
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public abstract class Heap<T extends Comparable<T>> implements Tree<T> , Serializable {
    protected ArrayList<T> heap;

    public Heap() {
        heap = new ArrayList<>();
    }

    @Override
    public void insert(T value) {
        heap.add(value);
        heapifyUp(heap.size() - 1);
    }

    @Override
    public boolean delete(T value) {
        int index = heap.indexOf(value);
        if (index == -1) return false;

        int lastIndex = heap.size() - 1;
        swap(index, lastIndex);
        heap.remove(lastIndex);

        if (index < heap.size()) {
            heapifyDown(index);
        }

        return true;
    }

    @Override
    public boolean contains(T value) {
        return heap.contains(value);
    }

    @Override
    public void clear() {
        heap.clear();
    }

    @Override
    public int size() {
        return heap.size();
    }

    @Override
    public List<T> inorderTraversal() {
        return new ArrayList<>(heap);
    }

    @Override
    public TreeNode<T> getRoot() {
        return heap.isEmpty() ? null : new HeapNode(0);
    }

    protected abstract void heapifyUp(int index);
    protected abstract void heapifyDown(int index);

    protected void swap(int i, int j) {
        T temp = heap.get(i);
        heap.set(i, heap.get(j));
        heap.set(j, temp);
    }

    protected int getParentIndex(int i) {
        return (i - 1) / 2;
    }

    protected int getLeftChildIndex(int i) {
        return 2 * i + 1;
    }

    protected int getRightChildIndex(int i) {
        return 2 * i + 2;
    }

    private class HeapNode implements TreeNode<T> {
        private int index;

        HeapNode(int index) {
            this.index = index;
        }

        @Override
        public T getValue() {
            return heap.get(index);
        }

        @Override
        public TreeNode<T> getLeft() {
            int leftIndex = getLeftChildIndex(index);
            return leftIndex < heap.size() ? new HeapNode(leftIndex) : null;
        }

        @Override
        public TreeNode<T> getRight() {
            int rightIndex = getRightChildIndex(index);
            return rightIndex < heap.size() ? new HeapNode(rightIndex) : null;
        }

        public String getColor() { return "null"; }
    }
}