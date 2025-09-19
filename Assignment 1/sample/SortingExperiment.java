import java.util.Arrays;
import java.util.Random;

/**
 * Experimental evaluation class for sorting algorithms
 * Tests algorithms on large arrays without GUI visualization
 */
public class SortingExperiment {

    private int[] originalArray;
    private int[] testArray;

    public SortingExperiment() {
        // Constructor
    }

    /**
     * Generates a random array of specified size
     */
    public void generateRandomArray(int size) {
        originalArray = new int[size];
        Random random = new Random();

        for (int i = 0; i < size; i++) {
            originalArray[i] = random.nextInt(size * 10); // Random numbers up to size*10
        }
    }

    /**
     * Creates a copy of original array for testing
     */
    private void resetTestArray() {
        testArray = Arrays.copyOf(originalArray, originalArray.length);
    }

    /**
     * Bubble Sort implementation for large arrays
     */
    public long bubbleSort() {
        resetTestArray();
        long startTime = System.nanoTime();

        int n = testArray.length;
        for (int i = 0; i < n - 1; i++) {
            boolean swapped = false;
            for (int j = 0; j < n - i - 1; j++) {
                if (testArray[j] > testArray[j + 1]) {
                    // Swap elements
                    int temp = testArray[j];
                    testArray[j] = testArray[j + 1];
                    testArray[j + 1] = temp;
                    swapped = true;
                }
            }
            if (!swapped) break; // if no swaps then array is sorted
        }

        long endTime = System.nanoTime();
        return endTime - startTime;
    }

    /**
     * Selection Sort implementation for large arrays
     */
    public long selectionSort() {
        resetTestArray();
        long startTime = System.nanoTime();

        int n = testArray.length;
        for (int i = 0; i < n - 1; i++) {
            int minIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (testArray[j] < testArray[minIdx]) {
                    minIdx = j;
                }
            }
            // Swap the found minimum element with the first element
            int temp = testArray[minIdx];
            testArray[minIdx] = testArray[i];
            testArray[i] = temp;
        }

        long endTime = System.nanoTime();
        return endTime - startTime;
    }

    /**
     * Insertion Sort implementation for large arrays
     */
    public long insertionSort() {
        resetTestArray();
        long startTime = System.nanoTime();

        int n = testArray.length;
        for (int i = 1; i < n; i++) {
            int key = testArray[i];
            int j = i - 1;

            while (j >= 0 && testArray[j] > key) {
                testArray[j + 1] = testArray[j];
                j = j - 1;
            }
            testArray[j + 1] = key;
        }

        long endTime = System.nanoTime();
        return endTime - startTime;
    }

    /**
     * Shell Sort implementation for large arrays
     */
    public long shellSort() {
        resetTestArray();
        long startTime = System.nanoTime();
        int n = testArray.length;

        // Start with a large gap and reduce it
        for (int gap = n / 2; gap > 0; gap /= 2) {
            for (int i = gap; i < n; i++) {
                int temp = testArray[i];
                int j;
                for (j = i; j >= gap && testArray[j - gap] > temp; j -= gap) {
                    testArray[j] = testArray[j - gap];
                }
                testArray[j] = temp;
            }
        }

        long endTime = System.nanoTime();
        return endTime - startTime;
    }

    /**
     * Merge Sort implementation for large arrays
     */
    public long mergeSort() {
        resetTestArray();
        long startTime = System.nanoTime();
        mergeSortHelper(0, testArray.length - 1);
        long endTime = System.nanoTime();
        return endTime - startTime;
    }

    private void mergeSortHelper(int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2;
            mergeSortHelper(left, mid);
            mergeSortHelper(mid + 1, right);
            merge(left, mid, right);
        }
    }

    private void merge(int left, int mid, int right) {
        int n1 = mid - left + 1;
        int n2 = right - mid;

        int[] leftArray = new int[n1];
        int[] rightArray = new int[n2];

        for (int i = 0; i < n1; i++) {
            leftArray[i] = testArray[left + i];
        }
        for (int j = 0; j < n2; j++) {
            rightArray[j] = testArray[mid + 1 + j];
        }

        int i = 0, j = 0;
        int k = left;

        while (i < n1 && j < n2) {
            if (leftArray[i] <= rightArray[j]) {
                testArray[k] = leftArray[i];
                i++;
            } else {
                testArray[k] = rightArray[j];
                j++;
            }
            k++;
        }

        while (i < n1) {
            testArray[k] = leftArray[i];
            i++;
            k++;
        }

        while (j < n2) {
            testArray[k] = rightArray[j];
            j++;
            k++;
        }
    }

    /**
     * Quick Sort implementation for large arrays
     */
    public long quickSort() {
        resetTestArray();
        long startTime = System.nanoTime();
        quickSortHelper(0, testArray.length - 1);
        long endTime = System.nanoTime();
        return endTime - startTime;
    }

    private void quickSortHelper(int low, int high) {
        if (low < high) {
            int pi = partition(low, high);

            quickSortHelper(low, pi - 1);
            quickSortHelper(pi + 1, high);
        }
    }

    private int partition(int low, int high) {
        int pivot = testArray[high];
        int i = (low - 1);

        for (int j = low; j < high; j++) {
            if (testArray[j] < pivot) {
                i++;

                int temp = testArray[i];
                testArray[i] = testArray[j];
                testArray[j] = temp;
            }
        }

        int temp = testArray[i + 1];
        testArray[i + 1] = testArray[high];
        testArray[high] = temp;
        return i + 1;
    }

    /**
     * Radix Sort implementation for large arrays
     */
    public long radixSort() {
        resetTestArray();
        long startTime = System.nanoTime();
        int max = getMax();

        for (int exp = 1; max / exp > 0; exp *= 10) {
            countingSort(exp);
        }

        long endTime = System.nanoTime();
        return endTime - startTime;
    }

    private int getMax() {
        int max = testArray[0];
        for (int i = 1; i < testArray.length; i++) {
            if (testArray[i] > max) {
                max = testArray[i];
            }
        }
        return max;
    }

    private void countingSort(int exp) {
        int n = testArray.length;
        int[] output = new int[n];
        int[] count = new int[10];

        for (int i = 0; i < n; i++) {
            count[(testArray[i] / exp) % 10]++;
        }

        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }

        for (int i = n - 1; i >= 0; i--) {
            output[count[(testArray[i] / exp) % 10] - 1] = testArray[i];
            count[(testArray[i] / exp) % 10]--;
        }

        for (int i = 0; i < n; i++) {
            testArray[i] = output[i];
        }
    }

    /**
     * Runs comprehensive performance tests
     */
    public static void main(String[] args) {
        SortingExperiment experiment = new SortingExperiment();
        int[] sizes = {1000, 10000, 100000};

        System.out.println("Sorting Algorithm Performance Analysis");
        System.out.println("=====================================");
        System.out.println();

        for (int size : sizes) {
            System.out.println("Testing with array size: " + size);
            System.out.println("-".repeat(40));
            experiment.generateRandomArray(size);

            // Test each sorting algorithm
            long bubbleTime = experiment.bubbleSort();
            long selectionTime = experiment.selectionSort();
            long insertionTime = experiment.insertionSort();
            long shellTime = experiment.shellSort();
            long mergeTime = experiment.mergeSort();
            long quickTime = experiment.quickSort();
            long radixTime = experiment.radixSort();

            // Convert nanoseconds to milliseconds
            System.out.printf("Bubble Sort:    %8.2f ms%n", bubbleTime / 1_000_000.0);
            System.out.printf("Selection Sort: %8.2f ms%n", selectionTime / 1_000_000.0);
            System.out.printf("Insertion Sort: %8.2f ms%n", insertionTime / 1_000_000.0);
            System.out.printf("Shell Sort:     %8.2f ms%n", shellTime / 1_000_000.0);
            System.out.printf("Merge Sort:     %8.2f ms%n", mergeTime / 1_000_000.0);
            System.out.printf("Quick Sort:     %8.2f ms%n", quickTime / 1_000_000.0);
            System.out.printf("Radix Sort:     %8.2f ms%n", radixTime / 1_000_000.0);
            System.out.println();
        }

        System.out.println("Performance testing completed!");
        System.out.println("Use these results for your comparative analysis report.");
    }
}