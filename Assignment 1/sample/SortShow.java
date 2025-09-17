/**
 *
 * @author Ouda
 */

//importing the libraries that will be needed in this program

import javax.swing.*;
import java.awt.*;
import java.util.Calendar;
import java.util.Random;

//The class that has all the sorts in it
public class SortShow extends JPanel { 

	
		// An array to hold the lines_lengths to be sorted
		public int[] lines_lengths;
		//The amount of lines needed
		public final int total_number_of_lines = 256;
		 // An array to holds the scrambled lines_lengths
		public int[] scramble_lines;
		//A temp Array that is used later for sorts
		public int[] tempArray;
		
		//the default constructor for the SortShow class
		public SortShow(){
			//assigning the size for the lines_lengths below
			lines_lengths = new int[total_number_of_lines];
			for(int i = 0; i < total_number_of_lines; i++) 
				lines_lengths[i] =  i+5;
			
		}
		

		//A method that scrambles the lines
		public void scramble_the_lines(){
			//A random generator
			Random num = new Random(); 
			//Randomly switching the lines
			for(int i = 0; i < total_number_of_lines; i++){
				//getting a random number using the nextInt method (a number between 0 to i + 1)
				int j = num.nextInt(i + 1); 
				//swapping The element at i and j 
				swap(i, j);
			}
			//assigning the size for the scramble_lines below
			scramble_lines = new int[total_number_of_lines];
			//copying the now scrambled lines_lengths array into the scramble_lines array 
			//to store for reuse for other sort methods
			//so that all sort methods will use the same scrambled lines for fair comparison 
			for (int i = 0; i < total_number_of_lines; i++)
			{
				scramble_lines[i] = lines_lengths[i];
			}
			//Drawing the now scrambled lines_lengths
			paintComponent(this.getGraphics());
		}
		
		//Swapping method that swaps two elements in the lines_lengths array
		public void swap(int i, int j){
			//storing the i element in lines_lengths in temp
			int temp = lines_lengths[i];
			//giving i element in lines_lengths the value of j element in lines_lengths
			lines_lengths[i] = lines_lengths[j];
			//giving j element in lines_lengths the value of temp
			lines_lengths[j] = temp;
		}
		
		//The selectionSort method
		public void SelectionSort(){
			//getting the date and time when the selection sort starts
			Calendar start = Calendar.getInstance();
			//Using the selection sort to lines_lengths sort the array

			// Selection Sort Implementation
			for (int i = 0; i < total_number_of_lines - 1; i++) {
				// Find the index of the smallest element in the unsorted portion
				int minIndex = getIndexOfSmallest(i, total_number_of_lines - 1);
				
				// Swap the smallest element with the first element of unsorted portion
				if (minIndex != i) {
					swap(i, minIndex);
					// Redraw the visualization
					paintComponent(this.getGraphics());
					// Add delay for visualization
					delay(10);
				}
			}

			//getting the date and time when the selection sort ends
			Calendar end = Calendar.getInstance();
			//getting the time it took for the selection sort to execute 
			//subtracting the end time with the start time
	        SortGUI.selectionTime = end.getTime().getTime() - start.getTime().getTime();
		}
		
		//this method gets the smallest element in the array of lines_lengths
		public int getIndexOfSmallest(int first, int last){

			// Find the index of the smallest element between first and last
			int minIndex = first;
			for (int i = first + 1; i <= last; i++) {
				if (lines_lengths[i] < lines_lengths[minIndex]) {
					minIndex = i;
				}
			}
			return minIndex;
		}
		
	///////////////////////////////////////////////////////////////////////////////////
		
		//recursive merge sort method
		public void R_MergeSort(){
			//getting the date and time when the recursive merge sort starts
			Calendar start = Calendar.getInstance();
			//assigning the size for the tempArray below

			// Initialize temp array for merge operations
			tempArray = new int[total_number_of_lines];
			
			// Call recursive merge sort
			R_MergeSort(0, total_number_of_lines - 1);

			Calendar end = Calendar.getInstance();
			//getting the time it took for the iterative merge sort to execute
			//subtracting the end time with the start time
	        SortGUI.rmergeTime = end.getTime().getTime() - start.getTime().getTime();
			
		}
		
		//recursive merge sort method
		public void R_MergeSort(int first, int last){
			if(first < last){

				// Find the middle point
				int mid = first + (last - first) / 2;
				
				// Recursively sort first and second halves
				R_MergeSort(first, mid);
				R_MergeSort(mid + 1, last);
				
				// Merge the sorted halves
				R_Merge(first, mid, last);
				
				// Redraw for visualization
				paintComponent(this.getGraphics());
				//Causing a delay for 10ms
				delay(10);
			}
		}

		
		//recursive merge sort method
		public void R_Merge(int first, int mid, int last){

			// Copy data to temp arrays
			int leftSize = mid - first + 1;
			int rightSize = last - mid;
			
			int[] leftArray = new int[leftSize];
			int[] rightArray = new int[rightSize];
			
			// Copy data to temp arrays
			for (int i = 0; i < leftSize; i++) {
				leftArray[i] = lines_lengths[first + i];
			}
			for (int j = 0; j < rightSize; j++) {
				rightArray[j] = lines_lengths[mid + 1 + j];
			}
			
			// Merge the temp arrays back into lines_lengths[first..last]
			int i = 0, j = 0; // Initial indexes of first and second subarrays
			int k = first; // Initial index of merged subarray
			
			while (i < leftSize && j < rightSize) {
				if (leftArray[i] <= rightArray[j]) {
					lines_lengths[k] = leftArray[i];
					i++;
				} else {
					lines_lengths[k] = rightArray[j];
					j++;
				}
				k++;
			}
			
			// Copy the remaining elements of leftArray[], if any
			while (i < leftSize) {
				lines_lengths[k] = leftArray[i];
				i++;
				k++;
			}
			
			// Copy the remaining elements of rightArray[], if any
			while (j < rightSize) {
				lines_lengths[k] = rightArray[j];
				j++;
				k++;
			}
				
		}
		
		// Bubble Sort Implementation
		public void BubbleSort() {
			Calendar start = Calendar.getInstance();
			
			for (int i = 0; i < total_number_of_lines - 1; i++) {
				boolean swapped = false;
				for (int j = 0; j < total_number_of_lines - i - 1; j++) {
					if (lines_lengths[j] > lines_lengths[j + 1]) {
						swap(j, j + 1);
						swapped = true;
						paintComponent(this.getGraphics());
						delay(1); // Faster delay for bubble sort
					}
				}
				if (!swapped) break; // Optimization: if no swaps, array is sorted
			}
			
			Calendar end = Calendar.getInstance();
			SortGUI.bubbleTime = end.getTime().getTime() - start.getTime().getTime();
		}
		
		// Insertion Sort Implementation
		public void InsertionSort() {
			Calendar start = Calendar.getInstance();
			
			for (int i = 1; i < total_number_of_lines; i++) {
				int key = lines_lengths[i];
				int j = i - 1;
				
				// Move elements that are greater than key one position ahead
				while (j >= 0 && lines_lengths[j] > key) {
					lines_lengths[j + 1] = lines_lengths[j];
					j--;
					paintComponent(this.getGraphics());
					delay(5);
				}
				lines_lengths[j + 1] = key;
				paintComponent(this.getGraphics());
				delay(5);
			}
			
			Calendar end = Calendar.getInstance();
			SortGUI.insertionTime = end.getTime().getTime() - start.getTime().getTime();
		}
		
		// Shell Sort Implementation
		public void ShellSort() {
			Calendar start = Calendar.getInstance();
			
			// Start with a large gap and reduce it
			for (int gap = total_number_of_lines / 2; gap > 0; gap /= 2) {
				// Perform gapped insertion sort
				for (int i = gap; i < total_number_of_lines; i++) {
					int temp = lines_lengths[i];
					int j;
					
					for (j = i; j >= gap && lines_lengths[j - gap] > temp; j -= gap) {
						lines_lengths[j] = lines_lengths[j - gap];
						paintComponent(this.getGraphics());
						delay(5);
					}
					lines_lengths[j] = temp;
				}
			}
			
			Calendar end = Calendar.getInstance();
			SortGUI.shellTime = end.getTime().getTime() - start.getTime().getTime();
		}
		
		// Quick Sort Implementation
		public void QuickSort() {
			Calendar start = Calendar.getInstance();
			quickSortHelper(0, total_number_of_lines - 1);
			Calendar end = Calendar.getInstance();
			SortGUI.quickTime = end.getTime().getTime() - start.getTime().getTime();
		}
		
		private void quickSortHelper(int low, int high) {
			if (low < high) {
				int pi = partition(low, high);
				quickSortHelper(low, pi - 1);
				quickSortHelper(pi + 1, high);
			}
		}
		
		private int partition(int low, int high) {
			int pivot = lines_lengths[high];
			int i = low - 1;
			
			for (int j = low; j < high; j++) {
				if (lines_lengths[j] < pivot) {
					i++;
					swap(i, j);
					paintComponent(this.getGraphics());
					delay(10);
				}
			}
			swap(i + 1, high);
			paintComponent(this.getGraphics());
			delay(10);
			return i + 1;
		}
		
		// Radix Sort Implementation
		public void RadixSort() {
			Calendar start = Calendar.getInstance();
			
			// Find the maximum number to know number of digits
			int max = getMax();
			
			// Do counting sort for every digit
			for (int exp = 1; max / exp > 0; exp *= 10) {
				countingSort(exp);
			}
			
			Calendar end = Calendar.getInstance();
			SortGUI.radixTime = end.getTime().getTime() - start.getTime().getTime();
		}
		
		private int getMax() {
			int max = lines_lengths[0];
			for (int i = 1; i < total_number_of_lines; i++) {
				if (lines_lengths[i] > max) {
					max = lines_lengths[i];
				}
			}
			return max;
		}
		
		private void countingSort(int exp) {
			int[] output = new int[total_number_of_lines];
			int[] count = new int[10];
			
			// Store count of occurrences in count[]
			for (int i = 0; i < total_number_of_lines; i++) {
				count[(lines_lengths[i] / exp) % 10]++;
			}
			
			// Change count[i] so that count[i] contains actual position
			for (int i = 1; i < 10; i++) {
				count[i] += count[i - 1];
			}
			
			// Build the output array
			for (int i = total_number_of_lines - 1; i >= 0; i--) {
				output[count[(lines_lengths[i] / exp) % 10] - 1] = lines_lengths[i];
				count[(lines_lengths[i] / exp) % 10]--;
			}
			
			// Copy the output array to lines_lengths[]
			for (int i = 0; i < total_number_of_lines; i++) {
				lines_lengths[i] = output[i];
			}
			
			paintComponent(this.getGraphics());
			delay(50);
		}

	//////////////////////////////////////////////////////////////////////////////////////////
		
		//iterative merge sort method
		public void I_MergeSort()
		{
		//getting the date and time when the iterative merge sort starts
		Calendar start = Calendar.getInstance();
		//assigning the size for the tempArray below
		tempArray = new int[total_number_of_lines]; 
		//saving the value of total_number_of_lines
		int beginLeftovers = total_number_of_lines;

		
		for (int segmentLength = 1; segmentLength <= total_number_of_lines/2; segmentLength = 2*segmentLength)
		{
			beginLeftovers = I_MergeSegmentPairs(total_number_of_lines, segmentLength);
			int endSegment = beginLeftovers + segmentLength - 1;
			if (endSegment < total_number_of_lines - 1) 
			{
			I_Merge(beginLeftovers, endSegment, total_number_of_lines - 1);
			}
		} 

		// merge the sorted leftovers with the rest of the sorted array
		if (beginLeftovers < total_number_of_lines) {
			I_Merge(0, beginLeftovers-1, total_number_of_lines - 1);
		}
		//getting the date and time when the iterative merge sort ends
		Calendar end = Calendar.getInstance();
		//getting the time it took for the iterative merge sort to execute 
		//subtracting the end time with the start time
	    SortGUI.imergeTime = end.getTime().getTime() - start.getTime().getTime();
	} 

	// Merges segments pairs (certain length) within an array 
	public int I_MergeSegmentPairs(int l, int segmentLength)
	{
		//The length of the two merged segments 

		//You suppose  to complete this part (Given).
		int mergedPairLength = 2 * segmentLength;
		int numberOfPairs = l / mergedPairLength;

		int beginSegment1 = 0;
		for (int count = 1; count <= numberOfPairs; count++)
		{
			int endSegment1 = beginSegment1 + segmentLength - 1;

			int beginSegment2 = endSegment1 + 1;
			int endSegment2 = beginSegment2 + segmentLength - 1;
			I_Merge(beginSegment1, endSegment1, endSegment2);

			beginSegment1 = endSegment2 + 1;
			//redrawing the lines_lengths
			paintComponent(this.getGraphics());
			//Causing a delay for 10ms
			delay(10);
		}
		// Returns index of last merged pair
		return beginSegment1;
		//return 1;//modify this line
	}

	public void I_Merge(int first, int mid, int last)
	{
		//You suppose  to complete this part (Given).
		// Two adjacent sub-arrays
		int beginHalf1 = first;
		int endHalf1 = mid;
		int beginHalf2 = mid + 1;
		int endHalf2 = last;

		// While both sub-arrays are not empty, copy the
		// smaller item into the temporary array
		int index = beginHalf1; // Next available location in tempArray
		for (; (beginHalf1 <= endHalf1) && (beginHalf2 <= endHalf2); index++)
		{
			// Invariant: tempArray[beginHalf1..index-1] is in order
			if (lines_lengths[beginHalf1] < lines_lengths[beginHalf2])
			{
				tempArray[index] = lines_lengths[beginHalf1];
				beginHalf1++;
			}
			else
			{
				tempArray[index] = lines_lengths[beginHalf2];
				beginHalf2++;
			}
		}
		//redrawing the lines_lengths
		//paintComponent(this.getGraphics());

		// Finish off the nonempty sub-array

		// Finish off the first sub-array, if necessary
		for (; beginHalf1 <= endHalf1; beginHalf1++, index++)
			// Invariant: tempArray[beginHalf1..index-1] is in order
			tempArray[index] = lines_lengths[beginHalf1];

		// Finish off the second sub-array, if necessary
		for (; beginHalf2 <= endHalf2; beginHalf2++, index++)
			// Invariant: tempa[beginHalf1..index-1] is in order
			tempArray[index] = lines_lengths[beginHalf2];

		// Copy the result back into the original array
		for (index = first; index <= last; index++)
			lines_lengths[index] = tempArray[index];
	}

	//////////////////////////////////////////////////////////////////////	
		
		//This method resets the window to the scrambled lines display
		public void reset(){
			if(scramble_lines != null)
			{
				//copying the old scrambled lines into lines_lengths
				for (int i = 0; i < total_number_of_lines; i++)
				{
					lines_lengths[i] = scramble_lines[i] ;
				}
			//Drawing the now scrambled lines_lengths
			paintComponent(this.getGraphics());
		}
			}
		
	
		//This method colours the lines and prints the lines
		public void paintComponent(Graphics g){
 			super.paintComponent(g);
			//A loop to assign a colour to each line
			for(int i = 0; i < total_number_of_lines; i++){
				//using eight colours for the lines
				if(i % 8 == 0){
					g.setColor(Color.green);
				} else if(i % 8 == 1){
					g.setColor(Color.blue);
				} else if(i % 8 == 2){
					g.setColor(Color.yellow);
				} else if(i%8 == 3){
					g.setColor(Color.red);
				} else if(i%8 == 4){
					g.setColor(Color.black);
				} else if(i%8 == 5){
					g.setColor(Color.orange);
				} else if(i%8 == 6){
					g.setColor(Color.magenta);
				} else
					g.setColor(Color.gray);
				
				//Drawing the lines using the x and y-components 
				g.drawLine(4*i + 25, 300, 4*i + 25, 300 - lines_lengths[i]);
			}
			
		}
		
		//A delay method that pauses the execution for the milliseconds time given as a parameter
		public void delay(int time){
			try{
	        	Thread.sleep(time);
	        }catch(InterruptedException ie){
	        	Thread.currentThread().interrupt();
	        }
		}
		
	}