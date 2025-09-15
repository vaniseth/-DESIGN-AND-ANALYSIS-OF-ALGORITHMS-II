/**
 *
 * @author Ouda
 */

//importing the libraries that will be needed in this program

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

//the class with button and main method
public class SortGUI {

	// Timing variables for all sorting algorithms
	public static double selectionTime = 0.0;
	public static double rmergeTime = 0.0;
	public static double imergeTime = 0.0;
	public static double bubbleTime = 0.0;
	public static double insertionTime = 0.0;
	public static double shellTime = 0.0;
	public static double quickTime = 0.0;
	public static double radixTime = 0.0;
	
	// Boolean variables to track which sorts have been completed
	public boolean Selection_Done = false;
	public boolean Recersive_Merge_Done = false;
	public boolean Iterative_Merge_Done = false;
	public boolean Bubble_Done = false;
	public boolean Insertion_Done = false;
	public boolean Shell_Done = false;
	public boolean Quick_Done = false;
	public boolean Radix_Done = false;
	
	//Making a object from the class SortShow
	SortShow sortArea = new SortShow();
	
	//Default constructor for SortGUI
	public SortGUI() {
		//making a MyScreen object
		MyScreen screen = new MyScreen();
		//Setting a title to the GUI window
		screen.setTitle("Assignment-1 Sorting Algorithms by Vani and Preya");
		//setting the size of the window 
		screen.setSize(1200, 600);
		//the operation when the frame is closed
		screen.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		//is set to true to display the frame
		screen.setVisible(true);
	}
	
	//A public class that extends JFrame
	public class MyScreen extends JFrame {
		// Buttons for all sorting algorithms
		JButton scramble_button = new JButton("Scramble Lines");
		JRadioButton selection = new JRadioButton("Selection Sort");
		JRadioButton rmerge = new JRadioButton("Merge Recursive");
		JRadioButton imerge = new JRadioButton("Merge Iterative");
		JRadioButton bubble = new JRadioButton("Bubble Sort");
		JRadioButton insertion = new JRadioButton("Insertion Sort");
		JRadioButton shell = new JRadioButton("Shell Sort");
		JRadioButton quick = new JRadioButton("Quick Sort");
		JRadioButton radix = new JRadioButton("Radix Sort");
		JRadioButton reset = new JRadioButton("Reset");
		
		// Labels for timing display
		JLabel selection_time_label = new JLabel("Selection Time:");
		JLabel selection_time_taken = new JLabel(""); 
		JLabel rmerge_time_label = new JLabel("Merge-Rec Time:");
		JLabel rmerge_time_taken = new JLabel("");
		JLabel imerge_time_label = new JLabel("Merge-Iter Time:");
		JLabel imerge_time_taken = new JLabel("");
		JLabel bubble_time_label = new JLabel("Bubble Time:");
		JLabel bubble_time_taken = new JLabel("");
		JLabel insertion_time_label = new JLabel("Insertion Time:");
		JLabel insertion_time_taken = new JLabel("");
		JLabel shell_time_label = new JLabel("Shell Time:");
		JLabel shell_time_taken = new JLabel("");
		JLabel quick_time_label = new JLabel("Quick Time:");
		JLabel quick_time_taken = new JLabel("");
		JLabel radix_time_label = new JLabel("Radix Time:");
		JLabel radix_time_taken = new JLabel("");
	
		//the default constructor for the class MyScreen
		public MyScreen() {
			// Set colors for time labels
			selection_time_taken.setForeground(Color.RED);
			rmerge_time_taken.setForeground(Color.RED);
			imerge_time_taken.setForeground(Color.RED);
			bubble_time_taken.setForeground(Color.RED);
			insertion_time_taken.setForeground(Color.RED);
			shell_time_taken.setForeground(Color.RED);
			quick_time_taken.setForeground(Color.RED);
			radix_time_taken.setForeground(Color.RED);
			
			// Set colors for radio buttons
			selection.setForeground(Color.BLUE);
			rmerge.setForeground(Color.BLUE);
			imerge.setForeground(Color.BLUE);
			bubble.setForeground(Color.BLUE);
			insertion.setForeground(Color.BLUE);
			shell.setForeground(Color.BLUE);
			quick.setForeground(Color.BLUE);
			radix.setForeground(Color.BLUE);
			scramble_button.setForeground(Color.BLUE);
			
			// Set font for scramble button
			scramble_button.setFont(new Font("Arial", Font.BOLD, 15));
			
			// Create panels for organization
			JPanel radio_button_selection_Panel = new JPanel(new GridLayout(9, 1, 3, 3));
			radio_button_selection_Panel.add(selection);
			radio_button_selection_Panel.add(bubble);
			radio_button_selection_Panel.add(insertion);
			radio_button_selection_Panel.add(shell);
			radio_button_selection_Panel.add(rmerge);
			radio_button_selection_Panel.add(imerge);
			radio_button_selection_Panel.add(quick);
			radio_button_selection_Panel.add(radix);
			radio_button_selection_Panel.add(reset);
			radio_button_selection_Panel.setBorder(new javax.swing.border.TitledBorder("Sort Algorithms"));

			// Time display panel
			JPanel time_Panel = new JPanel(new GridLayout(16, 1, 2, 2));
			time_Panel.add(selection_time_label);
			time_Panel.add(selection_time_taken);
			time_Panel.add(bubble_time_label);
			time_Panel.add(bubble_time_taken);
			time_Panel.add(insertion_time_label);
			time_Panel.add(insertion_time_taken);
			time_Panel.add(shell_time_label);
			time_Panel.add(shell_time_taken);
			time_Panel.add(rmerge_time_label);
			time_Panel.add(rmerge_time_taken);
			time_Panel.add(imerge_time_label);
			time_Panel.add(imerge_time_taken);
			time_Panel.add(quick_time_label);
			time_Panel.add(quick_time_taken);
			time_Panel.add(radix_time_label);
			time_Panel.add(radix_time_taken);

			// Buttons area panel
			JPanel buttons_area_Panel = new JPanel(new BorderLayout());
			buttons_area_Panel.add(scramble_button, BorderLayout.NORTH);
			buttons_area_Panel.add(radio_button_selection_Panel, BorderLayout.CENTER);
			buttons_area_Panel.add(time_Panel, BorderLayout.SOUTH);

			// Add to main frame
			add(buttons_area_Panel, BorderLayout.EAST);
			add(sortArea, BorderLayout.CENTER);
			
			// Initially disable all sorting options
			setAllSortingEnabled(false);
			reset.setEnabled(false);

			// Add action listeners
			setupActionListeners();
		}
		
		private void setupActionListeners() {
			// Scramble button listener
			scramble_button.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					sortArea.scramble_the_lines(); 
					scramble_button.setEnabled(false); 
					setAllSortingEnabled(true);
					reset.setEnabled(false);
					resetAllCompletionFlags();
					clearAllTimings();
				}
			});

			// Selection sort listener
			selection.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					sortArea.SelectionSort(); 
					Selection_Done = true;
					selection_time_taken.setText(selectionTime / 1000 + " Seconds");
					setAllSortingEnabled(false);
					reset.setEnabled(true);
				}
			});

			// Bubble sort listener
			bubble.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					sortArea.BubbleSort();
					Bubble_Done = true;
					bubble_time_taken.setText(bubbleTime / 1000 + " Seconds");
					setAllSortingEnabled(false);
					reset.setEnabled(true);
				}
			});
			
			// Insertion sort listener
			insertion.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					sortArea.InsertionSort();
					Insertion_Done = true;
					insertion_time_taken.setText(insertionTime / 1000 + " Seconds");
					setAllSortingEnabled(false);
					reset.setEnabled(true);
				}
			});
			
			// Shell sort listener
			shell.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					sortArea.ShellSort();
					Shell_Done = true;
					shell_time_taken.setText(shellTime / 1000 + " Seconds");
					setAllSortingEnabled(false);
					reset.setEnabled(true);
				}
			});

			// Recursive merge sort listener
			rmerge.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					sortArea.R_MergeSort();
					rmerge_time_taken.setText((rmergeTime / 1000) + " Seconds");
					Recersive_Merge_Done = true;
					setAllSortingEnabled(false);
					reset.setEnabled(true);
				}
			});
			
			// Iterative merge sort listener
			imerge.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					sortArea.I_MergeSort();
					imerge_time_taken.setText((imergeTime / 1000) + " Seconds");
					Iterative_Merge_Done = true;
					setAllSortingEnabled(false);
					reset.setEnabled(true);
				}
			});
			
			// Quick sort listener
			quick.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					sortArea.QuickSort();
					Quick_Done = true;
					quick_time_taken.setText(quickTime / 1000 + " Seconds");
					setAllSortingEnabled(false);
					reset.setEnabled(true);
				}
			});
			
			// Radix sort listener
			radix.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					sortArea.RadixSort();
					Radix_Done = true;
					radix_time_taken.setText(radixTime / 1000 + " Seconds");
					setAllSortingEnabled(false);
					reset.setEnabled(true);
				}
			});

			// Reset button listener
			reset.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					reset.setEnabled(false);
					sortArea.reset();
					
					// Check if all sorts have been completed
					if (allSortsCompleted()) {
						scramble_button.setEnabled(true);
						resetAllCompletionFlags();
						setAllSortingEnabled(false);
						clearAllTimings();
					} else {
						// Enable only the sorts that haven't been completed yet
						enableUncompletedSorts();
					}
				}
			});
		}
		
		// Helper method to enable/disable all sorting options
		private void setAllSortingEnabled(boolean enabled) {
			selection.setEnabled(enabled);
			bubble.setEnabled(enabled);
			insertion.setEnabled(enabled);
			shell.setEnabled(enabled);
			rmerge.setEnabled(enabled);
			imerge.setEnabled(enabled);
			quick.setEnabled(enabled);
			radix.setEnabled(enabled);
		}
		
		// Helper method to reset all completion flags
		private void resetAllCompletionFlags() {
			Selection_Done = false;
			Recersive_Merge_Done = false;
			Iterative_Merge_Done = false;
			Bubble_Done = false;
			Insertion_Done = false;
			Shell_Done = false;
			Quick_Done = false;
			Radix_Done = false;
		}
		
		// Helper method to clear all timing displays
		private void clearAllTimings() {
			selection_time_taken.setText("");
			rmerge_time_taken.setText("");
			imerge_time_taken.setText("");
			bubble_time_taken.setText("");
			insertion_time_taken.setText("");
			shell_time_taken.setText("");
			quick_time_taken.setText("");
			radix_time_taken.setText("");
		}
		
		// Helper method to check if all sorts have been completed
		private boolean allSortsCompleted() {
			return Selection_Done && Recersive_Merge_Done && Iterative_Merge_Done && 
				   Bubble_Done && Insertion_Done && Shell_Done && Quick_Done && Radix_Done;
		}
		
		// Helper method to enable only uncompleted sorts
		private void enableUncompletedSorts() {
			selection.setEnabled(!Selection_Done);
			bubble.setEnabled(!Bubble_Done);
			insertion.setEnabled(!Insertion_Done);
			shell.setEnabled(!Shell_Done);
			rmerge.setEnabled(!Recersive_Merge_Done);
			imerge.setEnabled(!Iterative_Merge_Done);
			quick.setEnabled(!Quick_Done);
			radix.setEnabled(!Radix_Done);
		}
	}

	//The main method
	public static void main(String[] args) {
		//initialize the class
		SortGUI sort_GUI = new SortGUI();
	}
}