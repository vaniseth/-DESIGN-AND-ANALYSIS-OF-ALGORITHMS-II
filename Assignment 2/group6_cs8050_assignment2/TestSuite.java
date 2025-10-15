package group6_cs8050_assignment2;

import java.util.*;

/**
 * Comprehensive Test Suite for Dictionary Implementations
 * Validates correctness and performance
 */
public class TestSuite {

    private static int testsPassed = 0;
    private static int testsFailed = 0;

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════╗");
        System.out.println("║   Dictionary Implementation Test Suite              ║");
        System.out.println("╚══════════════════════════════════════════════════════╝\n");

        runAllTests();

        System.out.println("\n╔══════════════════════════════════════════════════════╗");
        System.out.println("║   Test Results                                       ║");
        System.out.println("╚══════════════════════════════════════════════════════╝");
        System.out.println("Tests Passed: " + testsPassed);
        System.out.println("Tests Failed: " + testsFailed);
        System.out.println("Success Rate: " +
                String.format("%.2f%%", 100.0 * testsPassed / (testsPassed + testsFailed)));
    }

    public static void runAllTests() {
        testBasicOperations();
        testChainingHashTable();
        testOpenAddressingLinear();
        testOpenAddressingQuadratic();
        testCuckooHashing();
        testHashFunctions();
        testLoadFactorManagement();
        testResizing();
        testEdgeCases();
        testPerformance();
    }

    /**
     * Test basic dictionary operations
     */
    private static void testBasicOperations() {
        System.out.println("\n=== Testing Basic Operations ===\n");

        ChainingHashTable<String, Integer> dict = new ChainingHashTable<>(new PolynomialHash());

        // Test insert
        assertNull("Insert new key should return null", dict.insert("key1", 10));
        assertEquals("Size after insert", 1, dict.size());

        // Test find
        assertEquals("Find existing key", 10, dict.find("key1"));
        assertNull("Find non-existing key", dict.find("key2"));

        // Test update through insert
        assertEquals("Update existing key", 10, dict.insert("key1", 20));
        assertEquals("Value after update", 20, dict.find("key1"));
        assertEquals("Size unchanged after update", 1, dict.size());

        // Test explicit update
        assertTrue("Update existing key", dict.update("key1", 30));
        assertEquals("Value after explicit update", 30, dict.find("key1"));
        assertFalse("Update non-existing key", dict.update("key2", 40));

        // Test delete
        assertEquals("Delete existing key", 30, dict.delete("key1"));
        assertEquals("Size after delete", 0, dict.size());
        assertNull("Find deleted key", dict.find("key1"));
        assertNull("Delete non-existing key", dict.delete("key2"));

        // Test isEmpty and clear
        assertTrue("Dictionary should be empty", dict.isEmpty());
        dict.insert("key1", 1);
        dict.insert("key2", 2);
        assertFalse("Dictionary should not be empty", dict.isEmpty());
        dict.clear();
        assertTrue("Dictionary should be empty after clear", dict.isEmpty());
        assertEquals("Size should be 0 after clear", 0, dict.size());
    }

    /**
     * Test chaining hash table specifically
     */
    private static void testChainingHashTable() {
        System.out.println("\n=== Testing Chaining Hash Table ===\n");

        ChainingHashTable<String, Integer> dict = new ChainingHashTable<>(4, 0.75, new PolynomialHash());

        // Insert multiple items
        for (int i = 0; i < 20; i++) {
            dict.insert("key" + i, i);
        }

        assertEquals("Size after 20 insertions", 20, dict.size());

        // Verify all items can be found
        boolean allFound = true;
        for (int i = 0; i < 20; i++) {
            if (dict.find("key" + i) == null || dict.find("key" + i) != i) {
                allFound = false;
                break;
            }
        }
        assertTrue("All inserted items should be found", allFound);

        // Check that collisions were handled
        assertTrue("Should have some collisions",
                dict.getStats().getTotalCollisions() > 0);
    }

    /**
     * Test open addressing with linear probing
     */
    private static void testOpenAddressingLinear() {
        System.out.println("\n=== Testing Open Addressing (Linear Probing) ===\n");

        OpenAddressingHashTable<String, Integer> dict =
                new OpenAddressingHashTable<>(4, 0.5, new PolynomialHash(),
                        OpenAddressingHashTable.ProbingStrategy.LINEAR);

        // Insert items
        for (int i = 0; i < 15; i++) {
            dict.insert("key" + i, i);
        }

        assertEquals("Size after insertions", 15, dict.size());

        // Verify all items
        boolean allFound = true;
        for (int i = 0; i < 15; i++) {
            if (dict.find("key" + i) == null || dict.find("key" + i) != i) {
                allFound = false;
                break;
            }
        }
        assertTrue("All items should be found", allFound);

        // Test deletion
        dict.delete("key5");
        assertNull("Deleted key should not be found", dict.find("key5"));
        assertEquals("Size after deletion", 14, dict.size());

        // Can still find other keys
        assertEquals("Other keys still accessible", 6, dict.find("key6"));
    }

    /**
     * Test open addressing with quadratic probing
     */
    private static void testOpenAddressingQuadratic() {
        System.out.println("\n=== Testing Open Addressing (Quadratic Probing) ===\n");

        OpenAddressingHashTable<String, Integer> dict =
                new OpenAddressingHashTable<>(4, 0.5, new PolynomialHash(),
                        OpenAddressingHashTable.ProbingStrategy.QUADRATIC);

        // Insert items
        for (int i = 0; i < 15; i++) {
            dict.insert("item" + i, i * 10);
        }

        assertEquals("Size after insertions", 15, dict.size());

        // Verify retrieval
        assertEquals("Find item", 50, dict.find("item5"));
        assertEquals("Find item", 100, dict.find("item10"));
    }

    /**
     * Test Cuckoo Hashing
     */
    private static void testCuckooHashing() {
        System.out.println("\n=== Testing Cuckoo Hashing ===\n");

        CuckooHashTable<String, Integer> dict =
                new CuckooHashTable<>(new PolynomialHash(), new SHA256Hash());

        // Insert items
        for (int i = 0; i < 50; i++) {
            dict.insert("cuckoo" + i, i);
        }

        assertEquals("Size after insertions", 50, dict.size());

        // Verify O(1) worst-case lookups (should find in at most 2 probes)
        assertEquals("Find item", 25, dict.find("cuckoo25"));
        assertTrue("Max probes should be 2",
                dict.getStats().getAverageProbesPerOperation() <= 2.1);

        // Test deletion
        dict.delete("cuckoo10");
        assertNull("Deleted key not found", dict.find("cuckoo10"));
        assertEquals("Size after deletion", 49, dict.size());
    }

    /**
     * Test different hash functions
     */
    private static void testHashFunctions() {
        System.out.println("\n=== Testing Hash Functions ===\n");

        HashFunction<String>[] hashFunctions = new HashFunction[]{
                new PolynomialHash(),
                new SHA256Hash()
        };

        String[] testKeys = {"test", "hash", "function", "collision"};
        int tableSize = 17;

        for (HashFunction<String> hf : hashFunctions) {
            boolean producesValidHashes = true;

            for (String key : testKeys) {
                int hash = hf.hash(key, tableSize);
                if (hash < 0 || hash >= tableSize) {
                    producesValidHashes = false;
                    break;
                }
            }

            assertTrue(hf.getName() + " produces valid hashes", producesValidHashes);
        }
    }

    /**
     * Test load factor management
     */
    private static void testLoadFactorManagement() {
        System.out.println("\n=== Testing Load Factor Management ===\n");

        ChainingHashTable<String, Integer> dict =
                new ChainingHashTable<>(4, 0.75, new PolynomialHash());

        double initialLF = dict.getLoadFactor();
        assertEquals("Initial load factor", 0.0, initialLF, 0.01);

        // Insert enough items to trigger resize
        for (int i = 0; i < 20; i++) {
            dict.insert("lf_test" + i, i);
        }

        assertTrue("Load factor should be managed", dict.getLoadFactor() <= 0.75);
        assertTrue("Should have resized", dict.getStats().getTotalResizes() > 0);
    }

    /**
     * Test resizing behavior
     */
    private static void testResizing() {
        System.out.println("\n=== Testing Resizing ===\n");

        OpenAddressingHashTable<String, Integer> dict =
                new OpenAddressingHashTable<>(2, 0.5, new PolynomialHash(),
                        OpenAddressingHashTable.ProbingStrategy.LINEAR);

        int initialCapacity = dict.getCapacity();

        // Insert enough items to force resize
        for (int i = 0; i < 50; i++) {
            dict.insert("resize" + i, i);
        }

        assertTrue("Capacity should increase", dict.getCapacity() > initialCapacity);

        // All items should still be accessible
        boolean allAccessible = true;
        for (int i = 0; i < 50; i++) {
            if (dict.find("resize" + i) == null || dict.find("resize" + i) != i) {
                allAccessible = false;
                break;
            }
        }
        assertTrue("All items accessible after resize", allAccessible);
    }

    /**
     * Test edge cases
     */
    private static void testEdgeCases() {
        System.out.println("\n=== Testing Edge Cases ===\n");

        ChainingHashTable<String, Integer> dict = new ChainingHashTable<>(new PolynomialHash());

        // Test with null key (should throw exception)
        boolean threwException = false;
        try {
            dict.insert(null, 10);
        } catch (IllegalArgumentException e) {
            threwException = true;
        }
        assertTrue("Should throw exception for null key", threwException);

        // Test with duplicate keys
        dict.insert("dup", 1);
        dict.insert("dup", 2);
        assertEquals("Duplicate key should update", 2, dict.find("dup"));
        assertEquals("Size should be 1", 1, dict.size());

        // Test with empty string key
        dict.insert("", 100);
        assertEquals("Empty string key should work", 100, dict.find(""));

        // Test with very long key
        String longKey = "a".repeat(1000);
        dict.insert(longKey, 999);
        assertEquals("Long key should work", 999, dict.find(longKey));
    }

    /**
     * Test performance characteristics
     */
    private static void testPerformance() {
        System.out.println("\n=== Testing Performance ===\n");

        HashFunction<String> hf = new PolynomialHash();
        int testSize = 10000;

        // Generate test data
        List<String> keys = new ArrayList<>();
        for (int i = 0; i < testSize; i++) {
            keys.add("perf" + i);
        }

        // Test Chaining
        ChainingHashTable<String, Integer> chainingDict = new ChainingHashTable<>(hf);
        long startTime = System.nanoTime();
        for (int i = 0; i < keys.size(); i++) {
            chainingDict.insert(keys.get(i), i);
        }
        long chainingInsertTime = System.nanoTime() - startTime;

        // Test Linear Probing
        OpenAddressingHashTable<String, Integer> linearDict =
                new OpenAddressingHashTable<>(hf, OpenAddressingHashTable.ProbingStrategy.LINEAR);
        startTime = System.nanoTime();
        for (int i = 0; i < keys.size(); i++) {
            linearDict.insert(keys.get(i), i);
        }
        long linearInsertTime = System.nanoTime() - startTime;

        System.out.printf("Chaining insert time: %.2f ms%n", chainingInsertTime / 1_000_000.0);
        System.out.printf("Linear probing insert time: %.2f ms%n", linearInsertTime / 1_000_000.0);

        // Both should complete in reasonable time
        assertTrue("Chaining should complete quickly", chainingInsertTime < 5_000_000_000L);
        assertTrue("Linear probing should complete quickly", linearInsertTime < 5_000_000_000L);
    }

    // Helper assertion methods
    private static void assertTrue(String message, boolean condition) {
        if (condition) {
            System.out.println("✓ PASS: " + message);
            testsPassed++;
        } else {
            System.out.println("✗ FAIL: " + message);
            testsFailed++;
        }
    }

    private static void assertEquals(String message, Object expected, Object actual) {
        if (expected == null && actual == null) {
            System.out.println("✓ PASS: " + message);
            testsPassed++;
        } else if (expected != null && expected.equals(actual)) {
            System.out.println("✓ PASS: " + message);
            testsPassed++;
        } else {
            System.out.println("✗ FAIL: " + message + " (expected: " + expected + ", actual: " + actual + ")");
            testsFailed++;
        }
    }

    private static void assertEquals(String message, double expected, double actual, double delta) {
        if (Math.abs(expected - actual) <= delta) {
            System.out.println("✓ PASS: " + message);
            testsPassed++;
        } else {
            System.out.println("✗ FAIL: " + message + " (expected: " + expected + ", actual: " + actual + ")");
            testsFailed++;
        }
    }

    private static void assertNull(String message, Object obj) {
        assertEquals(message, null, obj);
    }

    private static void assertFalse(String message, boolean condition) {
        assertTrue(message, !condition);
    }
}