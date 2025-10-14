package group6_cs8050_assignment2;

/**
 * Dictionary Abstract Data Type Interface
 * Defines the contract for a key-value associative array with standard operations
 *
 * @param <K> Key type (must be comparable and hashable)
 * @param <V> Value type
 */
public interface Dictionary<K, V> {

    /**
     * Inserts a key-value pair into the dictionary
     * If key exists, updates the value
     *
     * @param key The key to insert
     * @param value The value associated with the key
     * @return Previous value if key existed, null otherwise
     */
    V insert(K key, V value);

    /**
     * Finds and returns the value associated with the given key
     *
     * @param key The key to search for
     * @return The value associated with the key, or null if not found
     */
    V find(K key);

    /**
     * Removes a key-value pair from the dictionary
     *
     * @param key The key to remove
     * @return The value that was associated with the key, or null if not found
     */
    V delete(K key);

    /**
     * Updates the value for an existing key
     *
     * @param key The key to update
     * @param value The new value
     * @return true if key existed and was updated, false otherwise
     */
    boolean update(K key, V value);

    /**
     * Returns the number of key-value pairs in the dictionary
     *
     * @return Size of the dictionary
     */
    int size();

    /**
     * Checks if the dictionary is empty
     *
     * @return true if dictionary contains no elements
     */
    boolean isEmpty();

    /**
     * Removes all entries from the dictionary
     */
    void clear();

    /**
     * Returns the current load factor of the dictionary
     *
     * @return Current load factor (size / capacity)
     */
    double getLoadFactor();

    /**
     * Returns statistics about the dictionary implementation
     *
     * @return DictionaryStats object containing performance metrics
     */
    DictionaryStats getStats();
}