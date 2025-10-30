package group6_cs8050_assignment2;

import java.util.LinkedList;

/**
 * Hash Table Dictionary Implementation using Chaining
 * Supports dynamic resizing and pluggable hash functions
 *
 * @param <K> Key type
 * @param <V> Value type
 */
public class ChainingHashTable<K, V> implements Dictionary<K, V> {

    private static class Entry<K, V> {
        K key;
        V value;

        Entry(K key, V value) {
            this.key = key;
            this.value = value;
        }
    }

    private LinkedList<Entry<K, V>>[] table;
    private int size;
    private int capacity;
    private double maxLoadFactor;
    private HashFunction<K> hashFunction;
    private DictionaryStats stats;

    @SuppressWarnings("unchecked")
    public ChainingHashTable(int initialCapacity, double maxLoadFactor, HashFunction<K> hashFunction) {
        this.capacity = getNextPrime(initialCapacity);
        this.table = new LinkedList[capacity];
        this.size = 0;
        this.maxLoadFactor = maxLoadFactor;
        this.hashFunction = hashFunction;
        this.stats = new DictionaryStats();
    }

    public ChainingHashTable(HashFunction<K> hashFunction) {
        this(16, 0.75, hashFunction);
    }

    @Override
    public V insert(K key, V value) {
        long startTime = System.nanoTime();

        if (key == null) {
            throw new IllegalArgumentException("Key cannot be null");
        }

        int index = getHash(key);

        if (table[index] == null) {
            table[index] = new LinkedList<>();
        }

        int collisions = 0;
        int probes = 0;

        // Check if key already exists
        for (Entry<K, V> entry : table[index]) {
            probes++;
            if (entry.key.equals(key)) {
                V oldValue = entry.value;
                entry.value = value;
                stats.recordInsertion(System.nanoTime() - startTime, collisions, probes);
                return oldValue;
            }
        }

        // Key doesn't exist, add new entry
        if (table[index].size() > 0) {
            collisions = 1;
        }

        table[index].add(new Entry<>(key, value));
        size++;

        stats.recordInsertion(System.nanoTime() - startTime, collisions, probes);
        stats.updateMaxChainLength(table[index].size());

        // Check if resize is needed
        if (getLoadFactor() > maxLoadFactor) {
            resize();
        }

        return null;
    }

    @Override
    public V find(K key) {
        long startTime = System.nanoTime();

        if (key == null) {
            return null;
        }

        int index = getHash(key);
        int probes = 0;

        if (table[index] != null) {
            for (Entry<K, V> entry : table[index]) {
                probes++;
                if (entry.key.equals(key)) {
                    stats.recordFind(System.nanoTime() - startTime, probes);
                    return entry.value;
                }
            }
        }

        stats.recordFind(System.nanoTime() - startTime, probes);
        return null;
    }

    @Override
    public V delete(K key) {
        long startTime = System.nanoTime();

        if (key == null) {
            return null;
        }

        int index = getHash(key);
        int probes = 0;

        if (table[index] != null) {
            for (int i = 0; i < table[index].size(); i++) {
                probes++;
                Entry<K, V> entry = table[index].get(i);
                if (entry.key.equals(key)) {
                    table[index].remove(i);
                    size--;
                    stats.recordDeletion(System.nanoTime() - startTime, probes);
                    return entry.value;
                }
            }
        }

        stats.recordDeletion(System.nanoTime() - startTime, probes);
        return null;
    }

    @Override
    public boolean update(K key, V value) {
        if (key == null) {
            return false;
        }

        int index = getHash(key);

        if (table[index] != null) {
            for (Entry<K, V> entry : table[index]) {
                if (entry.key.equals(key)) {
                    entry.value = value;
                    return true;
                }
            }
        }

        return false;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public boolean isEmpty() {
        return size == 0;
    }

    @Override
    public void clear() {
        for (int i = 0; i < capacity; i++) {
            table[i] = null;
        }
        size = 0;
        stats.reset();
    }

    @Override
    public double getLoadFactor() {
        return (double) size / capacity;
    }

    @Override
    public DictionaryStats getStats() {
        return stats;
    }

    private int getHash(K key) {
        return hashFunction.hash(key, capacity);
    }

    @SuppressWarnings("unchecked")
    private void resize() {
        stats.recordResize();

        int oldCapacity = capacity;
        LinkedList<Entry<K, V>>[] oldTable = table;

        capacity = getNextPrime(capacity * 2);
        table = new LinkedList[capacity];
        size = 0;

        // Rehash all entries
        for (int i = 0; i < oldCapacity; i++) {
            if (oldTable[i] != null) {
                for (Entry<K, V> entry : oldTable[i]) {
                    insert(entry.key, entry.value);
                }
            }
        }
    }

    private int getNextPrime(int n) {
        if (n <= 2) return 2;
        if (n % 2 == 0) n++;

        while (!isPrime(n)) {
            n += 2;
        }

        return n;
    }

    private boolean isPrime(int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;

        for (int i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) {
                return false;
            }
        }

        return true;
    }

    public int getCapacity() {
        return capacity;
    }

    public void setHashFunction(HashFunction<K> hashFunction) {
        this.hashFunction = hashFunction;
    }
}