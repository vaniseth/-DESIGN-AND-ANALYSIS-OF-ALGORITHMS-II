package group6_cs8050_assignment2;

/**
 * Cuckoo Hashing Implementation - Advanced Dictionary Extension
 * Uses two hash functions and two tables for O(1) worst-case lookup
 *
 * @param <K> Key type
 * @param <V> Value type
 */
public class CuckooHashTable<K, V> implements Dictionary<K, V> {

    private static class Entry<K, V> {
        K key;
        V value;

        Entry(K key, V value) {
            this.key = key;
            this.value = value;
        }
    }

    private Entry<K, V>[] table1;
    private Entry<K, V>[] table2;
    private int size;
    private int capacity;
    private double maxLoadFactor;
    private HashFunction<K> hashFunction1;
    private HashFunction<K> hashFunction2;
    private DictionaryStats stats;
    private static final int MAX_RELOCATIONS = 500;

    @SuppressWarnings("unchecked")
    public CuckooHashTable(int initialCapacity, double maxLoadFactor,
                           HashFunction<K> hashFunction1, HashFunction<K> hashFunction2) {
        this.capacity = getNextPrime(initialCapacity);
        this.table1 = new Entry[capacity];
        this.table2 = new Entry[capacity];
        this.size = 0;
        this.maxLoadFactor = maxLoadFactor;
        this.hashFunction1 = hashFunction1;
        this.hashFunction2 = hashFunction2;
        this.stats = new DictionaryStats();
    }

    public CuckooHashTable(HashFunction<K> hashFunction1, HashFunction<K> hashFunction2) {
        this(16, 0.5, hashFunction1, hashFunction2);
    }

    @Override
    public V insert(K key, V value) {
        long startTime = System.nanoTime();

        if (key == null) {
            throw new IllegalArgumentException("Key cannot be null");
        }

        // Check if key already exists
        int hash1 = getHash1(key);
        int hash2 = getHash2(key);

        if (table1[hash1] != null && table1[hash1].key.equals(key)) {
            V oldValue = table1[hash1].value;
            table1[hash1].value = value;
            stats.recordInsertion(System.nanoTime() - startTime, 0, 1);
            return oldValue;
        }

        if (table2[hash2] != null && table2[hash2].key.equals(key)) {
            V oldValue = table2[hash2].value;
            table2[hash2].value = value;
            stats.recordInsertion(System.nanoTime() - startTime, 0, 1);
            return oldValue;
        }

        // Check load factor before insertion
        if (getLoadFactor() >= maxLoadFactor) {
            resize();
            // Recalculate hashes after resize
            hash1 = getHash1(key);
            hash2 = getHash2(key);
        }

        // Try to insert in table1
        if (table1[hash1] == null) {
            table1[hash1] = new Entry<>(key, value);
            size++;
            stats.recordInsertion(System.nanoTime() - startTime, 0, 1);
            return null;
        }

        // Try to insert in table2
        if (table2[hash2] == null) {
            table2[hash2] = new Entry<>(key, value);
            size++;
            stats.recordInsertion(System.nanoTime() - startTime, 1, 2);
            return null;
        }

        // Both positions occupied - start cuckoo process
        Entry<K, V> currentEntry = new Entry<>(key, value);
        int relocations = 0;
        int probes = 2;

        while (relocations < MAX_RELOCATIONS) {
            // Try table1
            Entry<K, V> evicted = table1[getHash1(currentEntry.key)];
            table1[getHash1(currentEntry.key)] = currentEntry;

            if (evicted == null) {
                size++;
                stats.recordInsertion(System.nanoTime() - startTime, relocations, probes);
                return null;
            }

            currentEntry = evicted;
            relocations++;
            probes++;

            // Try table2
            evicted = table2[getHash2(currentEntry.key)];
            table2[getHash2(currentEntry.key)] = currentEntry;

            if (evicted == null) {
                size++;
                stats.recordInsertion(System.nanoTime() - startTime, relocations, probes);
                return null;
            }

            currentEntry = evicted;
            relocations++;
            probes++;
        }

        // Max relocations reached - rehash entire table
        rehash();
        stats.recordInsertion(System.nanoTime() - startTime, relocations, probes);
        return insert(currentEntry.key, currentEntry.value);
    }

    @Override
    public V find(K key) {
        long startTime = System.nanoTime();

        if (key == null) {
            return null;
        }

        // Check table1
        int hash1 = getHash1(key);
        if (table1[hash1] != null && table1[hash1].key.equals(key)) {
            stats.recordFind(System.nanoTime() - startTime, 1);
            return table1[hash1].value;
        }

        // Check table2
        int hash2 = getHash2(key);
        if (table2[hash2] != null && table2[hash2].key.equals(key)) {
            stats.recordFind(System.nanoTime() - startTime, 2);
            return table2[hash2].value;
        }

        stats.recordFind(System.nanoTime() - startTime, 2);
        return null;
    }

    @Override
    public V delete(K key) {
        long startTime = System.nanoTime();

        if (key == null) {
            return null;
        }

        // Check table1
        int hash1 = getHash1(key);
        if (table1[hash1] != null && table1[hash1].key.equals(key)) {
            V value = table1[hash1].value;
            table1[hash1] = null;
            size--;
            stats.recordDeletion(System.nanoTime() - startTime, 1);
            return value;
        }

        // Check table2
        int hash2 = getHash2(key);
        if (table2[hash2] != null && table2[hash2].key.equals(key)) {
            V value = table2[hash2].value;
            table2[hash2] = null;
            size--;
            stats.recordDeletion(System.nanoTime() - startTime, 2);
            return value;
        }

        stats.recordDeletion(System.nanoTime() - startTime, 2);
        return null;
    }

    @Override
    public boolean update(K key, V value) {
        if (key == null) {
            return false;
        }

        int hash1 = getHash1(key);
        if (table1[hash1] != null && table1[hash1].key.equals(key)) {
            table1[hash1].value = value;
            return true;
        }

        int hash2 = getHash2(key);
        if (table2[hash2] != null && table2[hash2].key.equals(key)) {
            table2[hash2].value = value;
            return true;
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
            table1[i] = null;
            table2[i] = null;
        }
        size = 0;
        stats.reset();
    }

    @Override
    public double getLoadFactor() {
        return (double) size / (2 * capacity);
    }

    @Override
    public DictionaryStats getStats() {
        return stats;
    }

    private int getHash1(K key) {
        return hashFunction1.hash(key, capacity);
    }

    private int getHash2(K key) {
        return hashFunction2.hash(key, capacity);
    }

    @SuppressWarnings("unchecked")
    private void resize() {
        stats.recordResize();

        int oldCapacity = capacity;
        Entry<K, V>[] oldTable1 = table1;
        Entry<K, V>[] oldTable2 = table2;

        capacity = getNextPrime(capacity * 2);
        table1 = new Entry[capacity];
        table2 = new Entry[capacity];
        size = 0;

        // Reinsert all entries
        for (int i = 0; i < oldCapacity; i++) {
            if (oldTable1[i] != null) {
                insert(oldTable1[i].key, oldTable1[i].value);
            }
            if (oldTable2[i] != null) {
                insert(oldTable2[i].key, oldTable2[i].value);
            }
        }
    }

    @SuppressWarnings("unchecked")
    private void rehash() {
        stats.recordResize();

        int oldCapacity = capacity;
        Entry<K, V>[] oldTable1 = table1;
        Entry<K, V>[] oldTable2 = table2;

        // Keep same capacity but clear tables
        table1 = new Entry[capacity];
        table2 = new Entry[capacity];
        size = 0;

        // Reinsert all entries (hash functions will produce different values due to randomization)
        for (int i = 0; i < oldCapacity; i++) {
            if (oldTable1[i] != null) {
                insert(oldTable1[i].key, oldTable1[i].value);
            }
            if (oldTable2[i] != null) {
                insert(oldTable2[i].key, oldTable2[i].value);
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

    /**
     * Get statistics specific to Cuckoo Hashing
     */
    public String getCuckooStats() {
        int table1Occupied = 0;
        int table2Occupied = 0;

        for (int i = 0; i < capacity; i++) {
            if (table1[i] != null) table1Occupied++;
            if (table2[i] != null) table2Occupied++;
        }

        return String.format(
                "Cuckoo Hash Statistics:\n" +
                        "  Total capacity: %d (2 tables × %d)\n" +
                        "  Size: %d\n" +
                        "  Load factor: %.4f\n" +
                        "  Table1 occupancy: %d/%d (%.2f%%)\n" +
                        "  Table2 occupancy: %d/%d (%.2f%%)\n" +
                        "  %s",
                2 * capacity, capacity, size, getLoadFactor(),
                table1Occupied, capacity, (100.0 * table1Occupied / capacity),
                table2Occupied, capacity, (100.0 * table2Occupied / capacity),
                stats.toString()
        );
    }
}