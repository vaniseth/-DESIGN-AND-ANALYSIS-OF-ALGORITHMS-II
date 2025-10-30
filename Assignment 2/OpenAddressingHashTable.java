package group6_cs8050_assignment2;

/**
 * Hash Table Dictionary Implementation using Open Addressing
 * Supports Linear Probing and Quadratic Probing
 *
 * @param <K> Key type
 * @param <V> Value type
 */
public class OpenAddressingHashTable<K, V> implements Dictionary<K, V> {

    public enum ProbingStrategy {
        LINEAR,
        QUADRATIC
    }

    private static class Entry<K, V> {
        K key;
        V value;
        boolean isDeleted;

        Entry(K key, V value) {
            this.key = key;
            this.value = value;
            this.isDeleted = false;
        }
    }

    private Entry<K, V>[] table;
    private int size;
    private int capacity;
    private double maxLoadFactor;
    private HashFunction<K> hashFunction;
    private ProbingStrategy probingStrategy;
    private DictionaryStats stats;

    @SuppressWarnings("unchecked")
    public OpenAddressingHashTable(int initialCapacity, double maxLoadFactor,
                                   HashFunction<K> hashFunction, ProbingStrategy strategy) {
        this.capacity = getNextPrime(initialCapacity);
        this.table = new Entry[capacity];
        this.size = 0;
        this.maxLoadFactor = maxLoadFactor;
        this.hashFunction = hashFunction;
        this.probingStrategy = strategy;
        this.stats = new DictionaryStats();
    }

    public OpenAddressingHashTable(HashFunction<K> hashFunction, ProbingStrategy strategy) {
        this(16, 0.5, hashFunction, strategy);
    }

    @Override
    public V insert(K key, V value) {
        long startTime = System.nanoTime();

        if (key == null) {
            throw new IllegalArgumentException("Key cannot be null");
        }

        if (getLoadFactor() >= maxLoadFactor) {
            resize();
        }

        int hash = getHash(key);
        int index = hash;
        int probes = 0;
        int collisions = 0;
        int firstDeletedIndex = -1;

        for (int i = 0; i < capacity; i++) {
            probes++;

            if (table[index] == null) {
                // Empty slot found
                if (firstDeletedIndex != -1) {
                    // Use previously deleted slot
                    index = firstDeletedIndex;
                }
                table[index] = new Entry<>(key, value);
                size++;
                stats.recordInsertion(System.nanoTime() - startTime, collisions, probes);
                stats.updateMaxProbeLength(probes);
                return null;
            } else if (table[index].isDeleted) {
                // Deleted slot - remember it but continue searching
                if (firstDeletedIndex == -1) {
                    firstDeletedIndex = index;
                }
            } else if (table[index].key.equals(key)) {
                // Key already exists - update value
                V oldValue = table[index].value;
                table[index].value = value;
                stats.recordInsertion(System.nanoTime() - startTime, collisions, probes);
                return oldValue;
            } else {
                // Collision
                if (i == 0) collisions = 1;
            }

            // Calculate next probe position
            index = getNextProbe(hash, i + 1);
        }

        // Table is full (shouldn't happen with proper load factor management)
        throw new IllegalStateException("Hash table is full");
    }

    @Override
    public V find(K key) {
        long startTime = System.nanoTime();

        if (key == null) {
            return null;
        }

        int hash = getHash(key);
        int index = hash;
        int probes = 0;

        for (int i = 0; i < capacity; i++) {
            probes++;

            if (table[index] == null) {
                // Empty slot - key not found
                stats.recordFind(System.nanoTime() - startTime, probes);
                return null;
            } else if (!table[index].isDeleted && table[index].key.equals(key)) {
                // Key found
                stats.recordFind(System.nanoTime() - startTime, probes);
                return table[index].value;
            }

            // Continue probing
            index = getNextProbe(hash, i + 1);
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

        int hash = getHash(key);
        int index = hash;
        int probes = 0;

        for (int i = 0; i < capacity; i++) {
            probes++;

            if (table[index] == null) {
                // Empty slot - key not found
                stats.recordDeletion(System.nanoTime() - startTime, probes);
                return null;
            } else if (!table[index].isDeleted && table[index].key.equals(key)) {
                // Key found - mark as deleted
                V value = table[index].value;
                table[index].isDeleted = true;
                size--;
                stats.recordDeletion(System.nanoTime() - startTime, probes);
                return value;
            }

            // Continue probing
            index = getNextProbe(hash, i + 1);
        }

        stats.recordDeletion(System.nanoTime() - startTime, probes);
        return null;
    }

    @Override
    public boolean update(K key, V value) {
        if (key == null) {
            return false;
        }

        int hash = getHash(key);
        int index = hash;

        for (int i = 0; i < capacity; i++) {
            if (table[index] == null) {
                return false;
            } else if (!table[index].isDeleted && table[index].key.equals(key)) {
                table[index].value = value;
                return true;
            }

            index = getNextProbe(hash, i + 1);
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

    private int getNextProbe(int hash, int i) {
        int probe;

        switch (probingStrategy) {
            case LINEAR:
                probe = (hash + i) % capacity;
                break;
            case QUADRATIC:
                // Quadratic probing: h(k, i) = (h(k) + c1*i + c2*i^2) mod m
                // Using c1=1, c2=1 for simplicity
                probe = (hash + i + i * i) % capacity;
                break;
            default:
                probe = hash;
        }

        return probe;
    }

    @SuppressWarnings("unchecked")
    private void resize() {
        stats.recordResize();

        int oldCapacity = capacity;
        Entry<K, V>[] oldTable = table;

        capacity = getNextPrime(capacity * 2);
        table = new Entry[capacity];
        size = 0;

        // Rehash all non-deleted entries
        for (int i = 0; i < oldCapacity; i++) {
            if (oldTable[i] != null && !oldTable[i].isDeleted) {
                insert(oldTable[i].key, oldTable[i].value);
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

    public void setProbingStrategy(ProbingStrategy strategy) {
        this.probingStrategy = strategy;
    }

    public void setHashFunction(HashFunction<K> hashFunction) {
        this.hashFunction = hashFunction;
    }
}