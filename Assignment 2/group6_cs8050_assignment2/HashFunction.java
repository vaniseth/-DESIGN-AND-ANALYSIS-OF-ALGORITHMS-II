package group6_cs8050_assignment2;

public interface HashFunction<K> {
    int hash(K key, int tableSize);
    String getName();
}