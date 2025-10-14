package group6_cs8050_assignment2;

public class FNV1aHash implements HashFunction<String> {
    private static final long FNV_PRIME = 0x100000001b3L;
    private static final long FNV_OFFSET_BASIS = 0xcbf29ce484222325L;

    @Override
    public int hash(String key, int tableSize) {
        if (key == null) return 0;

        long hash = FNV_OFFSET_BASIS;

        for (int i = 0; i < key.length(); i++) {
            hash ^= key.charAt(i);
            hash *= FNV_PRIME;
        }

        return (int) ((hash % tableSize + tableSize) % tableSize);
    }

    @Override
    public String getName() {
        return "FNV-1a Hash";
    }
}