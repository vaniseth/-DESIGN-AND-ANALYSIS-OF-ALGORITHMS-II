package group6_cs8050_assignment2;

public class PolynomialHash implements HashFunction<String> {
    private static final int BASE = 31; //uses base-31 polynomial

    @Override
    public int hash(String key, int tableSize) {
        if (key == null) return 0;

        long hash = 0;
        long pow = 1;

        for (int i = 0; i < key.length(); i++) {
            hash = (hash + (key.charAt(i) * pow)) % tableSize;
            pow = (pow * BASE) % tableSize;
        }

        return (int) ((hash % tableSize + tableSize) % tableSize);
    }

    @Override
    public String getName() {
        return "Polynomial Rolling Hash";
    }
}
