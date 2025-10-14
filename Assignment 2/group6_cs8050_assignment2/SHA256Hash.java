package group6_cs8050_assignment2;

import java.security.MessageDigest;

public class SHA256Hash implements HashFunction<String> {
    @Override
    public int hash(String key, int tableSize) {
        if (key == null) return 0;

        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hashBytes = md.digest(key.getBytes());

            int hash = 0;
            for (int i = 0; i < 4 && i < hashBytes.length; i++) {
                hash = (hash << 8) | (hashBytes[i] & 0xFF);
            }

            return (int) ((hash % tableSize + tableSize) % tableSize);
        } catch (Exception e) {
            return key.hashCode() % tableSize;
        }
    }

    @Override
    public String getName() {
        return "SHA-256 Cryptographic Hash";
    }
}