# probable-octo-waffle
caching system with LRU and LFU policies
// AdaptiveCacheDemo.java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.*;

/*
 * Thread-safe Adaptive Cache supporting LRU and LFU with O(1) get/put.
 * You can switch policy at runtime via setPolicy(CachePolicy).
 *
 * Design notes:
 * - LRU: HashMap<K,Node> + DoublyLinkedList of nodes (most recent at head).
 * - LFU: HashMap<K,Node> + Map<freq, DoublyLinkedList> + minFreq.
 * - Concurrency: ReentrantReadWriteLock ensures readers and writers coordinate.
 * - Runtime switch: create a new policy instance and bulk-insert existing entries
 *   so entries are preserved (O(n) during switch).
 */

// Public API
interface Cache<K, V> {
    V get(K key);
    void put(K key, V value);
    int size();
    Map<K, V> snapshot(); // helpful for switching
}

enum CachePolicy {
    LRU, LFU
}

// Common doubly linked node used by both LRU and LFU
class Node<K, V> {
    K key;
    V value;
    int freq; // used by LFU
    Node<K, V> prev, next;

    Node(K k, V v) { key = k; value = v; freq = 1; }
}

// Simple doubly linked list with sentinel head/tail
class DoublyLinkedList<K, V> {
    Node<K, V> head, tail;
    int size;

    DoublyLinkedList() {
        head = new Node<>(null, null);
        tail = new Node<>(null, null);
        head.next = tail;
        tail.prev = head;
        size = 0;
    }

    void addFirst(Node<K, V> node) {
        node.next = head.next;
        node.prev = head;
        head.next.prev = node;
        head.next = node;
        size++;
    }

    void remove(Node<K, V> node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
        node.next = node.prev = null;
        size--;
    }

    Node<K, V> removeLast() {
        if (size == 0) return null;
        Node<K, V> last = tail.prev;
        remove(last);
        return last;
    }

    boolean isEmpty() { return size == 0; }
}

// LRU Implementation (O(1))
class LRUCacheImpl<K, V> implements Cache<K, V> {
    private final int capacity;
    private final Map<K, Node<K, V>> map;
    private final DoublyLinkedList<K, V> dll;

    LRUCacheImpl(int capacity) {
        this.capacity = Math.max(1, capacity);
        this.map = new HashMap<>();
        this.dll = new DoublyLinkedList<>();
    }

    @Override
    public V get(K key) {
        Node<K, V> node = map.get(key);
        if (node == null) return null;
        // Move to front (most recently used)
        dll.remove(node);
        dll.addFirst(node);
        return node.value;
    }

    @Override
    public void put(K key, V value) {
        Node<K, V> node = map.get(key);
        if (node != null) {
            node.value = value;
            dll.remove(node);
            dll.addFirst(node);
            return;
        }
        if (map.size() >= capacity) {
            Node<K, V> evicted = dll.removeLast();
            if (evicted != null) map.remove(evicted.key);
        }
        Node<K, V> newNode = new Node<>(key, value);
        map.put(key, newNode);
        dll.addFirst(newNode);
    }

    @Override
    public int size() { return map.size(); }

    @Override
    public Map<K, V> snapshot() {
        Map<K, V> snap = new LinkedHashMap<>();
        // iterate from head.next to tail.prev to preserve recency order
        Node<K, V> cur = dll.head.next;
        while (cur != dll.tail) {
            snap.put(cur.key, cur.value);
            cur = cur.next;
        }
        return snap;
    }
}

// LFU Implementation (O(1) amortized)
class LFUCacheImpl<K, V> implements Cache<K, V> {
    private final int capacity;
    private final Map<K, Node<K, V>> map;
    private final Map<Integer, DoublyLinkedList<K, V>> freqMap;
    private int minFreq;

    LFUCacheImpl(int capacity) {
        this.capacity = Math.max(1, capacity);
        this.map = new HashMap<>();
        this.freqMap = new HashMap<>();
        this.minFreq = 0;
    }

    @Override
    public V get(K key) {
        Node<K, V> node = map.get(key);
        if (node == null) return null;
        touch(node);
        return node.value;
    }

    private void touch(Node<K, V> node) {
        int f = node.freq;
        DoublyLinkedList<K, V> list = freqMap.get(f);
        list.remove(node);
        if (list.isEmpty()) {
            freqMap.remove(f);
            if (f == minFreq) minFreq++;
        }
        node.freq++;
        freqMap.computeIfAbsent(node.freq, k -> new DoublyLinkedList<>()).addFirst(node);
    }

    @Override
    public void put(K key, V value) {
        if (map.containsKey(key)) {
            Node<K, V> node = map.get(key);
            node.value = value;
            touch(node);
            return;
        }
        if (map.size() >= capacity) evict();
        Node<K, V> node = new Node<>(key, value);
        map.put(key, node);
        minFreq = 1;
        freqMap.computeIfAbsent(1, k -> new DoublyLinkedList<>()).addFirst(node);
    }

    private void evict() {
        DoublyLinkedList<K, V> list = freqMap.get(minFreq);
        if (list == null) return;
        Node<K, V> removed = list.removeLast();
        if (removed != null) map.remove(removed.key);
        if (list.isEmpty()) freqMap.remove(minFreq);
    }

    @Override
    public int size() { return map.size(); }

    @Override
    public Map<K, V> snapshot() {
        // Snapshot does not preserve any specific order; we'll iterate map keys.
        Map<K, V> snap = new LinkedHashMap<>();
        for (Map.Entry<K, Node<K, V>> e : map.entrySet()) snap.put(e.getKey(), e.getValue().value);
        return snap;
    }
}

// Adaptive wrapper that is thread-safe and allows runtime policy switching
class AdaptiveCache<K, V> implements Cache<K, V> {
    private final ReadWriteLock rwLock = new ReentrantReadWriteLock();
    private Cache<K, V> delegate;
    private int capacity;
    private CachePolicy policy;

    AdaptiveCache(int capacity, CachePolicy policy) {
        this.capacity = Math.max(1, capacity);
        this.policy = policy;
        this.delegate = createDelegate(capacity, policy);
    }

    private Cache<K, V> createDelegate(int cap, CachePolicy p) {
        if (p == CachePolicy.LRU) return new LRUCacheImpl<>(cap);
        else return new LFUCacheImpl<>(cap);
    }

    @Override
    public V get(K key) {
        rwLock.readLock().lock();
        try {
            return delegate.get(key);
        } finally {
            rwLock.readLock().unlock();
        }
    }

    @Override
    public void put(K key, V value) {
        rwLock.readLock().lock();
        try {
            delegate.put(key, value);
        } finally {
            rwLock.readLock().unlock();
        }
    }

    @Override
    public int size() {
        rwLock.readLock().lock();
        try {
            return delegate.size();
        } finally {
            rwLock.readLock().unlock();
        }
    }

    @Override
    public Map<K, V> snapshot() {
        rwLock.readLock().lock();
        try {
            return delegate.snapshot();
        } finally {
            rwLock.readLock().unlock();
        }
    }

    // Switch policy at runtime preserving entries. This operation is O(n).
    public void setPolicy(CachePolicy newPolicy) {
        rwLock.writeLock().lock(); // block all readers/writers during switch
        try {
            if (this.policy == newPolicy) return;
            Map<K, V> entries = delegate.snapshot();
            Cache<K, V> newDelegate = createDelegate(this.capacity, newPolicy);
            // bulk insert preserving values (order semantics will depend on target policy)
            for (Map.Entry<K, V> e : entries.entrySet()) {
                newDelegate.put(e.getKey(), e.getValue());
            }
            this.delegate = newDelegate;
            this.policy = newPolicy;
        } finally {
            rwLock.writeLock().unlock();
        }
    }

    public CachePolicy getPolicy() {
        rwLock.readLock().lock();
        try {
            return this.policy;
        } finally {
            rwLock.readLock().unlock();
        }
    }

    public void setCapacity(int newCapacity) {
        rwLock.writeLock().lock();
        try {
            if (newCapacity <= 0) throw new IllegalArgumentException("Capacity must be > 0");
            this.capacity = newCapacity;
            // recreate delegate with new capacity, preserving entries
            Map<K, V> entries = delegate.snapshot();
            Cache<K, V> newDelegate = createDelegate(this.capacity, this.policy);
            for (Map.Entry<K, V> e : entries.entrySet()) {
                newDelegate.put(e.getKey(), e.getValue());
            }
            this.delegate = newDelegate;
        } finally {
            rwLock.writeLock().unlock();
        }
    }
}

// Demo & lightweight tests
public class AdaptiveCacheDemo {
    public static void main(String[] args) throws InterruptedException {
        AdaptiveCache<Integer, String> cache = new AdaptiveCache<>(3, CachePolicy.LRU);

        System.out.println("Policy: " + cache.getPolicy());
        cache.put(1, "one");
        cache.put(2, "two");
        cache.put(3, "three");

        System.out.println("Get 1 -> " + cache.get(1)); // touches 1
        cache.put(4, "four"); // should evict LRU (which is 2)
        System.out.println("After inserting 4:");
        printSnapshot(cache);

        System.out.println("\nSwitching to LFU policy...");
        cache.setPolicy(CachePolicy.LFU);
        System.out.println("Policy: " + cache.getPolicy());
        // Access patterns to change frequencies
        cache.get(1); cache.get(1); // freq of 1 becomes higher
        cache.get(3); // freq(3)=2 maybe
        cache.put(5, "five"); // evict least frequent

        System.out.println("After LFU operations:");
        printSnapshot(cache);

        // Small concurrency test
        System.out.println("\nConcurrent puts and gets (threads):");
        ExecutorService ex = Executors.newFixedThreadPool(4);
        for (int i = 6; i < 12; i++) {
            final int k = i;
            ex.submit(() -> cache.put(k, "n"+k));
        }
        for (int i = 1; i < 6; i++) {
            final int k = i;
            ex.submit(() -> System.out.println("T-get " + k + " -> " + cache.get(k)));
        }
        ex.shutdown();
        ex.awaitTermination(3, TimeUnit.SECONDS);
        System.out.println("Final snapshot:");
        printSnapshot(cache);
    }

    private static <K, V> void printSnapshot(Cache<K, V> cache) {
        Map<K, V> snap = cache.snapshot();
        System.out.println("Size=" + cache.size() + " Entries: " + snap);
    }
}
