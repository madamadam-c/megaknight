use cozy_chess::Move;

const CLUSTER_SIZE: usize = 2;
const EXACT_BONUS: i64 = 2;
const LEGACY_BUCKET_SIZE: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TTNodeType {
    EXACT,
    LOWER,
    UPPER,
}

#[derive(Clone, Copy)]
pub struct TableEntry {
    pub key: u64,
    pub score: i32,
    pub depth: i32,
    pub node_type: TTNodeType,
    pub best_move: Option<Move>,
}

pub struct Table {
    mask: usize,
    epoch: u32,
    buckets: Vec<TableBucket>,
}

#[derive(Clone, Copy)]
struct TableBucket {
    epoch: u32,
    len: u8,
    slots: [TableEntry; CLUSTER_SIZE],
}

impl Default for TableBucket {
    fn default() -> Self {
        Self {
            epoch: 0,
            len: 0,
            slots: [TableEntry {
                key: 0,
                score: 0,
                depth: 0,
                node_type: TTNodeType::UPPER,
                best_move: None,
            }; CLUSTER_SIZE],
        }
    }
}

impl Table {
    pub fn new(size: usize) -> Self {
        assert!(size > 0 && size.is_power_of_two());

        Self {
            mask: size - 1,
            epoch: 1,
            buckets: vec![TableBucket::default(); size],
        }
    }

    pub fn new_for_mb(megabytes: u64) -> Self {
        let bytes = u128::from(megabytes.max(1)) * 1024 * 1024;
        // Keep the original capacity and indexing despite the denser bucket layout.
        let bucket_size = LEGACY_BUCKET_SIZE as u128;
        let max_buckets = (bytes / bucket_size).min(usize::MAX as u128) as usize;
        let size = largest_power_of_two(max_buckets.max(1));

        Self::new(size)
    }

    pub fn clear(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.buckets.fill(TableBucket::default());
            self.epoch = 1;
        }
    }

    #[inline(always)]
    pub fn get(&self, key: u64) -> Option<TableEntry> {
        let bucket = &self.buckets[(key as usize) & self.mask];
        if bucket.epoch != self.epoch {
            return None;
        }

        let mut index = 0;
        while index < bucket.len as usize {
            let entry = bucket.slots[index];
            if entry.key == key {
                return Some(entry);
            }
            index += 1;
        }
        None
    }

    #[inline(always)]
    pub fn insert(&mut self, key: u64, entry: TableEntry) {
        let epoch = self.epoch;
        let bucket = &mut self.buckets[(key as usize) & self.mask];

        if bucket.epoch != epoch {
            bucket.epoch = epoch;
            bucket.len = 0;
        }

        let mut index = 0;
        while index < bucket.len as usize {
            let current = bucket.slots[index];
            if current.key == key {
                if entry.depth >= current.depth - 2
                    || (entry.node_type == TTNodeType::EXACT
                        && current.node_type != TTNodeType::EXACT)
                {
                    bucket.slots[index] = entry;
                }
                return;
            }
            index += 1;
        }

        if (bucket.len as usize) < CLUSTER_SIZE {
            bucket.slots[bucket.len as usize] = entry;
            bucket.len += 1;
            return;
        }

        let victim_index = if entry_value(bucket.slots[0]) <= entry_value(bucket.slots[1]) {
            0
        } else {
            1
        };
        bucket.slots[victim_index] = entry;
    }
}

fn entry_value(entry: TableEntry) -> i64 {
    i64::from(entry.depth)
        + if entry.node_type == TTNodeType::EXACT {
            EXACT_BONUS
        } else {
            0
        }
}

fn largest_power_of_two(value: usize) -> usize {
    1usize << (usize::BITS - 1 - value.leading_zeros())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(key: u64, depth: i32, node_type: TTNodeType) -> TableEntry {
        TableEntry {
            key,
            score: depth,
            depth,
            node_type,
            best_move: None,
        }
    }

    #[test]
    fn table_size_fits_requested_memory() {
        let table = Table::new_for_mb(1);
        let bytes = table.buckets.len() * std::mem::size_of::<TableBucket>();

        assert!(table.buckets.len().is_power_of_two());
        assert!(bytes <= 1024 * 1024);
        assert!(std::mem::size_of::<TableBucket>() < LEGACY_BUCKET_SIZE);
    }

    #[test]
    fn clear_invalidates_entries_without_reallocating() {
        let mut table = Table::new(1);
        let allocation = table.buckets.as_ptr();
        let entry = entry(42, 3, TTNodeType::EXACT);
        table.insert(entry.key, entry);

        table.clear();

        assert!(table.get(entry.key).is_none());
        assert_eq!(table.buckets.as_ptr(), allocation);
    }

    #[test]
    fn colliding_keys_share_a_cluster() {
        let mut table = Table::new(1);
        for key in 0..CLUSTER_SIZE as u64 {
            table.insert(key, entry(key, key as i32 + 1, TTNodeType::LOWER));
        }

        for key in 0..CLUSTER_SIZE as u64 {
            assert!(table.get(key).is_some());
        }
    }

    #[test]
    fn full_cluster_always_replaces_its_lowest_value_entry() {
        let mut table = Table::new(1);
        table.insert(1, entry(1, 10, TTNodeType::EXACT));
        table.insert(2, entry(2, 8, TTNodeType::LOWER));

        table.insert(100, entry(100, 1, TTNodeType::LOWER));

        assert!(table.get(100).is_some());
        assert!(table.get(1).is_some());
        assert!(table.get(2).is_none());
    }

    #[test]
    fn same_key_keeps_a_much_deeper_current_entry() {
        let mut table = Table::new(1);
        table.insert(42, entry(42, 10, TTNodeType::EXACT));

        table.insert(42, entry(42, 1, TTNodeType::LOWER));

        assert_eq!(table.get(42).unwrap().depth, 10);
    }
}
