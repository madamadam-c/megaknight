use std::mem::size_of;

use cozy_chess::Move;

const CLUSTER_SIZE: usize = 2;
const EXACT_BONUS: i64 = 2;

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

#[derive(Clone, Copy, Default)]
struct TableSlot {
    epoch: u32,
    entry: Option<TableEntry>,
}

#[derive(Clone, Copy, Default)]
struct TableBucket {
    slots: [TableSlot; CLUSTER_SIZE],
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
        let bucket_size = size_of::<TableBucket>() as u128;
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

    pub fn get(&self, key: u64) -> Option<TableEntry> {
        self.buckets[(key as usize) & self.mask]
            .slots
            .iter()
            .filter(|slot| slot.epoch == self.epoch)
            .find_map(|slot| slot.entry.filter(|entry| entry.key == key))
    }

    pub fn insert(&mut self, key: u64, entry: TableEntry) {
        let epoch = self.epoch;
        let bucket = &mut self.buckets[(key as usize) & self.mask];

        if let Some(slot) = bucket.slots.iter_mut().find(|slot| {
            slot.epoch == epoch && slot.entry.is_some_and(|current| current.key == key)
        }) {
            let current = slot.entry.unwrap();
            if entry.depth >= current.depth - 2
                || (entry.node_type == TTNodeType::EXACT && current.node_type != TTNodeType::EXACT)
            {
                *slot = TableSlot {
                    epoch,
                    entry: Some(entry),
                };
            }
            return;
        }

        if let Some(slot) = bucket
            .slots
            .iter_mut()
            .find(|slot| slot.epoch != epoch || slot.entry.is_none())
        {
            *slot = TableSlot {
                epoch,
                entry: Some(entry),
            };
            return;
        }

        let victim_index = bucket
            .slots
            .iter()
            .enumerate()
            .min_by_key(|(_, slot)| entry_value(slot.entry.unwrap()))
            .map(|(index, _)| index)
            .unwrap();

        bucket.slots[victim_index] = TableSlot {
            epoch,
            entry: Some(entry),
        };
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
        let bytes = table.buckets.len() * size_of::<TableBucket>();

        assert!(table.buckets.len().is_power_of_two());
        assert!(bytes <= 1024 * 1024);
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
