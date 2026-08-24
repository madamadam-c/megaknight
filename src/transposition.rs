use std::mem::size_of;

use cozy_chess::Move;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    EXACT,
    LOWER,
    UPPER,
}

#[derive(Clone, Copy)]
pub struct TableEntry {
    pub key: u64,
    pub score: i32,
    pub depth: i32,
    pub node_type: NodeType,
    pub best_move: Option<Move>,
}

pub struct Table {
    mask: usize,
    generation: u32,
    table: Vec<TableSlot>,
}

#[derive(Clone, Copy, Default)]
struct TableSlot {
    generation: u32,
    entry: Option<TableEntry>,
}

impl Table {
    pub fn new(size: usize) -> Self {
        assert!(size > 0 && size.is_power_of_two());

        Self {
            mask: size - 1,
            generation: 1,
            table: vec![TableSlot::default(); size],
        }
    }

    pub fn new_for_mb(megabytes: u64) -> Self {
        let bytes = u128::from(megabytes.max(1)) * 1024 * 1024;
        let slot_size = size_of::<TableSlot>() as u128;
        let max_slots = (bytes / slot_size).min(usize::MAX as u128) as usize;
        let size = largest_power_of_two(max_slots.max(1));

        Self::new(size)
    }

    pub fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.table.fill(TableSlot::default());
            self.generation = 1;
        }
    }

    pub fn get(&self, key: u64) -> Option<TableEntry> {
        let slot = self.table[(key as usize) & self.mask];
        if slot.generation != self.generation {
            return None;
        }
        if let Some(entry) = slot.entry {
            if entry.key == key { Some(entry) } else { None }
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: u64, entry: TableEntry) {
        let k = (key as usize) & self.mask;
        let slot = &mut self.table[k];

        if slot.generation != self.generation
            || slot.entry.is_none()
            || entry.depth >= slot.entry.unwrap().depth
            || key != slot.entry.unwrap().key
        {
            *slot = TableSlot {
                generation: self.generation,
                entry: Some(entry),
            };
        }
    }
}

fn largest_power_of_two(value: usize) -> usize {
    1usize << (usize::BITS - 1 - value.leading_zeros())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_size_fits_requested_memory() {
        let table = Table::new_for_mb(1);
        let bytes = table.table.len() * size_of::<TableSlot>();

        assert!(table.table.len().is_power_of_two());
        assert!(bytes <= 1024 * 1024);
    }

    #[test]
    fn clear_invalidates_entries_without_reallocating() {
        let mut table = Table::new(1);
        let allocation = table.table.as_ptr();
        let entry = TableEntry {
            key: 42,
            score: 100,
            depth: 3,
            node_type: NodeType::EXACT,
            best_move: None,
        };
        table.insert(entry.key, entry);

        table.clear();

        assert!(table.get(entry.key).is_none());
        assert_eq!(table.table.as_ptr(), allocation);
    }
}
