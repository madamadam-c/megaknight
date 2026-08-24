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
    table: Vec<Option<TableEntry>>,
}

impl Table {
    pub fn new(size: usize) -> Self {
        assert!(size > 0 && size.is_power_of_two());

        Self {
            mask: size - 1,
            table: vec![None; size],
        }
    }

    pub fn new_for_mb(megabytes: u64) -> Self {
        let bytes = u128::from(megabytes.max(1)) * 1024 * 1024;
        let slot_size = size_of::<Option<TableEntry>>() as u128;
        let max_slots = (bytes / slot_size).min(usize::MAX as u128) as usize;
        let size = largest_power_of_two(max_slots.max(1));

        Self::new(size)
    }

    pub fn clear(&mut self) {
        self.table = vec![None; self.mask + 1];
    }

    pub fn get(&self, key: u64) -> Option<TableEntry> {
        if let Some(entry) = self.table[(key as usize) & self.mask] {
            if entry.key == key { Some(entry) } else { None }
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: u64, entry: TableEntry) {
        let k = (key as usize) & self.mask;

        if self.table[k].is_none()
            || entry.depth >= self.table[k].unwrap().depth
            || key != self.table[k].unwrap().key
        {
            self.table[k] = Some(entry);
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
        let bytes = table.table.len() * size_of::<Option<TableEntry>>();

        assert!(table.table.len().is_power_of_two());
        assert!(bytes <= 1024 * 1024);
    }
}
