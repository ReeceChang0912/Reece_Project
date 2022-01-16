// lib.rs
mod utils;

use std::fmt;
use js_sys::Math;
use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Dead = 0,
    Alive = 1,
}

#[wasm_bindgen]
pub struct Universe {
    width: u32,
    height: u32,
    cells: Vec<Cell>,
}

#[wasm_bindgen]
impl Universe {
    pub fn new(len: u32) -> Universe {
        let width: u32 = len;
        let height: u32 = len;

        let cells = (0..width * height)
            .map(|_| {
                if Math::random() < 0.03 {
                    Cell::Alive
                } else {
                    Cell::Dead
                }
            })
            .collect();
        
            Universe {
                width,
                height,
                cells,
            }
    }

    pub fn render (&self) -> String {
        self.to_string()
    }

    fn get_index(&self, row: u32, column: u32) -> usize {
        (row * self.width + column) as usize
    }

    fn live_neighbor_count(&self, row: u32, column: u32) -> u8 {
        let mut count = 0;
        for delta_row in [self.height - 1, 0, 1].iter().cloned() {
            for delta_col in [self.width - 1, 0 ,1].iter().cloned() {
                if delta_row == 0 && delta_col == 0 {
                    continue;
                }
                
                let neighbor_x = (row + delta_row) % self.height;
                let neighbor_y = (column + delta_col) % self.width;
                count += self.cells[self.get_index(neighbor_x, neighbor_y)] as u8;
            }
        }
        count
    }
    
    pub fn next_tick(&mut self) {
        // 模拟下一个状态整个宇宙的state
        // 采用暴力遍历枚举
        let mut next = self.cells.clone();

        for x in 0..self.height {
            for y in 0..self.width {
                let idx = self.get_index(x, y);
                let cell = self.cells[idx];
                let live_neighbors_count = self.live_neighbor_count(x, y);

                let next_cell = match (cell, live_neighbors_count) {
                    // Rule 1: 当本身是存活状态，周围存活数量小于等于1
                    // 则本体死亡（数量稀少）
                    (Cell::Alive, count) if count <= 1 => Cell::Dead,
                    // Rule 2: 当本身是存活状态，周围存活数量小于等于3
                    // 则本体正常生存
                    (Cell::Alive, count) if count <= 3 => Cell::Alive,
                    // Rule 3: 当本身是存活状态，周围存活数量大于等于4
                    // 则本体死亡（数量过多）
                    (Cell::Alive, count) if count >= 4 => Cell::Dead,
                    // Rule 4: 当本身是死亡状态，周围存活数量大于等于3
                    // 则本体死亡
                    (Cell::Dead, count) if count >= 3 => Cell::Alive,
                    // 其他情况保持原样
                    (otherwise, _) => otherwise,
                };
                next[idx] = next_cell;
            }
        }

        self.cells = next;
    }

    pub fn width(&self) ->u32 {
        self.width
    }
    pub fn height(&self) ->u32 {
        self.height
    }
    pub fn cells(&self) ->*const Cell {
        self.cells.as_ptr()
    }
}

impl fmt::Display for Universe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for line in self.cells.as_slice().chunks(self.width as usize) {
            for &cell in line {
                let symbol = if cell == Cell::Dead {
                    '◻'
                } else {
                    '◼'
                };
                write!(f, "{}", symbol)?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}
