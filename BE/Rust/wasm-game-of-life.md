在wasm与rust结合的模拟生命的实例中有那么一段函数比较难理解



```rust
impl Universe {
    // ...

    fn live_neighbor_count(&self, row: u32, column: u32) -> u8 {
        let mut count = 0;
        for delta_row in [self.height - 1, 0, 1].iter().cloned() {
            for delta_col in [self.width - 1, 0, 1].iter().cloned() {
                if delta_row == 0 && delta_col == 0 {
                    continue;
                }

                let neighbor_row = (row + delta_row) % self.height;
                let neighbor_col = (column + delta_col) % self.width;
                let idx = self.get_index(neighbor_row, neighbor_col);
                count += self.cells[idx] as u8;
            }
        }
        count
    }
}
```

The `live_neighbor_count` method uses deltas and modulo to avoid special casing the edges of the universe with `if`s. When applying a delta of `-1`, we *add* `self.height - 1` and let the modulo do its thing, rather than attempting to subtract `1`. `row` and `column` can be `0`, and if we attempted to subtract `1` from them, there would be an unsigned integer underflow.



我的理解如下：

![universe](D:\Reece_Project\BE\Rust\universe.jpg)



最终的效果如下：

![image-20220116213816159](D:\Reece_Project\BE\Rust\cells.png)

核心：

```
	import {Universe, Cell} from "wasm-game-of-life";
	import {memory} from "wasm-game-of-	
    life/wasm_game_of_life_bg";   
    
	const universe = Universe.new(100);
	const width = universe.width();
	const height = universe.height()
	
```

通过wasm-pack编译之后，我们的rust变成了一个node 包 我么你直接从调用那个模块就ok

具体的编译过程目前只记得胶水代码 等我深究之后再发博客