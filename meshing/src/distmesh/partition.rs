use std::ops::Range;

use crate::error::{MeshError, MeshResult};

pub(super) fn by_budget(
    elements: usize,
    byte_budget: usize,
    row_limit: usize,
    max_tiles: usize,
    mut measure: impl FnMut(Range<usize>) -> MeshResult<(usize, usize)>,
) -> MeshResult<Vec<Range<usize>>> {
    if max_tiles == 0 {
        return Err(MeshError::LimitExceeded(
            "DistMesh exhausted the configured chunk limit".into(),
        ));
    }
    let mut tiles = Vec::with_capacity(1);
    tiles.push(0..elements);
    let mut index = 0;
    while index < tiles.len() {
        let tile = tiles[index].clone();
        let (bytes, rows) = measure(tile.clone())?;
        if bytes <= byte_budget && rows <= row_limit {
            index += 1;
            continue;
        }
        if tile.len() == 1 {
            return Err(MeshError::LimitExceeded(format!(
                "one indivisible DistMesh element requires {bytes} bytes and {rows} rows, exceeding the {byte_budget} byte chunk target or {row_limit} row limit"
            )));
        }
        if tiles.len() >= max_tiles {
            return Err(MeshError::LimitExceeded(format!(
                "DistMesh partitioning exhausted the configured {max_tiles} remaining chunks"
            )));
        }
        let middle = tile.start + tile.len() / 2;
        tiles.splice(index..=index, [tile.start..middle, middle..tile.end]);
    }
    Ok(tiles)
}
