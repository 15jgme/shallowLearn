use std::{
    str::FromStr,
    sync::{mpsc::Sender, Arc},
    thread::{self, JoinHandle},
    time::Duration,
};

use parking_lot::{RawRwLock, RwLock};
use pyo3::types::PyString;
use pyo3::{exceptions::PyValueError, prelude::*};
use shallow_red_engine::{
    engine::enter_engine,
    evaluation::{self, evaluate_board},
    managers::cache_manager::{Cache, CacheEntry, CacheInputGrouping},
    utils::engine_interface::EngineSettings,
    *,
};

use chess::Board;

#[pyfunction]
fn fen_eval(board_fen: String) -> PyResult<i16> {
    // Runs the evaluation function on a board fen, and returns the score for the colour to move (used for rewards)
    let board = Board::from_str(board_fen.as_str()).expect("Board fen should be valid");
    let board_eval = evaluation::evaluate_board(board);
    // let board_eval = evaluate_board(board, Some(|K:usize, Q:usize, R:usize, B:usize, N:usize, P:usize, k:usize, q:usize, r:usize, b:usize, n:usize, p:usize| 0));
    Ok(board_eval.score) // + is winning for the current side to move
}

/// A Python module implemented in Rust.
#[pymodule]
fn shallowred_interface(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fen_eval, m)?)?;
    Ok(())
}
