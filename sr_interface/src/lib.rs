use std::str::FromStr;
use pyo3::prelude::*;
use shallow_red_engine::{self, evaluation};
use chess::Board;

#[pyfunction]
fn fen_eval(board_fen: String) -> PyResult<i16> {
    // Runs the evaluation function on a board fen, and returns the score for the colour to move
    let board = Board::from_str(board_fen.as_str()).expect("Board fen should be valid");
    let board_eval = evaluation::evaluate_board(board);
    Ok(board_eval.score) // + is winning for the current side to move
}

#[pyfunction]
fn fen_eval_material(board_fen: String) -> PyResult<i16> {
    // Runs the evaluation function on a board fen, and returns the score for the colour to move
    let board = Board::from_str(board_fen.as_str()).expect("Board fen should be valid");
    let board_eval = evaluation::evaluate_board_material(&board);
    Ok(board_eval.score) // + is winning for the current side to move
}

#[pymodule]
fn sr_interface(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(fen_eval, m)?)?;    
    m.add_function(wrap_pyfunction!(fen_eval_material, m)?)?;   
    Ok(())
}