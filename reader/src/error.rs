//! Errors that can occur during PCD file reading and parsing.

use std::io;
use thiserror::Error;

/// Errors that can occur during PCD parsing.
#[derive(Debug, Error)]
pub enum ReaderError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Unsupported PCD format: {0}")]
    UnsupportedFormat(String),

    #[error("Missing required header field: {0}")]
    MissingField(&'static str),

    #[error("Parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },

    #[error("Missing 'x', 'y', or 'z' in point fields")]
    MissingCoordinates,
}
