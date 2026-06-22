//! PCD file parser.
//!
//! Supports ASCII and binary_compressed PCD formats with `x y z` fields.
//! RGB and intensity fields are parsed but not yet used for rendering.

use std::collections::HashMap;
use std::ffi::c_uint;
use std::fs;
use std::path::Path;

use super::error::ReaderError;
use super::point::{Point, PointCloud};

/// Parsed header information from a PCD file.
struct PcdHeader {
    data_format: String,
    width: u32,
    height: u32,
    field_names: Vec<String>,
    sizes: Vec<usize>,
    types: Vec<String>,
    counts: Vec<usize>,
    data_start_line: usize,
}

/// Parse a `.pcd` file (ASCII or binary_compressed) and return a [`PointCloud`].
pub fn read_pcd(path: &Path) -> Result<PointCloud, ReaderError> {
    // For binary formats we read raw bytes; for ASCII we read as string.
    // We need the ASCII header either way, so we read the header as text first.
    let bytes = fs::read(path)?;

    // Extract header: find "DATA " marker, everything before is ASCII text.
    let data_marker = bytes.windows(5).position(|w| &w == b"DATA " || &w == b"data ");
    let header_end = match data_marker {
        Some(pos) => {
            // Find end of the DATA line
            let line_end = bytes[pos..]
                .iter()
                .position(|&b| b == b'\n')
                .map(|p| pos + p + 1)
                .unwrap_or(bytes.len());
            line_end
        }
        None => return Err(ReaderError::MissingField("DATA")),
    };

    let header_text =
        std::str::from_utf8(&bytes[..header_end]).map_err(|_| ReaderError::ParseError {
            line: 0,
            message: "Invalid UTF-8 in PCD header".to_string(),
        })?;

    let header = parse_header(header_text)?;

    // Check required coordinates
    if !header.field_names.contains(&"x".to_string())
        || !header.field_names.contains(&"y".to_string())
        || !header.field_names.contains(&"z".to_string())
    {
        return Err(ReaderError::MissingCoordinates);
    }

    let points = match header.data_format.as_str() {
        "ascii" => {
            let content =
                std::str::from_utf8(&bytes).map_err(|_| ReaderError::ParseError {
                    line: 0,
                    message: "Invalid UTF-8 in PCD data".to_string(),
                })?;
            parse_ascii_points(content, &header)?
        }
        "binary_compressed" => {
            let data = &bytes[header_end..];
            parse_binary_compressed(data, &header)?
        }
        other => return Err(ReaderError::UnsupportedFormat(other.to_string())),
    };

    let is_organized = header.height > 1;
    Ok(PointCloud::new(
        points,
        header.width,
        header.height,
        is_organized,
    ))
}

/// Parse PCD header lines into a [`PcdHeader`] struct.
fn parse_header(content: &str) -> Result<PcdHeader, ReaderError> {
    let mut header = HashMap::new();
    let mut data_start_line = 0;
    let mut in_header = true;

    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if in_header {
            if line.starts_with("DATA ") || line.starts_with("data ") {
                header.insert("data".to_string(), line[5..].trim().to_lowercase());
                data_start_line = i + 1;
                in_header = false;
            } else {
                let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();
                if parts.len() == 2 {
                    header.insert(parts[0].to_lowercase(), parts[1].trim().to_string());
                }
            }
        }
    }

    let data_format = header.get("data").ok_or(ReaderError::MissingField("DATA"))?.clone();

    let width: u32 = header
        .get("width")
        .ok_or(ReaderError::MissingField("WIDTH"))?
        .parse()
        .map_err(|_| ReaderError::ParseError {
            line: 0,
            message: "Invalid WIDTH value".to_string(),
        })?;

    let height: u32 = header
        .get("height")
        .ok_or(ReaderError::MissingField("HEIGHT"))?
        .parse()
        .map_err(|_| ReaderError::ParseError {
            line: 0,
            message: "Invalid HEIGHT value".to_string(),
        })?;

    let fields = header.get("fields").ok_or(ReaderError::MissingField("FIELDS"))?;
    let field_names: Vec<String> = fields.split_whitespace().map(String::from).collect();

    let sizes: Vec<usize> = header
        .get("size")
        .ok_or(ReaderError::MissingField("SIZE"))?
        .split_whitespace()
        .map(|s| {
            s.parse().map_err(|_| ReaderError::ParseError {
                line: 0,
                message: format!("Invalid SIZE value: {}", s),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let types: Vec<String> = header
        .get("type")
        .ok_or(ReaderError::MissingField("TYPE"))?
        .split_whitespace()
        .map(String::from)
        .collect();

    let counts: Vec<usize> = header
        .get("count")
        .ok_or(ReaderError::MissingField("COUNT"))?
        .split_whitespace()
        .map(|c| {
            c.parse().map_err(|_| ReaderError::ParseError {
                line: 0,
                message: format!("Invalid COUNT value: {}", c),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PcdHeader {
        data_format,
        width,
        height,
        field_names,
        sizes,
        types,
        counts,
        data_start_line,
    })
}

/// Parse field value bytes into an `f32` based on PCD type/size/count.
fn parse_field(bytes: &[u8], size: usize, typ: &str, count: usize) -> Option<f32> {
    if count != 1 {
        return None;
    }
    match (typ, size) {
        ("F", 4) => Some(f32::from_le_bytes(bytes[0..4].try_into().ok()?)),
        ("I", 1) => Some(i8::from_le_bytes([bytes[0]]) as f32),
        ("I", 2) => Some(i16::from_le_bytes(bytes[0..2].try_into().ok()?) as f32),
        ("I", 4) => Some(i32::from_le_bytes(bytes[0..4].try_into().ok()?) as f32),
        ("I", 8) => Some(i64::from_le_bytes(bytes[0..8].try_into().ok()?) as f32),
        ("U", 1) => Some(u8::from_le_bytes([bytes[0]]) as f32),
        ("U", 2) => Some(u16::from_le_bytes(bytes[0..2].try_into().ok()?) as f32),
        ("U", 4) => Some(u32::from_le_bytes(bytes[0..4].try_into().ok()?) as f32),
        ("U", 8) => Some(u64::from_le_bytes(bytes[0..8].try_into().ok()?) as f32),
        _ => None,
    }
}

/// Parse ASCII point data using a pre-parsed header.
fn parse_ascii_points(content: &str, header: &PcdHeader) -> Result<Vec<Point>, ReaderError> {
    let x_idx = header.field_names.iter().position(|f| f == "x").unwrap();
    let y_idx = header.field_names.iter().position(|f| f == "y").unwrap();
    let z_idx = header.field_names.iter().position(|f| f == "z").unwrap();
    let rgb_idx = header.field_names.iter().position(|f| f == "rgb");

    let point_count = header.width * header.height;
    let mut points = Vec::with_capacity(point_count as usize);

    for (i, line) in content.lines().enumerate() {
        if i < header.data_start_line {
            continue;
        }
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let values: Vec<f32> = line
            .split_whitespace()
            .map(|v| {
                v.parse::<f32>().map_err(|_| ReaderError::ParseError {
                    line: i + 1,
                    message: format!("Invalid float value: {}", v),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        if values.len() < header.field_names.len() {
            return Err(ReaderError::ParseError {
                line: i + 1,
                message: format!(
                    "Expected {} values, got {}",
                    header.field_names.len(),
                    values.len()
                ),
            });
        }

        points.push(Point {
            x: values[x_idx],
            y: values[y_idx],
            z: values[z_idx],
            rgb: rgb_idx.map(|idx| values[idx]),
        });

        if points.len() >= point_count as usize {
            break;
        }
    }

    Ok(points)
}

/// Read raw binary point data and build points.
fn parse_binary_points(
    data: &[u8],
    header: &PcdHeader,
) -> Result<Vec<Point>, ReaderError> {
    // Compute point step (bytes per point) from header
    let mut point_step = 0;
    for i in 0..header.field_names.len() {
        point_step += header.sizes[i] * header.counts[i];
    }

    let x_idx = header
        .field_names
        .iter()
        .position(|f| f == "x")
        .ok_or(ReaderError::MissingCoordinates)?;
    let y_idx = header
        .field_names
        .iter()
        .position(|f| f == "y")
        .ok_or(ReaderError::MissingCoordinates)?;
    let z_idx = header
        .field_names
        .iter()
        .position(|f| f == "z")
        .ok_or(ReaderError::MissingCoordinates)?;
    let rgb_idx = header.field_names.iter().position(|f| f == "rgb");

    let point_count = header.width * header.height;
    let mut points = Vec::with_capacity(point_count as usize);

    let mut offset = 0;
    for _ in 0..point_count {
        if offset + point_step > data.len() {
            break;
        }

        // Walk fields to find x, y, z, rgb offsets within this point
        let mut field_offset = 0;
        let mut px = 0.0;
        let mut py = 0.0;
        let mut pz = 0.0;
        let mut prgb = None;

        for (i, _name) in header.field_names.iter().enumerate() {
            let field_size = header.sizes[i] * header.counts[i];
            let field_bytes =
                &data[offset + field_offset..offset + field_offset + field_size];

            if i == x_idx {
                px = parse_field(
                    field_bytes,
                    header.sizes[i],
                    &header.types[i],
                    header.counts[i],
                )
                .unwrap_or(0.0);
            } else if i == y_idx {
                py = parse_field(
                    field_bytes,
                    header.sizes[i],
                    &header.types[i],
                    header.counts[i],
                )
                .unwrap_or(0.0);
            } else if i == z_idx {
                pz = parse_field(
                    field_bytes,
                    header.sizes[i],
                    &header.types[i],
                    header.counts[i],
                )
                .unwrap_or(0.0);
            } else if Some(i) == rgb_idx {
                prgb = parse_field(
                    field_bytes,
                    header.sizes[i],
                    &header.types[i],
                    header.counts[i],
                );
            }

            field_offset += field_size;
        }

        points.push(Point {
            x: px,
            y: py,
            z: pz,
            rgb: prgb,
        });
        offset += point_step;
    }

    Ok(points)
}

/// Parse binary_compressed PCD data (LZF-compressed, structure-of-arrays layout).
///
/// Per the PCD spec:
///   bytes 0..4  = compressed size (u32 LE)
///   bytes 4..8  = uncompressed size (u32 LE)
///   bytes 8..   = LZF-compressed payload
///
/// After decompression the data is in structure-of-arrays order:
///   [all field_0 values][all field_1 values]...[all field_N values]
fn parse_binary_compressed(data: &[u8], header: &PcdHeader) -> Result<Vec<Point>, ReaderError> {
    if data.len() < 8 {
        return Err(ReaderError::ParseError {
            line: 0,
            message: "binary_compressed data too short".to_string(),
        });
    }

    // Per spec: first u32 = compressed size, second u32 = uncompressed size
    let compressed_size = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let uncompressed_size = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;

    if data.len() < 8 + compressed_size {
        return Err(ReaderError::ParseError {
            line: 0,
            message: format!(
                "File truncated: need {} compressed bytes, have {}",
                compressed_size,
                data.len() - 8
            ),
        });
    }

    let compressed = &data[8..8 + compressed_size];
    let mut decompressed = vec![0u8; uncompressed_size];

    let byte_size = unsafe {
        lzf_sys::lzf_decompress(
            compressed.as_ptr().cast(),
            compressed_size as c_uint,
            decompressed.as_mut_ptr().cast(),
            uncompressed_size as c_uint,
        )
    } as usize;

    if byte_size == 0 {
        return Err(ReaderError::ParseError {
            line: 0,
            message: "LZF decompression failed".to_string(),
        });
    }
    if byte_size != uncompressed_size {
        return Err(ReaderError::ParseError {
            line: 0,
            message: format!(
                "Wrong decompressed size: expected {}, got {}",
                uncompressed_size, byte_size
            ),
        });
    }

    let point_count = (header.width * header.height) as usize;
    eprintln!(
        "binary_compressed: {} points, compressed {} -> {} bytes",
        point_count, compressed_size, uncompressed_size
    );

    // Decompressed data is structure-of-arrays: all values for field 0,
    // then all values for field 1, etc.
    parse_soa_points(&decompressed, header, point_count)
}

/// Parse structure-of-arrays binary data into points.
///
/// Layout: [field_0[0..N]][field_1[0..N]]...[field_k[0..N]]
fn parse_soa_points(
    data: &[u8],
    header: &PcdHeader,
    point_count: usize,
) -> Result<Vec<Point>, ReaderError> {
    let x_idx = header
        .field_names
        .iter()
        .position(|f| f == "x")
        .ok_or(ReaderError::MissingCoordinates)?;
    let y_idx = header
        .field_names
        .iter()
        .position(|f| f == "y")
        .ok_or(ReaderError::MissingCoordinates)?;
    let z_idx = header
        .field_names
        .iter()
        .position(|f| f == "z")
        .ok_or(ReaderError::MissingCoordinates)?;
    let rgb_idx = header.field_names.iter().position(|f| f == "rgb");

    // Compute byte offset where each field's column starts in the SoA buffer
    let mut field_offsets = vec![0usize; header.field_names.len()];
    let mut running = 0usize;
    for i in 0..header.field_names.len() {
        field_offsets[i] = running;
        running += header.sizes[i] * header.counts[i] * point_count;
    }

    let mut points = Vec::with_capacity(point_count);
    for p in 0..point_count {
        let read_f32 = |field_idx: usize| -> f32 {
            let elem_size = header.sizes[field_idx] * header.counts[field_idx];
            let start = field_offsets[field_idx] + p * elem_size;
            let end = start + elem_size;
            if end > data.len() {
                return 0.0;
            }
            parse_field(
                &data[start..end],
                header.sizes[field_idx],
                &header.types[field_idx],
                header.counts[field_idx],
            )
            .unwrap_or(0.0)
        };

        let rgb = rgb_idx.map(|ri| {
            let elem_size = header.sizes[ri] * header.counts[ri];
            let start = field_offsets[ri] + p * elem_size;
            let end = start + elem_size;
            if end > data.len() {
                return 0.0;
            }
            parse_field(
                &data[start..end],
                header.sizes[ri],
                &header.types[ri],
                header.counts[ri],
            )
            .unwrap_or(0.0)
        });

        points.push(Point {
            x: read_f32(x_idx),
            y: read_f32(y_idx),
            z: read_f32(z_idx),
            rgb,
        });
    }

    Ok(points)
}
