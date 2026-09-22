//! `pcdviewr` — CLI entry point.
//!
//! Usage:
//! ```ignore
//! pcdviewr file.pcd
//! ```

use clap::Parser;
use std::path::PathBuf;

/// A lightweight point cloud viewer for .pcd files.
#[derive(Parser, Debug)]
#[command(name = "pcdviewr", version, about)]
struct Args {
    /// Optional path to the initial .pcd file
    file: Option<PathBuf>,

    /// Show the coordinate origin as an RGB axis frame (X=red, Y=green, Z=blue)
    #[arg(long)]
    show_origin: bool,
}

fn main() {
    let args = Args::parse();

    let Some(file) = args.file else {
        render::run_optional(None, args.show_origin);
        return;
    };
    println!("Loading {} ...", file.display());
    let cloud = match reader::read_pcd(&file) {
        Ok(cloud) => cloud,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    };

    println!(
        "Loaded {} points ({}x{}, {})",
        cloud.len(),
        cloud.width(),
        cloud.height(),
        if cloud.is_organized() {
            "organized"
        } else {
            "unorganized"
        }
    );

    // Launch viewer (blocks until window closes)
    render::run(cloud, args.show_origin);
}
