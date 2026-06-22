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
    /// Path to the .pcd file to visualize
    file: PathBuf,

    /// Show the coordinate origin as an RGB axis frame (X=red, Y=green, Z=blue)
    #[arg(long)]
    show_origin: bool,
}

fn main() {
    let args = Args::parse();

    println!("Loading {} ...", args.file.display());
    let cloud = match reader::read_pcd(&args.file) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    println!(
        "Loaded {} points ({}x{}, {})",
        cloud.len(),
        cloud.width(),
        cloud.height(),
        if cloud.is_organized() { "organized" } else { "unorganized" }
    );

    // Launch viewer (blocks until window closes)
    render::run(cloud, args.show_origin);
}
