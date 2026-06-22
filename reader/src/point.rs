//! Point cloud data types.

/// A single point in 3D space.
#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub(super) x: f32,
    pub(super) y: f32,
    pub(super) z: f32,
    /// Optional RGB packed as a single f32 (0xRRGGBB encoding).
    pub(super) rgb: Option<f32>,
}

impl Point {
    /// Create a new point.
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, rgb: None }
    }

    /// Create a new point with RGB color.
    pub fn new_with_rgb(x: f32, y: f32, z: f32, rgb: f32) -> Self {
        Self { x, y, z, rgb: Some(rgb) }
    }

    /// Get the X coordinate.
    pub fn x(&self) -> f32 {
        self.x
    }

    /// Get the Y coordinate.
    pub fn y(&self) -> f32 {
        self.y
    }

    /// Get the Z coordinate.
    pub fn z(&self) -> f32 {
        self.z
    }

    /// Get the packed RGB value, if present.
    pub fn rgb(&self) -> Option<f32> {
        self.rgb
    }

    /// Get RGB as normalized [r, g, b] in [0.0, 1.0], or white if absent.
    pub fn rgb_normalized(&self) -> [f32; 3] {
        match self.rgb {
            Some(rgb) => {
                let r = ((rgb as u32) >> 16) & 0xFF;
                let g = ((rgb as u32) >> 8) & 0xFF;
                let b = (rgb as u32) & 0xFF;
                [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
            }
            None => [1.0, 1.0, 1.0],
        }
    }
}

/// A parsed point cloud from a `.pcd` file.
#[derive(Debug)]
pub struct PointCloud {
    points: Vec<Point>,
    width: u32,
    height: u32,
    /// Whether the cloud is organized (width * height == points.len()).
    is_organized: bool,
}

impl PointCloud {
    /// Create a new point cloud.
    pub fn new(points: Vec<Point>, width: u32, height: u32, is_organized: bool) -> Self {
        Self {
            points,
            width,
            height,
            is_organized,
        }
    }

    /// Iterate over all points.
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the cloud has no points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Cloud width (for organized clouds).
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Cloud height (for organized clouds).
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Whether the cloud is organized (grid-like structure).
    pub fn is_organized(&self) -> bool {
        self.is_organized
    }
}
