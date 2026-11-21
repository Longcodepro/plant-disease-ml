//! Rust Parallel Dataset Integrity & Data-Leakage Auditor (Rayon + SHA-256 + image)
//!
//! Thay thế bước quét đơn luồng chậm bằng Python trên 87.867 ảnh:
//!   1. Kiểm tra toàn bộ ảnh hỏng (corrupted files) trên tất cả CPU cores.
//!   2. Tính SHA-256 của từng ảnh để phát hiện:
//!      - Ảnh trùng lặp nội bộ trong tập train.
//!      - Rò rỉ dữ liệu tuyệt đối (exact leakage): ảnh giống hệt nhau nằm ở cả train & valid.
//!   3. Xuất báo cáo JSON ra `notebooks/eda_reports/rust_data_audit.json`.
//!
//! Chạy từ thư mục gốc dự án:
//!   cargo run --manifest-path backend/rust/Cargo.toml --release --bin data-audit -- --data-dir data

use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};
use walkdir::WalkDir;

#[derive(Serialize)]
struct AuditReport {
    elapsed_seconds: f64,
    train_images: usize,
    valid_images: usize,
    corrupted_count: usize,
    corrupted_files: Vec<String>,
    train_internal_duplicates: usize,
    train_valid_leakage_hashes: usize,
    resolutions: Vec<String>,
}

struct FileAudit {
    path: String,
    corrupted: bool,
    sha256: Option<String>,
    resolution: Option<(u32, u32)>,
}

fn collect_images(dir: &Path) -> Vec<PathBuf> {
    if !dir.exists() {
        return Vec::new();
    }
    WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| {
            p.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                .unwrap_or(false)
        })
        .collect()
}

fn audit_one(path: &Path) -> FileAudit {
    let path_str = path.to_string_lossy().into_owned();
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(_) => {
            return FileAudit {
                path: path_str,
                corrupted: true,
                sha256: None,
                resolution: None,
            }
        }
    };

    let digest = format!("{:x}", Sha256::digest(&bytes));
    match image::load_from_memory(&bytes) {
        Ok(img) => FileAudit {
            path: path_str,
            corrupted: false,
            sha256: Some(digest),
            resolution: Some((img.width(), img.height())),
        },
        Err(_) => FileAudit {
            path: path_str,
            corrupted: true,
            sha256: Some(digest),
            resolution: None,
        },
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let data_dir = args
        .windows(2)
        .find(|w| w[0] == "--data-dir")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| PathBuf::from("data"));
    let out_file = args
        .windows(2)
        .find(|w| w[0] == "--out")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| PathBuf::from("notebooks/eda_reports/rust_data_audit.json"));

    let t0 = Instant::now();
    let train_files = collect_images(&data_dir.join("train"));
    let valid_files = collect_images(&data_dir.join("valid"));

    if train_files.is_empty() && valid_files.is_empty() {
        eprintln!("❌ Không tìm thấy ảnh trong {}/train hoặc {}/valid", data_dir.display(), data_dir.display());
        std::process::exit(1);
    }

    println!(
        "🔍 [Rust Rayon] Quét song song {} ảnh train + {} ảnh valid...",
        train_files.len(),
        valid_files.len()
    );

    let train_results: Vec<FileAudit> = train_files.par_iter().map(|p| audit_one(p)).collect();
    let valid_results: Vec<FileAudit> = valid_files.par_iter().map(|p| audit_one(p)).collect();

    let mut corrupted_files = Vec::new();
    let mut resolutions: HashSet<(u32, u32)> = HashSet::new();
    let mut train_hash_counts: HashMap<String, usize> = HashMap::new();
    let mut valid_hashes: HashSet<String> = HashSet::new();

    for r in &train_results {
        if r.corrupted {
            corrupted_files.push(r.path.clone());
        }
        if let Some(res) = r.resolution {
            resolutions.insert(res);
        }
        if let Some(h) = &r.sha256 {
            *train_hash_counts.entry(h.clone()).or_insert(0) += 1;
        }
    }

    for r in &valid_results {
        if r.corrupted {
            corrupted_files.push(r.path.clone());
        }
        if let Some(res) = r.resolution {
            resolutions.insert(res);
        }
        if let Some(h) = &r.sha256 {
            valid_hashes.insert(h.clone());
        }
    }

    let train_internal_duplicates: usize = train_hash_counts
        .values()
        .filter(|&&c| c > 1)
        .map(|&c| c - 1)
        .sum();

    let train_valid_leakage_hashes = valid_hashes
        .iter()
        .filter(|h| train_hash_counts.contains_key(*h))
        .count();

    let mut res_strings: Vec<String> = resolutions
        .into_iter()
        .map(|(w, h)| format!("{w}x{h}"))
        .collect();
    res_strings.sort();

    let report = AuditReport {
        elapsed_seconds: (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
        train_images: train_results.len(),
        valid_images: valid_results.len(),
        corrupted_count: corrupted_files.len(),
        corrupted_files,
        train_internal_duplicates,
        train_valid_leakage_hashes,
        resolutions: res_strings,
    };

    if let Some(parent) = out_file.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_file, serde_json::to_string_pretty(&report)?)?;

    println!("\n============================================================");
    println!(" 🦀 KẾT QUẢ KIỂM ĐỊNH DỮ LIỆU BẰNG RUST (RAYON + SHA-256)");
    println!("============================================================");
    println!(" ⏱️  Thời gian quét         : {:.2} giây", report.elapsed_seconds);
    println!(" 📁 Train / Valid          : {} / {} ảnh", report.train_images, report.valid_images);
    println!(" ⚠️  Ảnh hỏng (Corrupted)   : {}", report.corrupted_count);
    println!(" 🔁 Trùng lặp trong Train  : {} ảnh", report.train_internal_duplicates);
    println!(" 🚨 Rò rỉ Train ↔ Valid    : {} hash trùng nhau", report.train_valid_leakage_hashes);
    println!(" 📄 Báo cáo đã lưu tại     : {}", out_file.display());
    println!("============================================================");

    Ok(())
}
