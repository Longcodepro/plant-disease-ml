//! Rust High-Performance Inference API (Axum + ONNX Runtime + image)
//!
//! Giữ nguyên 100% hợp đồng API của FastAPI (`backend/main.py`):
//!   - GET  /health
//!   - POST /predict  (multipart/form-data, field `file`)
//! => Frontend Streamlit (`frontend/app.py`) kết nối trực tiếp không cần sửa code.
//!
//! Chạy từ thư mục gốc dự án:
//!   MODEL_PATH="models/convnext_tiny_khoi_int8.onnx" \
//!   MAPPING_PATH="models/class_indices.json" \
//!   cargo run --manifest-path backend/rust/Cargo.toml --release --bin plant-disease-api

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use image::imageops::FilterType;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env,
    fs,
    net::SocketAddr,
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};
use tower_http::cors::{Any, CorsLayer};

const IMAGE_SIZE: u32 = 224;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Deserialize)]
struct ClassMappingFile {
    idx_to_class: HashMap<String, String>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    model_loaded: bool,
    model_version: String,
}

#[derive(Serialize)]
struct PredictionItem {
    class_name: String,
    confidence: f32,
}

#[derive(Serialize)]
struct PredictionResponse {
    predicted_class: String,
    confidence: f32,
    top_3: Vec<PredictionItem>,
    inference_time_ms: f64,
    model_version: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    detail: String,
}

struct AppState {
    // Mutex đảm bảo an toàn luồng khi chạy đồng thời với mọi phiên bản ort
    session: Mutex<Session>,
    idx_to_class: HashMap<usize, String>,
    model_version: String,
}

/// Tiền xử lý ảnh chuẩn ImageNet: Decode -> RGB -> Resize 224x224 (Bilinear) -> NCHW Float32.
fn preprocess_image(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| format!("Không đọc được định dạng ảnh: {e}"))?
        .resize_exact(IMAGE_SIZE, IMAGE_SIZE, FilterType::Triangle)
        .to_rgb8();

    let hw = (IMAGE_SIZE * IMAGE_SIZE) as usize;
    let mut tensor = vec![0.0f32; 3 * hw];

    for (idx, pixel) in img.pixels().enumerate() {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;
        tensor[idx] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        tensor[hw + idx] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        tensor[2 * hw + idx] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }

    Ok(tensor)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn round4(v: f32) -> f32 {
    (v * 10_000.0).round() / 10_000.0
}

async fn health_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy",
        model_loaded: true,
        model_version: state.model_version.clone(),
    })
}

async fn predict(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut image_bytes: Option<Vec<u8>> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("file") {
            if let Some(ct) = field.content_type() {
                if !ct.starts_with("image/") {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!(ErrorResponse {
                            detail: "File tải lên bắt buộc phải là định dạng ảnh.".into()
                        })),
                    );
                }
            }
            match field.bytes().await {
                Ok(b) => {
                    image_bytes = Some(b.to_vec());
                    break;
                }
                Err(e) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!(ErrorResponse {
                            detail: format!("Lỗi đọc dữ liệu ảnh: {e}")
                        })),
                    );
                }
            }
        }
    }

    let bytes = match image_bytes {
        Some(b) if !b.is_empty() => b,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!(ErrorResponse {
                    detail: "Thiếu trường 'file' trong multipart/form-data.".into()
                })),
            );
        }
    };

    // Chạy decode + suy luận trong thread pool chặn (blocking) để không block async runtime
    let state_clone = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || -> Result<PredictionResponse, String> {
        let chw = preprocess_image(&bytes)?;
        let input_tensor = Tensor::from_array(([1usize, 3, IMAGE_SIZE as usize, IMAGE_SIZE as usize], chw))
            .map_err(|e| format!("Lỗi tạo ONNX tensor: {e}"))?;

        let start = Instant::now();
        let logits_vec: Vec<f32> = {
            let session = state_clone
                .session
                .lock()
                .map_err(|_| "Mutex bị lỗi khi gọi ONNX session".to_string())?;
            let outputs = session
                .run(ort::inputs![input_tensor].map_err(|e| format!("Lỗi inputs ONNX: {e}"))?)
                .map_err(|e| format!("Lỗi suy luận ONNX: {e}"))?;
            let (_, raw_logits) = outputs[0]
                .try_extract_raw_tensor::<f32>()
                .map_err(|e| format!("Lỗi trích xuất logits: {e}"))?;
            raw_logits.to_vec()
        };
        let latency_ms = (start.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0;

        let probs = softmax(&logits_vec);
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

        let top_3: Vec<PredictionItem> = indexed
            .into_iter()
            .take(3)
            .map(|(idx, prob)| PredictionItem {
                class_name: state_clone
                    .idx_to_class
                    .get(&idx)
                    .cloned()
                    .unwrap_or_else(|| format!("Class_{idx}")),
                confidence: round4(prob),
            })
            .collect();

        let best = top_3
            .first()
            .ok_or_else(|| "Model không trả về lớp nào.".to_string())?;

        Ok(PredictionResponse {
            predicted_class: best.class_name.clone(),
            confidence: best.confidence,
            top_3,
            inference_time_ms: latency_ms,
            model_version: state_clone.model_version.clone(),
        })
    })
    .await;

    match result {
        Ok(Ok(resp)) => (StatusCode::OK, Json(serde_json::json!(resp))),
        Ok(Err(msg)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!(ErrorResponse {
                detail: format!("Lỗi suy luận nội bộ: {msg}")
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!(ErrorResponse {
                detail: format!("Lỗi worker thread: {e}")
            })),
        ),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/convnext_tiny_khoi_int8.onnx".to_string());
    let mapping_path =
        env::var("MAPPING_PATH").unwrap_or_else(|_| "models/class_indices.json".to_string());
    let port: u16 = env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8000);

    // 1. Nạp bảng ánh xạ 38 lớp
    let mapping_text = fs::read_to_string(&mapping_path)
        .map_err(|e| format!("Không đọc được {mapping_path}: {e}"))?;
    let parsed: ClassMappingFile = serde_json::from_str(&mapping_text)?;
    let idx_to_class: HashMap<usize, String> = parsed
        .idx_to_class
        .into_iter()
        .filter_map(|(k, v)| k.parse::<usize>().ok().map(|idx| (idx, v)))
        .collect();

    // 2. Nạp ONNX Session với GraphOptimizationLevel::Level3
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(num_threads)?
        .commit_from_file(&model_path)?;

    let stem = Path::new(&model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("onnx_model");
    let model_version = format!("{stem} (Rust + ONNX)");

    println!("[OK] Rust API đã nạp model: {model_path} ({} lớp, {num_threads} threads)", idx_to_class.len());

    let state = Arc::new(AppState {
        session: Mutex::new(session),
        idx_to_class,
        model_version,
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/predict", post(predict))
        .layer(cors)
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("🚀 Rust Inference API đang chạy tại http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
