import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Download,
  ImagePlus,
  LayoutGrid,
  List,
  Search,
  ShieldAlert,
  Sparkles,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";
import { predictBatchWithProgress, type BatchProgressState } from "../services/api";
import type { BatchApiResponse, BatchResultItem } from "../types/api";
import {
  escapeCsvCell,
  formatConfidence,
  formatMs,
  getSeverityMeta,
} from "../utils/format";

type BatchFilter = "all" | "diseased" | "healthy" | "uncertain";
type ViewLayout = "grid" | "table";

const MAX_BATCH_FILES = 32;

export function BatchView() {
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState<BatchProgressState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [batchResult, setBatchResult] = useState<BatchApiResponse | null>(null);
  const [previewMap, setPreviewMap] = useState<Record<string, string>>({});

  const [searchQuery, setSearchQuery] = useState("");
  const [filter, setFilter] = useState<BatchFilter>("all");
  const [viewLayout, setViewLayout] = useState<ViewLayout>("grid");
  const [activeDetail, setActiveDetail] = useState<BatchResultItem | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const previewUrlsRef = useRef<string[]>([]);

  const revokePreviews = () => {
    for (const url of previewUrlsRef.current) {
      URL.revokeObjectURL(url);
    }
    previewUrlsRef.current = [];
  };

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      revokePreviews();
    };
  }, []);

  const handleFilesSelected = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;

    const imageFiles = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
    if (imageFiles.length === 0) {
      setError("Không tìm thấy tệp ảnh hợp lệ (JPG, PNG, WEBP).");
      return;
    }

    let cappedFiles = imageFiles;
    if (imageFiles.length > MAX_BATCH_FILES) {
      cappedFiles = imageFiles.slice(0, MAX_BATCH_FILES);
      setNotice(
        `Đã chọn ${imageFiles.length} ảnh — hệ thống tự động giới hạn tối đa ${MAX_BATCH_FILES} ảnh mỗi lô.`,
      );
    } else {
      setNotice(null);
    }

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    revokePreviews();
    const nextPreviews: Record<string, string> = {};
    for (const file of cappedFiles) {
      const url = URL.createObjectURL(file);
      previewUrlsRef.current.push(url);
      nextPreviews[file.name] = url;
    }
    setPreviewMap(nextPreviews);
    setBatchResult(null);
    setError(null);
    setActiveDetail(null);
    setIsRunning(true);
    setProgress({ phase: "uploading", percent: 0, loadedBytes: 0, totalBytes: 1 });

    try {
      const data = await predictBatchWithProgress(
        cappedFiles,
        (state) => setProgress(state),
        controller.signal,
      );
      setBatchResult(data);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof Error ? err.message : "Chẩn đoán hàng loạt thất bại.");
    } finally {
      if (abortControllerRef.current === controller) {
        setIsRunning(false);
      }
    }
  };

  const stats = useMemo(() => {
    if (!batchResult) {
      return { total: 0, diseased: 0, healthy: 0, uncertain: 0, diseasedRate: 0 };
    }
    const items = batchResult.results;
    const diseased = items.filter((i) => !i.is_healthy).length;
    const healthy = items.filter((i) => i.is_healthy).length;
    const uncertain = items.filter((i) => i.is_uncertain).length;
    const diseasedRate = items.length > 0 ? (diseased / items.length) * 100 : 0;
    return { total: items.length, diseased, healthy, uncertain, diseasedRate };
  }, [batchResult]);

  const filteredResults = useMemo(() => {
    if (!batchResult) return [];
    const q = searchQuery.trim().toLowerCase();
    return batchResult.results.filter((item) => {
      if (filter === "diseased" && item.is_healthy) return false;
      if (filter === "healthy" && !item.is_healthy) return false;
      if (filter === "uncertain" && !item.is_uncertain) return false;

      if (!q) return true;
      return (
        item.filename.toLowerCase().includes(q) ||
        item.plant.toLowerCase().includes(q) ||
        item.condition.toLowerCase().includes(q) ||
        item.predicted_class.toLowerCase().includes(q)
      );
    });
  }, [batchResult, filter, searchQuery]);

  const handleExportCsv = () => {
    if (!batchResult || filteredResults.length === 0) return;

    const headers = [
      "Tệp ảnh",
      "Cây trồng",
      "Tình trạng / Bệnh",
      "Trạng thái",
      "Độ tin cậy (%)",
      "Mức độ",
      "Cần xem lại",
      "Khuyến nghị tóm tắt",
    ];

    const rows = filteredResults.map((item) => [
      item.filename,
      item.plant,
      item.condition,
      item.is_healthy ? "Khỏe mạnh" : "Có bệnh",
      (item.confidence > 1 ? item.confidence : item.confidence * 100).toFixed(2),
      item.advisory.severity,
      item.is_uncertain ? "Có" : "Không",
      item.advisory.summary,
    ]);

    const csvContent = [
      headers.map(escapeCsvCell).join(","),
      ...rows.map((r) => r.map(escapeCsvCell).join(",")),
    ].join("\r\n");

    // Prefix UTF-8 BOM (\uFEFF) so Vietnamese accents display cleanly in Excel
    const blob = new Blob(["\uFEFF" + csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `plantguard-batch-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleReset = () => {
    abortControllerRef.current?.abort();
    revokePreviews();
    setPreviewMap({});
    setBatchResult(null);
    setProgress(null);
    setError(null);
    setNotice(null);
    setSearchQuery("");
    setFilter("all");
    setActiveDetail(null);
  };

  return (
    <div className="space-y-6">
      {/* Multi-Image Dropzone */}
      <section
        onDragOver={(e) => {
          e.preventDefault();
          setIsDraggingOver(true);
        }}
        onDragLeave={() => setIsDraggingOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDraggingOver(false);
          void handleFilesSelected(e.dataTransfer.files);
        }}
        className={`rounded-2xl border-2 border-dashed p-6 transition-all sm:p-8 ${
          isDraggingOver
            ? "border-emerald-500 bg-emerald-50/70"
            : "border-slate-300/90 bg-white hover:border-emerald-400"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => {
            void handleFilesSelected(e.target.files);
            e.target.value = "";
          }}
        />

        <div className="flex flex-col items-center justify-between gap-4 sm:flex-row">
          <div className="flex items-center gap-4">
            <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-emerald-50 text-emerald-600">
              <UploadCloud className="h-7 w-7" />
            </div>
            <div>
              <h1 className="text-base font-extrabold text-slate-900 sm:text-lg">
                Chẩn đoán hàng loạt (Tối đa {MAX_BATCH_FILES} ảnh)
              </h1>
              <p className="mt-0.5 text-xs text-slate-500 sm:text-sm">
                Kéo thả nhiều ảnh lá cây cùng lúc để phân tích tổng hợp và xuất báo cáo CSV.
              </p>
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-2.5">
            {batchResult && !isRunning && (
              <button
                type="button"
                onClick={handleReset}
                className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 transition-colors hover:bg-slate-50"
              >
                <Trash2 className="h-3.5 w-3.5" />
                Xóa kết quả
              </button>
            )}
            <button
              type="button"
              disabled={isRunning}
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-2 rounded-xl bg-emerald-600 px-4 py-2.5 text-xs font-bold text-white shadow-sm shadow-emerald-600/20 transition-all hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-60 sm:text-sm"
            >
              <ImagePlus className="h-4 w-4" />
              <span>{batchResult ? "Chọn lô ảnh mới" : "Chọn nhiều ảnh"}</span>
            </button>
          </div>
        </div>

        {/* Real Upload + Inference Progress Bar */}
        {isRunning && progress && (
          <div className="mt-5 rounded-xl border border-emerald-200 bg-emerald-50/60 p-4">
            <div className="mb-2 flex items-center justify-between text-xs font-semibold">
              <span className="flex items-center gap-2 text-emerald-900">
                <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-600" />
                {progress.phase === "uploading"
                  ? "Đang tải tệp ảnh lên máy chủ..."
                  : "Đã tải xong — Mô hình AI đang suy luận hàng loạt..."}
              </span>
              <span className="font-mono font-bold text-emerald-700">
                {progress.phase === "uploading" ? `${progress.percent}%` : "Đang xử lý"}
              </span>
            </div>
            <div className="h-2.5 w-full overflow-hidden rounded-full bg-emerald-100">
              <div
                style={{ width: `${progress.percent}%` }}
                className={`h-full rounded-full bg-emerald-600 transition-all duration-200 ${
                  progress.phase === "processing" ? "animate-pulse" : ""
                }`}
              />
            </div>
          </div>
        )}
      </section>

      {notice && (
        <div className="flex items-center gap-2.5 rounded-xl border border-amber-200 bg-amber-50 px-4 py-2.5 text-xs font-medium text-amber-800">
          <AlertTriangle className="h-4 w-4 shrink-0 text-amber-600" />
          <span>{notice}</span>
        </div>
      )}

      {error && (
        <div
          role="alert"
          className="flex items-start gap-3 rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800"
        >
          <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-rose-600" />
          <div>
            <p className="font-bold">Lỗi xử lý lô ảnh</p>
            <p className="mt-0.5 text-xs text-rose-700">{error}</p>
          </div>
        </div>
      )}

      {/* Results Section */}
      {batchResult && (
        <>
          {/* Summary KPI Strip */}
          <section className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
            <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="text-xs font-semibold text-slate-500">Tổng số ảnh</span>
              <p className="mt-1 font-mono text-2xl font-bold text-slate-900">
                {batchResult.count}
              </p>
              <p className="mt-0.5 text-[11px] text-slate-400">
                Phiên bản {batchResult.model_version}
              </p>
            </div>
            <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="flex items-center gap-1 text-xs font-semibold text-slate-500">
                <Clock className="h-3.5 w-3.5" />
                Tổng thời gian
              </span>
              <p className="mt-1 font-mono text-2xl font-bold text-slate-900">
                {formatMs(batchResult.total_inference_time_ms)}
              </p>
              <p className="mt-0.5 text-[11px] text-slate-400">Toàn bộ lô suy luận</p>
            </div>
            <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="text-xs font-semibold text-slate-500">Trung bình / ảnh</span>
              <p className="mt-1 font-mono text-2xl font-bold text-emerald-700">
                {formatMs(batchResult.avg_time_per_image_ms)}
              </p>
              <p className="mt-0.5 text-[11px] text-slate-400">Tốc độ xử lý trung bình</p>
            </div>
            <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="text-xs font-semibold text-slate-500">Tỷ lệ phát hiện bệnh</span>
              <p className="mt-1 font-mono text-2xl font-bold text-rose-600">
                {stats.diseasedRate.toFixed(1)}%
              </p>
              <p className="mt-0.5 font-mono text-[11px] text-slate-500">
                {stats.diseased} bệnh · {stats.healthy} khỏe
              </p>
            </div>
          </section>

          {/* Filter & Action Toolbar */}
          <section className="flex flex-col gap-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm lg:flex-row lg:items-center lg:justify-between">
            <div className="flex flex-wrap items-center gap-2">
              {(
                [
                  { id: "all", label: `Tất cả (${stats.total})` },
                  { id: "diseased", label: `Có bệnh (${stats.diseased})` },
                  { id: "healthy", label: `Khỏe mạnh (${stats.healthy})` },
                  { id: "uncertain", label: `Cần xem lại (${stats.uncertain})` },
                ] as Array<{ id: BatchFilter; label: string }>
              ).map((pill) => (
                <button
                  key={pill.id}
                  type="button"
                  onClick={() => setFilter(pill.id)}
                  className={`rounded-full border px-3 py-1.5 text-xs font-semibold transition-all ${
                    filter === pill.id
                      ? "border-emerald-600 bg-emerald-600 text-white shadow-sm"
                      : "border-slate-200 bg-slate-50 text-slate-600 hover:bg-slate-100"
                  }`}
                >
                  {pill.label}
                </button>
              ))}
            </div>

            <div className="flex flex-wrap items-center gap-2.5">
              {/* Search */}
              <div className="relative flex-1 sm:w-56 sm:flex-none">
                <Search className="pointer-events-none absolute top-1/2 left-3 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
                <input
                  type="search"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Tìm tên ảnh, cây, bệnh..."
                  className="w-full rounded-xl border border-slate-200 bg-slate-50/70 py-1.5 pr-3 pl-8 text-xs text-slate-900 placeholder:text-slate-400 focus:border-emerald-500 focus:bg-white focus:outline-none"
                />
              </div>

              {/* View Switch */}
              <div className="inline-flex rounded-xl border border-slate-200 bg-slate-100/80 p-1">
                <button
                  type="button"
                  onClick={() => setViewLayout("grid")}
                  aria-label="Xem dạng lưới"
                  className={`rounded-lg p-1.5 transition-all ${
                    viewLayout === "grid"
                      ? "bg-white text-emerald-700 shadow-sm"
                      : "text-slate-500 hover:text-slate-800"
                  }`}
                >
                  <LayoutGrid className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => setViewLayout("table")}
                  aria-label="Xem dạng bảng"
                  className={`rounded-lg p-1.5 transition-all ${
                    viewLayout === "table"
                      ? "bg-white text-emerald-700 shadow-sm"
                      : "text-slate-500 hover:text-slate-800"
                  }`}
                >
                  <List className="h-3.5 w-3.5" />
                </button>
              </div>

              {/* Export CSV */}
              <button
                type="button"
                onClick={handleExportCsv}
                disabled={filteredResults.length === 0}
                className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs font-bold text-slate-700 transition-colors hover:border-emerald-300 hover:bg-emerald-50/50 hover:text-emerald-800 disabled:opacity-40"
              >
                <Download className="h-3.5 w-3.5" />
                <span>Xuất CSV</span>
              </button>
            </div>
          </section>

          {/* Grid or Table Content */}
          {filteredResults.length === 0 ? (
            <div className="rounded-2xl border border-slate-200 bg-white p-10 text-center">
              <p className="text-sm font-semibold text-slate-700">
                Không có kết quả nào khớp với bộ lọc hiện tại.
              </p>
            </div>
          ) : viewLayout === "grid" ? (
            <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {filteredResults.map((item, idx) => {
                const thumb = previewMap[item.filename];
                const severity = getSeverityMeta(item.advisory.severity);
                return (
                  <article
                    key={`${item.filename}-${idx}`}
                    onClick={() => setActiveDetail(item)}
                    className="group cursor-pointer overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm transition-all hover:-translate-y-0.5 hover:border-emerald-300 hover:shadow-md"
                  >
                    <div className="relative aspect-[4/3] w-full bg-slate-100">
                      {thumb ? (
                        <img
                          src={thumb}
                          alt={item.filename}
                          className="h-full w-full object-cover"
                        />
                      ) : (
                        <div className="flex h-full items-center justify-center text-slate-400">
                          <Sparkles className="h-6 w-6" />
                        </div>
                      )}
                      <span
                        className={`absolute top-2.5 left-2.5 inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-bold backdrop-blur-sm ${
                          item.is_uncertain
                            ? "border-amber-200 bg-amber-50/95 text-amber-800"
                            : item.is_healthy
                              ? "border-emerald-200 bg-emerald-50/95 text-emerald-800"
                              : "border-rose-200 bg-rose-50/95 text-rose-800"
                        }`}
                      >
                        {item.is_uncertain ? (
                          <AlertTriangle className="h-3 w-3" />
                        ) : item.is_healthy ? (
                          <CheckCircle2 className="h-3 w-3" />
                        ) : (
                          <ShieldAlert className="h-3 w-3" />
                        )}
                        {item.is_uncertain
                          ? "Cần xem lại"
                          : item.is_healthy
                            ? "Khỏe mạnh"
                            : "Có bệnh"}
                      </span>
                      <span className="absolute right-2.5 bottom-2.5 rounded-md bg-slate-900/80 px-2 py-0.5 font-mono text-[11px] font-bold text-white backdrop-blur-sm">
                        {formatConfidence(item.confidence)}
                      </span>
                    </div>

                    <div className="p-3.5">
                      <p className="truncate font-mono text-[11px] text-slate-400">
                        {item.filename}
                      </p>
                      <p className="mt-1 text-xs font-bold uppercase tracking-wider text-emerald-700">
                        {item.plant}
                      </p>
                      <h3 className="mt-0.5 line-clamp-1 text-sm font-bold text-slate-900">
                        {item.condition}
                      </h3>
                      <div className="mt-2.5 flex items-center justify-between border-t border-slate-100 pt-2.5">
                        <span
                          className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${severity.className}`}
                        >
                          {severity.label}
                        </span>
                        <span className="text-[11px] font-semibold text-emerald-700 group-hover:underline">
                          Chi tiết
                        </span>
                      </div>
                    </div>
                  </article>
                );
              })}
            </section>
          ) : (
            <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50/80 text-[11px] font-bold uppercase tracking-wider text-slate-500">
                      <th className="px-4 py-3">Ảnh</th>
                      <th className="px-4 py-3">Tệp</th>
                      <th className="px-4 py-3">Cây trồng</th>
                      <th className="px-4 py-3">Chẩn đoán</th>
                      <th className="px-4 py-3">Trạng thái</th>
                      <th className="px-4 py-3 text-right">Độ tin cậy</th>
                      <th className="px-4 py-3">Mức độ</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {filteredResults.map((item, idx) => {
                      const thumb = previewMap[item.filename];
                      const severity = getSeverityMeta(item.advisory.severity);
                      return (
                        <tr
                          key={`${item.filename}-${idx}`}
                          onClick={() => setActiveDetail(item)}
                          className="cursor-pointer transition-colors hover:bg-emerald-50/40"
                        >
                          <td className="px-4 py-2.5">
                            {thumb ? (
                              <img
                                src={thumb}
                                alt={item.filename}
                                className="h-10 w-10 rounded-lg object-cover ring-1 ring-slate-200"
                              />
                            ) : (
                              <div className="h-10 w-10 rounded-lg bg-slate-100" />
                            )}
                          </td>
                          <td className="max-w-[160px] truncate px-4 py-2.5 font-mono text-slate-600">
                            {item.filename}
                          </td>
                          <td className="px-4 py-2.5 font-semibold text-slate-900">
                            {item.plant}
                          </td>
                          <td className="px-4 py-2.5 font-medium text-slate-700">
                            {item.condition}
                          </td>
                          <td className="px-4 py-2.5">
                            <span
                              className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-semibold ${
                                item.is_uncertain
                                  ? "border-amber-200 bg-amber-50 text-amber-800"
                                  : item.is_healthy
                                    ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                                    : "border-rose-200 bg-rose-50 text-rose-800"
                              }`}
                            >
                              {item.is_uncertain
                                ? "Cần xem lại"
                                : item.is_healthy
                                  ? "Khỏe mạnh"
                                  : "Có bệnh"}
                            </span>
                          </td>
                          <td className="px-4 py-2.5 text-right font-mono font-bold text-slate-900">
                            {formatConfidence(item.confidence)}
                          </td>
                          <td className="px-4 py-2.5">
                            <span
                              className={`inline-flex rounded-full border px-2 py-0.5 text-[10px] font-semibold ${severity.className}`}
                            >
                              {severity.label}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}

      {/* Detail Modal */}
      {activeDetail && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={`Chi tiết chẩn đoán ${activeDetail.filename}`}
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 p-4 backdrop-blur-xs"
          onClick={() => setActiveDetail(null)}
        >
          <div
            className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-2xl border border-slate-200 bg-white p-5 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-3">
              <div className="min-w-0">
                <p className="truncate font-mono text-xs text-slate-400">
                  {activeDetail.filename}
                </p>
                <h3 className="mt-0.5 text-lg font-extrabold text-slate-900">
                  {activeDetail.plant} — {activeDetail.condition}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setActiveDetail(null)}
                aria-label="Đóng"
                className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {previewMap[activeDetail.filename] && (
              <img
                src={previewMap[activeDetail.filename]}
                alt={activeDetail.filename}
                className="mt-4 max-h-56 w-full rounded-xl bg-slate-950 object-contain"
              />
            )}

            <div className="mt-4 space-y-3">
              <div className="flex items-center justify-between rounded-xl border border-slate-100 bg-slate-50 px-3.5 py-2.5">
                <span className="text-xs font-semibold text-slate-600">Độ tin cậy Top-1</span>
                <span className="font-mono text-base font-bold text-emerald-700">
                  {formatConfidence(activeDetail.confidence)}
                </span>
              </div>

              <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-3.5">
                <p className="text-xs font-bold text-slate-800">Tóm tắt khuyến nghị</p>
                <p className="mt-1 text-xs leading-relaxed text-slate-600">
                  {activeDetail.advisory.summary}
                </p>
              </div>

              {activeDetail.advisory.treatment.length > 0 && (
                <div>
                  <p className="mb-1.5 text-xs font-bold text-slate-800">Biện pháp xử lý</p>
                  <ul className="space-y-1">
                    {activeDetail.advisory.treatment.map((step, i) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-slate-600">
                        <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-500" />
                        <span>{step}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
