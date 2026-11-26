import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Clock,
  FileImage,
  Leaf,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  UploadCloud,
  Zap,
} from "lucide-react";
import { fetchSamples, predictSingle, resolveApiUrl } from "../services/api";
import type { PredictApiResponse, SampleItem } from "../types/api";
import {
  formatConfidence,
  formatMs,
  getSeverityMeta,
  toImageDataUri,
  toPercentNumber,
} from "../utils/format";
import { XaiInspector } from "./XaiInspector";

const ALL_PLANTS_FILTER = "Tất cả";
type SampleSourceFilter = "all" | "field" | "lab";
const isFieldSample = (sample: SampleItem): boolean => sample.filename.startsWith("PlantDoc_");

/**
 * Interleave samples round-robin by (plant, condition) so the initial
 * visible strip showcases diverse crops and conditions immediately.
 */
function interleaveSamples(items: SampleItem[]): SampleItem[] {
  if (items.length <= 1) return items;

  const byPlant = new Map<string, Map<string, SampleItem[]>>();
  for (const item of items) {
    let condMap = byPlant.get(item.plant);
    if (!condMap) {
      condMap = new Map<string, SampleItem[]>();
      byPlant.set(item.plant, condMap);
    }
    const bucket = condMap.get(item.condition) ?? [];
    bucket.push(item);
    condMap.set(item.condition, bucket);
  }

  const plantCondQueues = Array.from(byPlant.keys())
    .sort((a, b) => a.localeCompare(b))
    .map((plant) => {
      const condMap = byPlant.get(plant)!;
      return Array.from(condMap.keys())
        .sort((a, b) => a.localeCompare(b))
        .map((condition) => [...condMap.get(condition)!]);
    });

  const orderedBuckets: SampleItem[][] = [];
  let hasMoreBuckets = true;
  while (hasMoreBuckets) {
    hasMoreBuckets = false;
    for (const queues of plantCondQueues) {
      const nextBucket = queues.shift();
      if (nextBucket) {
        orderedBuckets.push(nextBucket);
        hasMoreBuckets = true;
      }
    }
  }

  const interleaved: SampleItem[] = [];
  let hasMoreItems = true;
  while (hasMoreItems) {
    hasMoreItems = false;
    for (const bucket of orderedBuckets) {
      const nextItem = bucket.shift();
      if (nextItem) {
        interleaved.push(nextItem);
        hasMoreItems = true;
      }
    }
  }

  return interleaved;
}

export function DiagnoseView() {
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [samplesLoading, setSamplesLoading] = useState<boolean>(true);
  const [samplesError, setSamplesError] = useState<string | null>(null);
  const [sourceFilter, setSourceFilter] = useState<SampleSourceFilter>("all");
  const [selectedPlantFilter, setSelectedPlantFilter] = useState<string>(ALL_PLANTS_FILTER);
  const [canScrollLeft, setCanScrollLeft] = useState<boolean>(false);
  const [canScrollRight, setCanScrollRight] = useState<boolean>(false);

  const [selectedSampleName, setSelectedSampleName] = useState<string | null>(null);
  const [localPreviewUrl, setLocalPreviewUrl] = useState<string | null>(null);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [isDraggingOver, setIsDraggingOver] = useState<boolean>(false);

  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictApiResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const sampleStripRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const loadSamples = useCallback(async () => {
    setSamplesLoading(true);
    setSamplesError(null);
    try {
      const data = await fetchSamples();
      setSamples(data.samples);
    } catch (err) {
      setSamplesError(err instanceof Error ? err.message : "Không thể tải danh sách ảnh mẫu.");
    } finally {
      setSamplesLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSamples();
    return () => {
      abortControllerRef.current?.abort();
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
    };
  }, [loadSamples]);

  const runPrediction = useCallback(
    async (opts: { file?: File; sampleName?: string; previewUrl: string; label: string }) => {
      abortControllerRef.current?.abort();
      const controller = new AbortController();
      abortControllerRef.current = controller;

      setLocalPreviewUrl(opts.previewUrl);
      setSelectedFileName(opts.label);
      setResult(null);
      setPredictError(null);
      setIsPredicting(true);

      try {
        const response = await predictSingle({
          file: opts.file,
          sampleName: opts.sampleName,
          includeXai: true,
          signal: controller.signal,
        });
        setResult(response);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setPredictError(
          err instanceof Error ? err.message : "Đã xảy ra lỗi khi chẩn đoán hình ảnh.",
        );
      } finally {
        if (abortControllerRef.current === controller) {
          setIsPredicting(false);
        }
      }
    },
    [],
  );

  const handleFileSelected = useCallback(
    (file: File | undefined | null) => {
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        setPredictError("Định dạng tệp không hợp lệ. Vui lòng chọn tệp ảnh (JPG, PNG, WEBP).");
        return;
      }
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
      const nextUrl = URL.createObjectURL(file);
      objectUrlRef.current = nextUrl;
      setSelectedSampleName(null);
      void runPrediction({ file, previewUrl: nextUrl, label: file.name });
    },
    [runPrediction],
  );

  const handleSampleClick = useCallback(
    (sample: SampleItem) => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
      setSelectedSampleName(sample.filename);
      void runPrediction({
        sampleName: sample.filename,
        previewUrl: resolveApiUrl(sample.url),
        label: `${sample.plant} — ${sample.condition}`,
      });
    },
    [runPrediction],
  );

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDraggingOver(false);
    const droppedFile = event.dataTransfer.files?.[0];
    handleFileSelected(droppedFile);
  };

  const sourceFilteredSamples = useMemo(() => {
    if (sourceFilter === "field") return samples.filter(isFieldSample);
    if (sourceFilter === "lab") return samples.filter((s) => !isFieldSample(s));
    return samples;
  }, [samples, sourceFilter]);

  const plantOptions = useMemo(() => {
    const uniquePlants = Array.from(new Set(sourceFilteredSamples.map((s) => s.plant))).sort((a, b) =>
      a.localeCompare(b),
    );
    return [ALL_PLANTS_FILTER, ...uniquePlants];
  }, [sourceFilteredSamples]);

  const activePlantFilter = plantOptions.includes(selectedPlantFilter)
    ? selectedPlantFilter
    : ALL_PLANTS_FILTER;

  const displayedSamples = useMemo(() => {
    if (activePlantFilter === ALL_PLANTS_FILTER) {
      return interleaveSamples(sourceFilteredSamples);
    }
    return sourceFilteredSamples.filter((s) => s.plant === activePlantFilter);
  }, [sourceFilteredSamples, activePlantFilter]);

  const updateScrollButtons = useCallback(() => {
    const el = sampleStripRef.current;
    if (!el) {
      setCanScrollLeft(false);
      setCanScrollRight(false);
      return;
    }
    setCanScrollLeft(el.scrollLeft > 4);
    setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 4);
  }, []);

  useEffect(() => {
    const el = sampleStripRef.current;
    if (el) {
      el.scrollLeft = 0;
    }
    const rafId = window.requestAnimationFrame(updateScrollButtons);
    window.addEventListener("resize", updateScrollButtons);
    return () => {
      window.cancelAnimationFrame(rafId);
      window.removeEventListener("resize", updateScrollButtons);
    };
  }, [displayedSamples, samplesLoading, updateScrollButtons]);

  const scrollSampleStrip = (direction: "left" | "right") => {
    const el = sampleStripRef.current;
    if (!el) return;
    const distance = Math.max(220, Math.round(el.clientWidth * 0.72));
    el.scrollBy({
      left: direction === "left" ? -distance : distance,
      behavior: "smooth",
    });
  };

  const originalSrc = isPredicting
    ? localPreviewUrl
    : toImageDataUri(result?.original_image_base64) || localPreviewUrl;
  const heatmapOnlySrc = toImageDataUri(result?.heatmap_only_base64);
  const heatmapOverlaySrc = toImageDataUri(result?.heatmap_overlay_base64);

  const severityMeta = result ? getSeverityMeta(result.advisory.severity) : null;

  return (
    <div className="space-y-6">
      {/* Top Row: Upload Dropzone + Horizontal Sample Strip */}
      <section className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        {/* Dropzone */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDraggingOver(true);
          }}
          onDragLeave={() => setIsDraggingOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
          role="button"
          tabIndex={0}
          aria-label="Kéo thả hoặc chọn ảnh lá cây để chẩn đoán tự động"
          className={`group flex cursor-pointer items-center gap-4 rounded-2xl border-2 border-dashed p-4 transition-all lg:col-span-5 ${
            isDraggingOver
              ? "border-emerald-500 bg-emerald-50/70 shadow-sm"
              : "border-slate-300/90 bg-white hover:border-emerald-400 hover:bg-emerald-50/30"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              handleFileSelected(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
          <div
            className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl transition-colors ${
              isDraggingOver
                ? "bg-emerald-600 text-white"
                : "bg-emerald-50 text-emerald-600 group-hover:bg-emerald-100"
            }`}
          >
            <UploadCloud className="h-6 w-6" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-sm font-bold text-slate-900">Tải ảnh lá cây lên</p>
              <span className="rounded-md bg-emerald-50 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-700">
                Tự động chẩn đoán
              </span>
            </div>
            <p className="mt-0.5 text-xs text-slate-500">
              Kéo thả hoặc nhấp để chọn tệp JPG, PNG, WEBP.
            </p>
            {selectedFileName && (
              <p className="mt-1.5 flex items-center gap-1.5 truncate font-mono text-[11px] font-medium text-emerald-700">
                <FileImage className="h-3.5 w-3.5 shrink-0" />
                <span className="truncate">{selectedFileName}</span>
              </p>
            )}
          </div>
        </div>

        {/* Horizontal Sample Strip */}
        <div className="flex flex-col justify-between rounded-2xl border border-slate-200 bg-white p-4 shadow-sm lg:col-span-7">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex items-center gap-2">
                <Leaf className="h-4 w-4 text-emerald-600" />
                <h2 className="text-xs font-bold uppercase tracking-wider text-slate-700">
                  Thư viện ảnh mẫu nhanh ({displayedSamples.length})
                </h2>
              </div>
              {!samplesLoading && !samplesError && samples.some(isFieldSample) && (
                <div
                  role="group"
                  aria-label="Chọn nguồn ảnh mẫu"
                  className="inline-flex items-center rounded-lg border border-slate-200 bg-slate-100/70 p-0.5 text-[11px] font-semibold"
                >
                  {(
                    [
                      ["all", "Tất cả"],
                      ["field", "Ngoài trời (PlantDoc)"],
                      ["lab", "Phòng Lab"],
                    ] as const
                  ).map(([value, label]) => (
                    <button
                      key={value}
                      type="button"
                      onClick={() => setSourceFilter(value)}
                      className={`rounded-md px-2 py-0.5 transition-all ${
                        sourceFilter === value
                          ? "bg-white text-emerald-700 shadow-xs ring-1 ring-slate-200/80"
                          : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div className="flex items-center gap-1.5">
              {samplesError && (
                <button
                  type="button"
                  onClick={() => void loadSamples()}
                  className="inline-flex items-center gap-1 text-xs font-semibold text-emerald-700 hover:text-emerald-800"
                >
                  <RefreshCw className="h-3 w-3" />
                  Thử lại
                </button>
              )}
              {!samplesLoading && !samplesError && displayedSamples.length > 0 && (
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => scrollSampleStrip("left")}
                    disabled={!canScrollLeft}
                    aria-label="Cuộn danh sách mẫu sang trái"
                    className="flex h-7 w-7 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-600 transition-colors hover:border-emerald-300 hover:bg-emerald-50 hover:text-emerald-700 disabled:cursor-not-allowed disabled:opacity-35"
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    onClick={() => scrollSampleStrip("right")}
                    disabled={!canScrollRight}
                    aria-label="Cuộn danh sách mẫu sang phải"
                    className="flex h-7 w-7 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-600 transition-colors hover:border-emerald-300 hover:bg-emerald-50 hover:text-emerald-700 disabled:cursor-not-allowed disabled:opacity-35"
                  >
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Plant Filter Chips */}
          {!samplesLoading && !samplesError && plantOptions.length > 1 && (
            <div
              role="group"
              aria-label="Lọc ảnh mẫu theo loại cây"
              className="thin-scrollbar mb-2.5 flex items-center gap-1.5 overflow-x-auto pb-0.5"
            >
              {plantOptions.map((plant) => {
                const isSelected = activePlantFilter === plant;
                return (
                  <button
                    key={plant}
                    type="button"
                    onClick={() => setSelectedPlantFilter(plant)}
                    className={`shrink-0 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold transition-all ${
                      isSelected
                        ? "border-emerald-600 bg-emerald-600 text-white shadow-xs"
                        : "border-slate-200 bg-slate-50 text-slate-600 hover:border-emerald-300 hover:bg-emerald-50/60 hover:text-emerald-800"
                    }`}
                  >
                    {plant}
                  </button>
                );
              })}
            </div>
          )}

          {samplesLoading ? (
            <div className="flex gap-2.5 overflow-hidden">
              {[1, 2, 3, 4].map((idx) => (
                <div
                  key={idx}
                  className="h-16 w-36 shrink-0 animate-pulse rounded-xl bg-slate-100"
                />
              ))}
            </div>
          ) : samplesError ? (
            <div className="rounded-xl border border-rose-200 bg-rose-50/70 px-3 py-2 text-xs text-rose-700">
              {samplesError}
            </div>
          ) : displayedSamples.length === 0 ? (
            <p className="py-3 text-xs text-slate-500">Không có ảnh mẫu khả dụng trên máy chủ.</p>
          ) : (
            <div className="relative">
              <div
                ref={sampleStripRef}
                onScroll={updateScrollButtons}
                className="thin-scrollbar flex gap-2.5 overflow-x-auto pb-1 pr-2 scroll-smooth"
              >
                {displayedSamples.map((sample) => {
                  const active = selectedSampleName === sample.filename;
                  return (
                    <button
                      key={sample.filename}
                      type="button"
                      onClick={() => handleSampleClick(sample)}
                      className={`group flex shrink-0 items-center gap-2.5 rounded-xl border p-1.5 pr-3 text-left transition-all ${
                        active
                          ? "border-emerald-500 bg-emerald-50/60 ring-2 ring-emerald-500/20"
                          : "border-slate-200 bg-slate-50/50 hover:border-emerald-300 hover:bg-white"
                      }`}
                    >
                      <img
                        src={resolveApiUrl(sample.url)}
                        alt={`${sample.plant} ${sample.condition}`}
                        loading="lazy"
                        className="h-12 w-12 shrink-0 rounded-lg object-cover ring-1 ring-slate-200/80"
                      />
                      <div className="max-w-[144px]">
                        <div className="flex items-center gap-1.5">
                          <span
                            className={`h-2 w-2 shrink-0 rounded-full ${
                              sample.is_healthy ? "bg-emerald-500" : "bg-amber-500"
                            }`}
                          />
                          <span className="truncate text-xs font-bold text-slate-900">
                            {sample.plant}
                          </span>
                          {isFieldSample(sample) && (
                            <span className="shrink-0 rounded bg-sky-50 px-1 py-0.2 font-mono text-[9px] font-semibold text-sky-700 ring-1 ring-sky-200/70">
                              Field
                            </span>
                          )}
                        </div>
                        <p className="mt-0.5 truncate text-[11px] text-slate-500">
                          {sample.condition}
                        </p>
                      </div>
                    </button>
                  );
                })}
              </div>
              {canScrollRight && (
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute top-0 right-0 bottom-1 w-12 bg-gradient-to-l from-white to-transparent"
                />
              )}
            </div>
          )}
        </div>
      </section>

      {/* Error Alert */}
      {predictError && (
        <div
          role="alert"
          className="flex items-start gap-3 rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800"
        >
          <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-rose-600" />
          <div>
            <p className="font-bold">Chẩn đoán không thành công</p>
            <p className="mt-0.5 text-xs text-rose-700">{predictError}</p>
          </div>
        </div>
      )}

      {/* Main Split Bento Workspace */}
      <section className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12">
        {/* Left: Interactive XAI Inspector */}
        <div className="lg:col-span-7">
          <XaiInspector
            originalSrc={originalSrc}
            heatmapOnlySrc={heatmapOnlySrc}
            heatmapOverlaySrc={heatmapOverlaySrc}
            xaiMethod={result?.xai_method ?? null}
            explainTimeMs={result?.explain_time_ms ?? null}
            isLoading={isPredicting}
          />
        </div>

        {/* Right: Bento Grid (Hero + Top-3 + Advisory) */}
        <div className="space-y-4 lg:col-span-5">
          {isPredicting ? (
            /* Skeleton Shimmer on Right Bento Cards */
            <div className="space-y-4" aria-busy="true" aria-label="Đang phân tích kết quả">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex items-center justify-between">
                  <div className="h-5 w-28 animate-pulse rounded-full bg-slate-200" />
                  <div className="h-5 w-20 animate-pulse rounded-full bg-slate-100" />
                </div>
                <div className="mt-4 h-7 w-3/4 animate-pulse rounded-lg bg-slate-200" />
                <div className="mt-2 h-4 w-1/2 animate-pulse rounded-md bg-slate-100" />
                <div className="mt-5 grid grid-cols-2 gap-3">
                  <div className="h-16 animate-pulse rounded-xl bg-slate-100" />
                  <div className="h-16 animate-pulse rounded-xl bg-slate-100" />
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="h-4 w-40 animate-pulse rounded bg-slate-200" />
                <div className="mt-4 space-y-3">
                  {[1, 2, 3].map((row) => (
                    <div key={row} className="space-y-1.5">
                      <div className="h-3 w-full animate-pulse rounded bg-slate-100" />
                      <div className="h-2.5 w-full animate-pulse rounded-full bg-slate-200" />
                    </div>
                  ))}
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="h-4 w-44 animate-pulse rounded bg-slate-200" />
                <div className="mt-3 space-y-2">
                  <div className="h-3.5 w-full animate-pulse rounded bg-slate-100" />
                  <div className="h-3.5 w-5/6 animate-pulse rounded bg-slate-100" />
                  <div className="h-3.5 w-4/6 animate-pulse rounded bg-slate-100" />
                </div>
              </div>
            </div>
          ) : !result ? (
            /* Empty State */
            <div className="flex flex-col items-center justify-center rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm">
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-emerald-50 text-emerald-600">
                <Sparkles className="h-6 w-6" />
              </div>
              <h3 className="text-base font-bold text-slate-900">Sẵn sàng chẩn đoán bệnh lá</h3>
              <p className="mt-1.5 max-w-sm text-xs leading-relaxed text-slate-500">
                Chọn một ảnh mẫu phía trên hoặc kéo thả ảnh lá cây của bạn. Kết quả phân loại,
                bản đồ nhiệt Grad-CAM và khuyến nghị nông học sẽ xuất hiện tại đây.
              </p>
            </div>
          ) : (
            <>
              {/* Card 1: Diagnosis Hero Card */}
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-bold ${
                        result.is_uncertain
                          ? "border-amber-200 bg-amber-50 text-amber-800"
                          : result.is_healthy
                            ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                            : "border-rose-200 bg-rose-50 text-rose-800"
                      }`}
                    >
                      {result.is_uncertain ? (
                        <AlertTriangle className="h-3.5 w-3.5" />
                      ) : result.is_healthy ? (
                        <CheckCircle2 className="h-3.5 w-3.5" />
                      ) : (
                        <ShieldAlert className="h-3.5 w-3.5" />
                      )}
                      {result.is_uncertain
                        ? "Cần kiểm tra thêm"
                        : result.is_healthy
                          ? "Lá khỏe mạnh"
                          : "Phát hiện bệnh"}
                    </span>

                    {severityMeta && (
                      <span
                        className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${severityMeta.className}`}
                      >
                        {severityMeta.label}
                      </span>
                    )}
                  </div>

                  {result.cache_hit && (
                    <span className="inline-flex items-center gap-1 rounded-full border border-sky-200 bg-sky-50 px-2.5 py-0.5 font-mono text-[11px] font-semibold text-sky-700">
                      <Zap className="h-3 w-3" />
                      Cache Hit
                    </span>
                  )}
                </div>

                {/* Plant + Condition Heading */}
                <div className="mt-4">
                  <p className="text-xs font-bold uppercase tracking-wider text-emerald-700">
                    Cây trồng: {result.plant}
                  </p>
                  <h2 className="mt-1 text-xl font-extrabold tracking-tight text-slate-900">
                    {result.condition}
                  </h2>
                  <p className="mt-0.5 font-mono text-[11px] text-slate-400">
                    {result.predicted_class} · {result.model_version}
                  </p>
                </div>

                {/* Confidence & Latency Readouts */}
                <div className="mt-4 grid grid-cols-2 gap-3">
                  <div className="rounded-xl border border-slate-100 bg-slate-50/80 p-3">
                    <span className="text-[11px] font-semibold text-slate-500">Độ tin cậy</span>
                    <p className="mt-0.5 font-mono text-2xl font-bold tracking-tight text-emerald-700">
                      {formatConfidence(result.confidence)}
                    </p>
                  </div>
                  <div className="rounded-xl border border-slate-100 bg-slate-50/80 p-3">
                    <span className="flex items-center gap-1 text-[11px] font-semibold text-slate-500">
                      <Clock className="h-3 w-3" />
                      Thời gian xử lý
                    </span>
                    <p className="mt-0.5 font-mono text-2xl font-bold tracking-tight text-slate-900">
                      {result.cache_hit
                        ? `${result.inference_time_ms.toFixed(2)} ms`
                        : formatMs(result.inference_time_ms)}
                    </p>
                    {result.cache_hit && (
                      <p className="mt-1 text-[10px] font-medium leading-snug text-sky-700">
                        Phản hồi từ bộ nhớ đệm (Inference gốc: ~24 ms)
                      </p>
                    )}
                  </div>
                </div>

                {/* Uncertain Warning Callout */}
                {result.is_uncertain && (
                  <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-amber-200 bg-amber-50/90 p-3 text-xs text-amber-900">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
                    <div>
                      <p className="font-bold">Độ tin cậy thấp hơn ngưỡng an toàn</p>
                      <p className="mt-0.5 leading-relaxed text-amber-800">
                        {result.warning ||
                          "Ảnh có thể bị mờ, thiếu sáng hoặc chứa nhiều lá chồng chéo. Khuyến nghị chụp lại cận cảnh dưới ánh sáng tự nhiên."}
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Card 2: Top-3 Confidence Card */}
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="mb-3.5 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-emerald-600" />
                    <h3 className="text-sm font-bold text-slate-900">Top-3 xác suất dự đoán</h3>
                  </div>
                  <span className="font-mono text-[11px] text-slate-400">Softmax</span>
                </div>

                <div className="space-y-3">
                  {result.top_3.slice(0, 3).map((item, idx) => {
                    const pct = toPercentNumber(item.confidence);
                    const isTop = idx === 0;
                    return (
                      <div key={item.class_name} className="space-y-1.5">
                        <div className="flex items-baseline justify-between gap-2 text-xs">
                          <div className="min-w-0 truncate">
                            <span className="font-bold text-slate-800">{item.plant}</span>
                            <span className="mx-1 text-slate-400">·</span>
                            <span className="font-medium text-slate-600">{item.condition}</span>
                          </div>
                          <span
                            className={`shrink-0 font-mono text-xs font-bold ${
                              isTop ? "text-emerald-700" : "text-slate-600"
                            }`}
                          >
                            {formatConfidence(item.confidence)}
                          </span>
                        </div>
                        <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                          <div
                            style={{ width: `${pct}%` }}
                            className={`h-full rounded-full transition-all duration-500 ease-out ${
                              isTop ? "bg-emerald-600" : "bg-slate-400"
                            }`}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>

                {(() => {
                  const top3Items = result.top_3.slice(0, 3);
                  const sumTop3Pct = top3Items.reduce(
                    (acc, item) => acc + toPercentNumber(item.confidence),
                    0,
                  );
                  const remainingClasses = Math.max(0, 38 - top3Items.length);
                  const remainingPct = Math.max(0, 100 - sumTop3Pct);
                  return (
                    <div className="mt-3.5 flex items-center justify-between border-t border-slate-100 pt-2.5 text-[11px] text-slate-500">
                      <span>{remainingClasses} lớp còn lại</span>
                      <span className="font-mono font-semibold text-slate-600">
                        {remainingPct.toFixed(2)}%
                      </span>
                    </div>
                  );
                })()}
              </div>

              {/* Card 3: Agronomic Advisory Card */}
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="mb-3 flex items-center gap-2">
                  <Stethoscope className="h-4 w-4 text-emerald-600" />
                  <h3 className="text-sm font-bold text-slate-900">Khuyến nghị nông học</h3>
                </div>

                <p className="rounded-xl border border-slate-100 bg-slate-50/80 p-3 text-xs leading-relaxed text-slate-700">
                  {result.advisory.summary}
                </p>

                {result.advisory.treatment.length > 0 && (
                  <div className="mt-4">
                    <h4 className="mb-2 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-700">
                      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                      Biện pháp xử lý đề xuất
                    </h4>
                    <ul className="space-y-1.5">
                      {result.advisory.treatment.map((step, idx) => (
                        <li
                          key={idx}
                          className="flex items-start gap-2 text-xs leading-relaxed text-slate-600"
                        >
                          <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-500" />
                          <span>{step}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {result.advisory.prevention.length > 0 && (
                  <div className="mt-4 border-t border-slate-100 pt-3.5">
                    <h4 className="mb-2 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-700">
                      <ShieldCheck className="h-3.5 w-3.5 text-sky-600" />
                      Phòng ngừa tái nhiễm
                    </h4>
                    <ul className="space-y-1.5">
                      {result.advisory.prevention.map((tip, idx) => (
                        <li
                          key={idx}
                          className="flex items-start gap-2 text-xs leading-relaxed text-slate-600"
                        >
                          <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-sky-500" />
                          <span>{tip}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </section>
    </div>
  );
}
