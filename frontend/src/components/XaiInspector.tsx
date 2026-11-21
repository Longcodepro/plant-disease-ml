import { useRef, useState } from "react";
import {
  Columns2,
  Eye,
  Flame,
  GripVertical,
  ImageIcon,
  Layers,
  MoveHorizontal,
  Sparkles,
} from "lucide-react";
import { formatMs } from "../utils/format";

type ViewMode = "before-after" | "overlay" | "split";

interface XaiInspectorProps {
  originalSrc: string | null;
  heatmapOnlySrc: string | null;
  heatmapOverlaySrc: string | null;
  xaiMethod: string | null;
  explainTimeMs: number | null;
  isLoading: boolean;
}

export function XaiInspector({
  originalSrc,
  heatmapOnlySrc,
  heatmapOverlaySrc,
  xaiMethod,
  explainTimeMs,
  isLoading,
}: XaiInspectorProps) {
  const [mode, setMode] = useState<ViewMode>("before-after");
  const [splitPos, setSplitPos] = useState<number>(50);
  const [opacity, setOpacity] = useState<number>(72);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Prefer transparent heatmap_only layer for 60fps CSS opacity blending;
  // fall back to pre-blended heatmap_overlay if heatmap_only is absent.
  const activeHeatmapLayer = heatmapOnlySrc || heatmapOverlaySrc;
  const hasHeatmap = Boolean(activeHeatmapLayer);

  const updateSplitFromClientX = (clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return;
    const rawPct = ((clientX - rect.left) / rect.width) * 100;
    setSplitPos(Math.max(4, Math.min(96, rawPct)));
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (mode !== "before-after" || !hasHeatmap || isLoading) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setIsDragging(true);
    updateSplitFromClientX(event.clientX);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!isDragging) return;
    updateSplitFromClientX(event.clientX);
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!isDragging) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    setIsDragging(false);
  };

  const handleSliderKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      setSplitPos((prev) => Math.max(4, prev - 3));
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      setSplitPos((prev) => Math.min(96, prev + 3));
    }
  };

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      {/* Header & Mode Switch */}
      <div className="mb-3.5 flex flex-wrap items-center justify-between gap-2.5">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-50 text-emerald-600">
            <Eye className="h-4 w-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-900">Trình kiểm tra XAI trực quan</h2>
            <p className="text-xs text-slate-500">
              Vùng ảnh hưởng quyết định chẩn đoán của mạng nơ-ron
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {xaiMethod && (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 font-mono text-[11px] font-semibold text-emerald-700">
              <Sparkles className="h-3 w-3" />
              {xaiMethod}
              {explainTimeMs !== null && (
                <span className="text-emerald-600/80">· {formatMs(explainTimeMs)}</span>
              )}
            </span>
          )}
        </div>
      </div>

      {/* Mode Selector Tabs */}
      <div className="mb-3.5 flex flex-wrap items-center justify-between gap-2">
        <div
          role="tablist"
          aria-label="Chế độ xem bản đồ nhiệt XAI"
          className="inline-flex rounded-xl border border-slate-200 bg-slate-100/80 p-1"
        >
          <button
            type="button"
            role="tab"
            aria-selected={mode === "before-after"}
            onClick={() => setMode("before-after")}
            className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-semibold transition-all ${
              mode === "before-after"
                ? "bg-white text-emerald-700 shadow-sm ring-1 ring-slate-200/80"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <MoveHorizontal className="h-3.5 w-3.5" />
            <span>So sánh kéo</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "overlay"}
            onClick={() => setMode("overlay")}
            className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-semibold transition-all ${
              mode === "overlay"
                ? "bg-white text-emerald-700 shadow-sm ring-1 ring-slate-200/80"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Layers className="h-3.5 w-3.5" />
            <span>Lớp phủ</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "split"}
            onClick={() => setMode("split")}
            className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-semibold transition-all ${
              mode === "split"
                ? "bg-white text-emerald-700 shadow-sm ring-1 ring-slate-200/80"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Columns2 className="h-3.5 w-3.5" />
            <span>Song song</span>
          </button>
        </div>

        {mode === "before-after" && hasHeatmap && !isLoading && (
          <span className="font-mono text-[11px] font-medium text-slate-500">
            Kéo thanh phân tách ({Math.round(splitPos)}%)
          </span>
        )}
      </div>

      {/* Viewport */}
      {!originalSrc ? (
        <div className="flex aspect-[4/3] w-full flex-col items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50/60 p-6 text-center">
          <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-white text-slate-400 shadow-sm ring-1 ring-slate-200/80">
            <ImageIcon className="h-6 w-6" />
          </div>
          <p className="text-sm font-semibold text-slate-700">Chưa có ảnh lá cây để phân tích</p>
          <p className="mt-1 max-w-xs text-xs text-slate-500">
            Kéo thả ảnh vào khung bên trên hoặc chọn một mẫu lá từ thư viện để xem chẩn đoán tức thì.
          </p>
        </div>
      ) : isLoading ? (
        /* Laser Scan Loading State */
        <div className="relative aspect-[4/3] w-full overflow-hidden rounded-xl border border-slate-200 bg-slate-950">
          <img
            src={originalSrc}
            alt="Đang quét ảnh lá cây"
            className="h-full w-full object-contain opacity-80"
          />
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-emerald-500/5 via-transparent to-emerald-500/10" />
          {/* Animated Laser Line */}
          <div
            className="animate-laser-scan pointer-events-none absolute right-0 left-0 z-10 h-0.5 bg-emerald-400 shadow-[0_0_18px_4px_rgba(16,185,129,0.8)]"
            aria-hidden="true"
          >
            <div className="h-8 -translate-y-4 bg-gradient-to-b from-transparent via-emerald-400/20 to-transparent" />
          </div>
          <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-full border border-emerald-400/30 bg-slate-900/85 px-3 py-1.5 backdrop-blur-sm">
            <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
            <span className="font-mono text-xs font-medium text-emerald-100">
              Đang suy luận &amp; tổng hợp bản đồ XAI...
            </span>
          </div>
        </div>
      ) : mode === "split" ? (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="relative aspect-square overflow-hidden rounded-xl border border-slate-200 bg-slate-950">
            <img
              src={originalSrc}
              alt="Ảnh lá gốc"
              className="h-full w-full object-contain"
              draggable={false}
            />
            <span className="absolute top-2.5 left-2.5 rounded-md bg-slate-900/75 px-2 py-0.5 text-[11px] font-semibold text-white backdrop-blur-sm">
              Ảnh gốc
            </span>
          </div>
          <div className="relative aspect-square overflow-hidden rounded-xl border border-slate-200 bg-slate-950">
            <img
              src={originalSrc}
              alt="Ảnh nền XAI"
              className="h-full w-full object-contain"
              draggable={false}
            />
            {activeHeatmapLayer && (
              <img
                src={activeHeatmapLayer}
                alt="Bản đồ nhiệt XAI"
                style={{ opacity: opacity / 100 }}
                className="pointer-events-none absolute inset-0 h-full w-full object-contain will-change-[opacity]"
                draggable={false}
              />
            )}
            <span className="absolute top-2.5 left-2.5 rounded-md bg-emerald-700/85 px-2 py-0.5 text-[11px] font-semibold text-white backdrop-blur-sm">
              Bản đồ XAI ({opacity}%)
            </span>
          </div>
        </div>
      ) : mode === "overlay" ? (
        <div className="relative aspect-[4/3] w-full overflow-hidden rounded-xl border border-slate-200 bg-slate-950">
          <img
            src={originalSrc}
            alt="Ảnh lá gốc"
            className="h-full w-full object-contain"
            draggable={false}
          />
          {activeHeatmapLayer && (
            <img
              src={activeHeatmapLayer}
              alt="Lớp phủ nhiệt XAI"
              style={{ opacity: opacity / 100 }}
              className="pointer-events-none absolute inset-0 h-full w-full object-contain will-change-[opacity]"
              draggable={false}
            />
          )}
          <div className="pointer-events-none absolute top-3 left-3 rounded-md bg-slate-900/75 px-2.5 py-1 text-[11px] font-semibold text-white backdrop-blur-sm">
            Lớp phủ XAI · <span className="font-mono">{opacity}%</span>
          </div>
        </div>
      ) : (
        /* Before / After Draggable Divider using CSS clip-path */
        <div
          ref={containerRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          className={`relative aspect-[4/3] w-full select-none overflow-hidden rounded-xl border border-slate-200 bg-slate-950 ${
            hasHeatmap ? "cursor-ew-resize touch-none" : ""
          }`}
        >
          {/* Base (Right / After): Original + XAI Heatmap at current opacity */}
          <img
            src={originalSrc}
            alt="Ảnh lá kết hợp XAI"
            className="h-full w-full object-contain"
            draggable={false}
          />
          {activeHeatmapLayer && (
            <img
              src={activeHeatmapLayer}
              alt="Lớp phủ nhiệt XAI"
              style={{ opacity: opacity / 100 }}
              className="pointer-events-none absolute inset-0 h-full w-full object-contain will-change-[opacity]"
              draggable={false}
            />
          )}

          {/* Top Clipped Layer (Left / Before): Original Image Only */}
          {hasHeatmap && (
            <div
              className="pointer-events-none absolute inset-0"
              style={{ clipPath: `inset(0 ${100 - splitPos}% 0 0)` }}
            >
              <img
                src={originalSrc}
                alt="Ảnh gốc trước khi phủ XAI"
                className="h-full w-full object-contain"
                draggable={false}
              />
            </div>
          )}

          {/* Labels */}
          <span className="pointer-events-none absolute top-3 left-3 rounded-md bg-slate-900/75 px-2.5 py-1 text-[11px] font-semibold text-white backdrop-blur-sm">
            Ảnh gốc
          </span>
          {hasHeatmap && (
            <span className="pointer-events-none absolute top-3 right-3 rounded-md bg-emerald-700/85 px-2.5 py-1 text-[11px] font-semibold text-white backdrop-blur-sm">
              Bản đồ XAI
            </span>
          )}

          {/* Vertical Hairline & Drag Handle */}
          {hasHeatmap && (
            <div
              role="slider"
              tabIndex={0}
              aria-label="Thanh kéo so sánh ảnh gốc và bản đồ XAI"
              aria-valuemin={4}
              aria-valuemax={96}
              aria-valuenow={Math.round(splitPos)}
              onKeyDown={handleSliderKeyDown}
              style={{ left: `${splitPos}%` }}
              className="absolute top-0 bottom-0 z-20 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_8px_rgba(15,23,42,0.55)] focus:outline-none"
            >
              <div className="absolute top-1/2 left-1/2 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 border-emerald-600 bg-white text-emerald-700 shadow-lg transition-transform hover:scale-105">
                <GripVertical className="h-4 w-4" />
              </div>
            </div>
          )}
        </div>
      )}

      {/* 60fps Opacity Slider & Colorbar Scale Legend */}
      <div className="mt-4 space-y-3 rounded-xl border border-slate-100 bg-slate-50/70 p-3.5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <label
            htmlFor="xai-opacity-slider"
            className="flex items-center gap-1.5 text-xs font-semibold text-slate-700"
          >
            <Flame className="h-3.5 w-3.5 text-emerald-600" />
            <span>Độ đậm lớp phủ nhiệt (60fps)</span>
          </label>
          <div className="flex items-center gap-3 sm:w-64">
            <input
              id="xai-opacity-slider"
              type="range"
              min={0}
              max={100}
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              disabled={!hasHeatmap}
              className="precision-slider w-full disabled:cursor-not-allowed disabled:opacity-40"
            />
            <span className="w-11 text-right font-mono text-xs font-bold text-slate-800">
              {opacity}%
            </span>
          </div>
        </div>

        {/* Colorbar Legend */}
        <div className="flex items-center gap-2.5 pt-1 text-[11px] font-medium text-slate-500">
          <span className="shrink-0">Ít ảnh hưởng</span>
          <div
            className="h-2 flex-1 rounded-full bg-gradient-to-r from-blue-600 via-emerald-400 via-60% to-rose-600"
            aria-hidden="true"
          />
          <span className="shrink-0 font-semibold text-slate-700">
            Vùng tổn thương quyết định
          </span>
        </div>
      </div>
    </div>
  );
}
