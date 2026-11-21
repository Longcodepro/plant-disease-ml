import { type RefObject, useEffect, useRef, useState } from "react";
import {
  Columns2,
  Eye,
  Flame,
  GripVertical,
  ImageIcon,
  Layers,
  Maximize2,
  MoveHorizontal,
  Sparkles,
  X,
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

const COLORBAR_GRADIENT =
  "linear-gradient(90deg, #1d4ed8 0%, #06b6d4 25%, #22c55e 50%, #facc15 75%, #dc2626 100%)";

function ModeTabs({
  mode,
  onChange,
  dark = false,
}: {
  mode: ViewMode;
  onChange: (next: ViewMode) => void;
  dark?: boolean;
}) {
  const tabs: Array<{ id: ViewMode; label: string; icon: typeof MoveHorizontal }> = [
    { id: "before-after", label: "So sánh kéo", icon: MoveHorizontal },
    { id: "overlay", label: "Lớp phủ", icon: Layers },
    { id: "split", label: "Song song", icon: Columns2 },
  ];

  return (
    <div
      role="tablist"
      aria-label="Chế độ xem bản đồ nhiệt XAI"
      className={`inline-flex rounded-xl border p-1 ${
        dark
          ? "border-slate-700 bg-slate-800/90"
          : "border-slate-200 bg-slate-100/80"
      }`}
    >
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const selected = mode === tab.id;
        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={selected}
            onClick={() => onChange(tab.id)}
            className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-semibold transition-all ${
              selected
                ? dark
                  ? "bg-emerald-600 text-white shadow-sm"
                  : "bg-white text-emerald-700 shadow-sm ring-1 ring-slate-200/80"
                : dark
                  ? "text-slate-300 hover:text-white"
                  : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Icon className="h-3.5 w-3.5" />
            <span>{tab.label}</span>
          </button>
        );
      })}
    </div>
  );
}

function ActivationColorbar({ dark = false }: { dark?: boolean }) {
  return (
    <div
      className={`rounded-xl border px-3.5 py-2.5 ${
        dark
          ? "border-slate-800 bg-slate-900/90"
          : "border-slate-200/80 bg-slate-50/80"
      }`}
    >
      <div className="mb-1.5 flex items-center justify-between text-[11px]">
        <span className={`font-semibold ${dark ? "text-slate-200" : "text-slate-700"}`}>
          Thang đo mức độ chú ý XAI
        </span>
        <span className={`font-mono text-[10px] ${dark ? "text-slate-400" : "text-slate-400"}`}>
          0.0 — 1.0
        </span>
      </div>
      <div
        aria-hidden="true"
        style={{ background: COLORBAR_GRADIENT }}
        className="h-2 w-full rounded-full shadow-inner ring-1 ring-black/10"
      />
      <div
        className={`mt-1.5 flex items-center justify-between gap-2 text-[10px] sm:text-[11px] ${
          dark ? "text-slate-300" : "text-slate-500"
        }`}
      >
        <span className="font-medium">Thấp · Nền lá</span>
        <span className="font-medium">Trung bình</span>
        <span className={`font-semibold ${dark ? "text-white" : "text-slate-700"}`}>
          Cao · Vùng tổn thương quyết định
        </span>
      </div>
    </div>
  );
}

function OpacitySliderControl({
  inputId,
  opacity,
  onChange,
  disabled,
  dark = false,
}: {
  inputId: string;
  opacity: number;
  onChange: (value: number) => void;
  disabled: boolean;
  dark?: boolean;
}) {
  return (
    <div
      className={`flex flex-col gap-2 rounded-xl border p-3.5 sm:flex-row sm:items-center sm:justify-between ${
        dark
          ? "border-slate-800 bg-slate-900/90"
          : "border-slate-100 bg-slate-50/70"
      }`}
    >
      <label
        htmlFor={inputId}
        className={`flex items-center gap-1.5 text-xs font-semibold ${
          dark ? "text-slate-200" : "text-slate-700"
        }`}
      >
        <Flame className="h-3.5 w-3.5 text-emerald-500" />
        <span>Độ mờ Heatmap (0% – 100%)</span>
      </label>
      <div className="flex items-center gap-3 sm:w-64">
        <input
          id={inputId}
          type="range"
          min={0}
          max={100}
          value={opacity}
          onChange={(e) => onChange(Number(e.target.value))}
          disabled={disabled}
          className="precision-slider w-full disabled:cursor-not-allowed disabled:opacity-40"
        />
        <span
          className={`w-11 text-right font-mono text-xs font-bold ${
            dark ? "text-emerald-300" : "text-slate-800"
          }`}
        >
          {opacity}%
        </span>
      </div>
    </div>
  );
}

interface XaiViewportProps {
  mode: ViewMode;
  originalSrc: string;
  activeHeatmapLayer: string | null;
  hasHeatmap: boolean;
  opacity: number;
  splitPos: number;
  onSplitPosChange: (pos: number) => void;
  containerRef: RefObject<HTMLDivElement | null>;
  isLightbox?: boolean;
  onOpenLightbox?: () => void;
}

function XaiViewport({
  mode,
  originalSrc,
  activeHeatmapLayer,
  hasHeatmap,
  opacity,
  splitPos,
  onSplitPosChange,
  containerRef,
  isLightbox = false,
  onOpenLightbox,
}: XaiViewportProps) {
  const [isDragging, setIsDragging] = useState(false);

  const updateSplitFromClientX = (clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return;
    const rawPct = ((clientX - rect.left) / rect.width) * 100;
    onSplitPosChange(Math.max(4, Math.min(96, rawPct)));
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (mode !== "before-after" || !hasHeatmap) return;
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
      onSplitPosChange(Math.max(4, splitPos - 3));
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      onSplitPosChange(Math.min(96, splitPos + 3));
    }
  };

  const lightboxButton = !isLightbox && onOpenLightbox && (
    <button
      type="button"
      onClick={onOpenLightbox}
      onPointerDown={(e) => e.stopPropagation()}
      aria-label="Phóng to ảnh XAI"
      title="Phóng to (Lightbox)"
      className="absolute top-3 right-3 z-30 flex h-8 w-8 items-center justify-center rounded-lg border border-white/20 bg-slate-900/75 text-white shadow-sm backdrop-blur-sm transition-all hover:scale-105 hover:bg-slate-900/95"
    >
      <Maximize2 className="h-4 w-4" />
    </button>
  );

  if (mode === "split") {
    return (
      <div className="relative grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div
          className={`relative overflow-hidden rounded-xl border border-slate-200 bg-slate-950 ${
            isLightbox ? "aspect-[4/3] sm:aspect-square max-h-[64vh]" : "aspect-square"
          }`}
        >
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
        <div
          className={`relative overflow-hidden rounded-xl border border-slate-200 bg-slate-950 ${
            isLightbox ? "aspect-[4/3] sm:aspect-square max-h-[64vh]" : "aspect-square"
          }`}
        >
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
          {lightboxButton}
        </div>
      </div>
    );
  }

  if (mode === "overlay") {
    return (
      <div
        className={`relative w-full overflow-hidden rounded-xl border border-slate-200 bg-slate-950 ${
          isLightbox ? "h-[62vh]" : "aspect-[4/3]"
        }`}
      >
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
        {lightboxButton}
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      className={`relative w-full select-none overflow-hidden rounded-xl border border-slate-200 bg-slate-950 ${
        isLightbox ? "h-[62vh]" : "aspect-[4/3]"
      } ${hasHeatmap ? "cursor-ew-resize touch-none" : ""}`}
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
        <span
          className={`pointer-events-none absolute top-3 rounded-md bg-emerald-700/85 px-2.5 py-1 text-[11px] font-semibold text-white backdrop-blur-sm ${
            !isLightbox && onOpenLightbox ? "right-13" : "right-3"
          }`}
        >
          Bản đồ XAI
        </span>
      )}
      {lightboxButton}

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
  );
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
  const [lightboxOpen, setLightboxOpen] = useState<boolean>(false);

  const inlineContainerRef = useRef<HTMLDivElement>(null);
  const lightboxContainerRef = useRef<HTMLDivElement>(null);

  // Prefer transparent heatmap_only layer for 60fps CSS opacity blending;
  // fall back to pre-blended heatmap_overlay if heatmap_only is absent.
  const activeHeatmapLayer = heatmapOnlySrc || heatmapOverlaySrc;
  const hasHeatmap = Boolean(activeHeatmapLayer);

  useEffect(() => {
    if (!lightboxOpen) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setLightboxOpen(false);
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [lightboxOpen]);

  return (
    <>
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
          <ModeTabs mode={mode} onChange={setMode} />

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
        ) : (
          <XaiViewport
            mode={mode}
            originalSrc={originalSrc}
            activeHeatmapLayer={activeHeatmapLayer}
            hasHeatmap={hasHeatmap}
            opacity={opacity}
            splitPos={splitPos}
            onSplitPosChange={setSplitPos}
            containerRef={inlineContainerRef}
            onOpenLightbox={() => setLightboxOpen(true)}
          />
        )}

        {/* Activation Colorbar directly below image frame & 60fps Opacity Slider */}
        <div className="mt-3.5 space-y-2.5">
          <ActivationColorbar />
          <OpacitySliderControl
            inputId="xai-opacity-slider"
            opacity={opacity}
            onChange={setOpacity}
            disabled={!hasHeatmap}
          />
        </div>
      </div>

      {/* Lightbox Modal */}
      {lightboxOpen && originalSrc && !isLoading && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Phóng to trình kiểm tra XAI"
          onClick={() => setLightboxOpen(false)}
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-sm"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="flex max-h-[94vh] w-full max-w-5xl flex-col gap-3 rounded-2xl border border-slate-800 bg-slate-950 p-4 shadow-2xl sm:p-5"
          >
            {/* Lightbox Header */}
            <div className="flex flex-wrap items-center justify-between gap-2.5">
              <div className="flex flex-wrap items-center gap-3">
                <span className="text-sm font-bold text-white">
                  Trình kiểm tra XAI phóng to
                </span>
                <ModeTabs mode={mode} onChange={setMode} dark />
              </div>

              <div className="flex items-center gap-2">
                {xaiMethod && (
                  <span className="hidden items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-950/70 px-2.5 py-1 font-mono text-[11px] font-semibold text-emerald-300 sm:inline-flex">
                    <Sparkles className="h-3 w-3" />
                    {xaiMethod}
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => setLightboxOpen(false)}
                  aria-label="Đóng phóng to"
                  className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-700 bg-slate-900 text-slate-300 transition-colors hover:bg-slate-800 hover:text-white"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>

            {/* Enlarged Viewport */}
            <XaiViewport
              mode={mode}
              originalSrc={originalSrc}
              activeHeatmapLayer={activeHeatmapLayer}
              hasHeatmap={hasHeatmap}
              opacity={opacity}
              splitPos={splitPos}
              onSplitPosChange={setSplitPos}
              containerRef={lightboxContainerRef}
              isLightbox
            />

            {/* Colorbar + Opacity Control inside Lightbox */}
            <div className="grid grid-cols-1 gap-2.5 lg:grid-cols-2">
              <ActivationColorbar dark />
              <OpacitySliderControl
                inputId="xai-lightbox-opacity-slider"
                opacity={opacity}
                onChange={setOpacity}
                disabled={!hasHeatmap}
                dark
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
