import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ChevronDown,
  Cpu,
  Layers,
  RefreshCw,
  ScanSearch,
  ShieldCheck,
  Sprout,
  X,
} from "lucide-react";
import type { HealthApiResponse, ModelInfoApiResponse, NavTab } from "../types/api";
import { formatConfidence } from "../utils/format";

interface NavbarProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  health: HealthApiResponse | null;
  modelInfo: ModelInfoApiResponse | null;
  isCheckingHealth: boolean;
  onRefreshStatus: () => void;
}

const NAV_ITEMS: Array<{ id: NavTab; label: string; shortLabel: string; icon: typeof ScanSearch }> = [
  { id: "diagnose", label: "Chẩn đoán & XAI", shortLabel: "Chẩn đoán", icon: ScanSearch },
  { id: "batch", label: "Chẩn đoán hàng loạt", shortLabel: "Hàng loạt", icon: Layers },
  { id: "monitor", label: "Giám sát hệ thống", shortLabel: "Giám sát", icon: Activity },
];

export function Navbar({
  activeTab,
  onTabChange,
  health,
  modelInfo,
  isCheckingHealth,
  onRefreshStatus,
}: NavbarProps) {
  const [popoverOpen, setPopoverOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  const isOnline = Boolean(
    (health?.status === "healthy" || health?.status === "ok") && health?.model_loaded,
  );

  useEffect(() => {
    if (!popoverOpen) return;
    const handleClickOutside = (event: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(event.target as Node)) {
        setPopoverOpen(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setPopoverOpen(false);
    };
    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [popoverOpen]);

  return (
    <header className="sticky top-0 z-40 border-b border-slate-200/80 bg-white/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-3 px-4 py-3 sm:px-6">
        {/* Brand */}
        <button
          type="button"
          onClick={() => onTabChange("diagnose")}
          className="group flex items-center gap-2.5 text-left focus:outline-none"
        >
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-emerald-600 text-white shadow-sm shadow-emerald-600/20 transition-transform group-hover:scale-[1.03]">
            <Sprout className="h-5 w-5" />
          </div>
          <div className="hidden sm:block">
            <div className="flex items-center gap-2">
              <span className="text-[15px] font-bold tracking-tight text-slate-900">
                PlantGuard AI
              </span>
              <span className="rounded-md border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-emerald-700">
                Bio-Vision
              </span>
            </div>
            <p className="text-[11px] font-medium text-slate-500">
              Chẩn đoán bệnh lá cây &amp; Giải thích XAI
            </p>
          </div>
        </button>

        {/* Navigation Tabs */}
        <nav
          aria-label="Điều hướng chính"
          className="flex items-center gap-1 rounded-xl border border-slate-200/80 bg-slate-100/70 p-1"
        >
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onTabChange(item.id)}
                className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-semibold transition-all sm:px-3.5 sm:text-[13px] ${
                  active
                    ? "bg-white text-emerald-700 shadow-sm ring-1 ring-slate-200/70"
                    : "text-slate-600 hover:bg-white/60 hover:text-slate-900"
                }`}
              >
                <Icon className={`h-4 w-4 shrink-0 ${active ? "text-emerald-600" : "text-slate-500"}`} />
                <span className="hidden md:inline">{item.label}</span>
                <span className="md:hidden">{item.shortLabel}</span>
              </button>
            );
          })}
        </nav>

        {/* Model Status Pill + Popover */}
        <div className="relative" ref={popoverRef}>
          <button
            type="button"
            onClick={() => setPopoverOpen((prev) => !prev)}
            aria-expanded={popoverOpen}
            aria-haspopup="dialog"
            className={`flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold transition-all ${
              isOnline
                ? "border-emerald-200 bg-emerald-50/80 text-emerald-800 hover:bg-emerald-100/70"
                : "border-rose-200 bg-rose-50/90 text-rose-700 hover:bg-rose-100/80"
            }`}
          >
            <span className="relative flex h-2.5 w-2.5">
              {isOnline && (
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-500 opacity-75" />
              )}
              <span
                className={`relative inline-flex h-2.5 w-2.5 rounded-full ${
                  isOnline ? "bg-emerald-500" : "bg-rose-500"
                }`}
              />
            </span>
            <span className="hidden sm:inline">
              {isOnline ? "Mô hình sẵn sàng" : "Mất kết nối API"}
            </span>
            {modelInfo?.model_version && (
              <span className="hidden rounded bg-white/80 px-1.5 py-0.5 font-mono text-[11px] text-slate-700 lg:inline">
                {modelInfo.model_version}
              </span>
            )}
            <ChevronDown
              className={`h-3.5 w-3.5 transition-transform ${popoverOpen ? "rotate-180" : ""}`}
            />
          </button>

          {popoverOpen && (
            <div
              role="dialog"
              aria-label="Thông tin mô hình AI"
              className="absolute right-0 mt-2 w-[390px] max-w-[calc(100vw-2rem)] origin-top-right rounded-2xl border border-slate-200/80 bg-white p-4 shadow-xl shadow-slate-900/15 ring-1 ring-slate-900/5"
            >
              <div className="flex items-start justify-between gap-2 border-b border-slate-100 pb-3">
                <div className="flex items-center gap-2">
                  <div
                    className={`flex h-8 w-8 items-center justify-center rounded-lg ${
                      isOnline ? "bg-emerald-50 text-emerald-600" : "bg-rose-50 text-rose-600"
                    }`}
                  >
                    <Cpu className="h-4 w-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-slate-900">
                      {modelInfo?.model_name || "PlantGuard Classifier"}
                    </h3>
                    <p className="text-xs text-slate-500">
                      {isOnline ? "Đang hoạt động trực tuyến" : "Backend chưa phản hồi"}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={onRefreshStatus}
                    disabled={isCheckingHealth}
                    title="Kiểm tra lại trạng thái"
                    className="rounded-lg p-1.5 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-700 disabled:opacity-50"
                  >
                    <RefreshCw className={`h-3.5 w-3.5 ${isCheckingHealth ? "animate-spin" : ""}`} />
                  </button>
                  <button
                    type="button"
                    onClick={() => setPopoverOpen(false)}
                    aria-label="Đóng"
                    className="rounded-lg p-1.5 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-700"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>

              {modelInfo ? (
                <div className="mt-3 space-y-3">
                  <div className="grid grid-cols-2 gap-2">
                    <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-2.5">
                      <span className="text-[11px] font-medium text-slate-500">Val Macro-F1</span>
                      <p className="mt-0.5 font-mono text-sm font-bold text-emerald-700">
                        {(
                          modelInfo.val_macro_f1 > 1
                            ? modelInfo.val_macro_f1
                            : modelInfo.val_macro_f1 * 100
                        ).toFixed(2)}
                        %
                      </p>
                      <p className="mt-0.5 text-[10px] font-medium text-slate-500">
                        17.572 ảnh · 38 lớp Validation
                      </p>
                    </div>
                    <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-2.5">
                      <span className="text-[11px] font-medium text-slate-500">Test Accuracy</span>
                      <p className="mt-0.5 font-mono text-sm font-bold text-slate-900">
                        {Math.round(
                          (modelInfo.test_accuracy > 1
                            ? modelInfo.test_accuracy / 100
                            : modelInfo.test_accuracy) * 33,
                        )}
                        /33 mẫu
                      </p>
                      <p className="mt-0.5 text-[10px] font-medium text-slate-500">
                        {Math.round(
                          modelInfo.test_accuracy > 1
                            ? modelInfo.test_accuracy
                            : modelInfo.test_accuracy * 100,
                        )}
                        % · Tập test độc lập
                      </p>
                    </div>
                    <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-2.5">
                      <span className="text-[11px] font-medium text-slate-500">Kích thước</span>
                      <p className="mt-0.5 font-mono text-sm font-semibold text-slate-900">
                        {modelInfo.model_size_mb.toFixed(1)} MB
                      </p>
                    </div>
                    <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-2.5">
                      <span className="text-[11px] font-medium text-slate-500">Số lớp bệnh</span>
                      <p className="mt-0.5 font-mono text-sm font-semibold text-slate-900">
                        {modelInfo.num_classes} lớp
                      </p>
                    </div>
                  </div>

                  <dl className="space-y-1.5 text-xs">
                    <div className="flex items-center justify-between gap-3">
                      <dt className="whitespace-nowrap shrink-0 text-slate-500">Kiến trúc</dt>
                      <dd className="text-right font-mono font-medium text-slate-800">
                        {modelInfo.architecture}
                      </dd>
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <dt className="whitespace-nowrap shrink-0 text-slate-500">
                        Runtime / Thiết bị
                      </dt>
                      <dd className="text-right font-mono font-medium text-slate-800">
                        {modelInfo.backend_type} · {modelInfo.device}
                      </dd>
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <dt className="whitespace-nowrap shrink-0 text-slate-500">
                        Phương pháp XAI
                      </dt>
                      <dd className="text-right font-mono font-medium text-emerald-700">
                        {modelInfo.xai_method}
                      </dd>
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <dt className="whitespace-nowrap shrink-0 text-slate-500">
                        Ngưỡng tin cậy
                      </dt>
                      <dd className="text-right font-mono font-medium text-slate-800">
                        {formatConfidence(modelInfo.confidence_threshold)}
                      </dd>
                    </div>
                  </dl>

                  {modelInfo.providers?.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1.5 border-t border-slate-100 pt-2.5">
                      <ShieldCheck className="h-3.5 w-3.5 text-emerald-600" />
                      {modelInfo.providers.map((provider) => (
                        <span
                          key={provider}
                          className="rounded-md bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] font-medium text-slate-600"
                        >
                          {provider}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="mt-3 rounded-xl border border-rose-200 bg-rose-50/70 p-3 text-xs text-rose-700">
                  Không thể tải thông tin từ <span className="font-mono">/api/model-info</span>. Hãy
                  đảm bảo FastAPI server đang chạy tại cổng <span className="font-mono">8000</span>.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
