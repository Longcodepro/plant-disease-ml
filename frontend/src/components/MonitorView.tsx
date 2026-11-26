import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Clock,
  DatabaseZap,
  RefreshCw,
  Server,
  ShieldAlert,
  TrendingUp,
} from "lucide-react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  LabelList,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchMetrics } from "../services/api";
import type { MetricsApiResponse } from "../types/api";
import { formatConfidence, formatMs, formatTimestampTime, formatUptime } from "../utils/format";

interface ClassBarDatum {
  name: string;
  shortName: string;
  count: number;
}

interface HistoryChartDatum {
  idx: number;
  time: string;
  latencyMs: number;
  confidencePct: number;
  label: string;
  cacheHit: boolean;
}

interface CustomTooltipPayloadItem {
  name?: string;
  value?: number;
  color?: string;
  dataKey?: string;
  payload?: HistoryChartDatum | ClassBarDatum;
}

function HistoryTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: CustomTooltipPayloadItem[];
}) {
  if (!active || !payload || payload.length === 0) return null;
  const point = payload[0]?.payload as HistoryChartDatum | undefined;
  if (!point) return null;

  return (
    <div className="rounded-xl border border-slate-200 bg-white/95 p-3 shadow-lg backdrop-blur-sm">
      <div className="mb-1.5 flex items-center justify-between gap-4 border-b border-slate-100 pb-1">
        <span className="font-mono text-[11px] font-bold text-slate-800">{point.time}</span>
        {point.cacheHit && (
          <span className="rounded bg-sky-50 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-sky-700">
            Cache
          </span>
        )}
      </div>
      <p className="mb-1.5 max-w-[210px] truncate text-xs font-semibold text-slate-700">
        {point.label}
      </p>
      <div className="space-y-1 font-mono text-xs">
        <div className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5 text-slate-500">
            <span className="h-2 w-2 rounded-full bg-[#059669]" />
            Latency:
          </span>
          <span className="font-bold text-emerald-700">{point.latencyMs.toFixed(1)} ms</span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5 text-slate-500">
            <span className="h-2 w-2 rounded-full bg-[#2563eb]" />
            Độ tin cậy:
          </span>
          <span className="font-bold text-[#2563eb]">{point.confidencePct.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}

function ClassBarTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: CustomTooltipPayloadItem[];
}) {
  if (!active || !payload || payload.length === 0) return null;
  const item = payload[0]?.payload as ClassBarDatum | undefined;
  if (!item) return null;

  return (
    <div className="rounded-xl border border-slate-200 bg-white/95 px-3 py-2 shadow-lg backdrop-blur-sm">
      <p className="text-xs font-semibold text-slate-800">{item.name}</p>
      <p className="mt-0.5 font-mono text-xs font-bold text-emerald-700">{item.count} lượt</p>
    </div>
  );
}

export function MonitorView() {
  const [metrics, setMetrics] = useState<MetricsApiResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  const loadMetrics = useCallback(async (isInitial = false) => {
    if (isInitial) setLoading(true);
    setIsRefreshing(true);
    setError(null);
    try {
      const data = await fetchMetrics();
      setMetrics(data);
      setLastUpdated(
        new Date().toLocaleTimeString("vi-VN", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Không thể tải dữ liệu giám sát.");
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void loadMetrics(true);
  }, [loadMetrics]);

  useEffect(() => {
    if (!autoRefresh) return;
    const timer = window.setInterval(() => {
      void loadMetrics(false);
    }, 5000);
    return () => window.clearInterval(timer);
  }, [autoRefresh, loadMetrics]);

  const classDistribution = useMemo<ClassBarDatum[]>(() => {
    if (!metrics?.top_predicted_classes) return [];
    return Object.entries(metrics.top_predicted_classes)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([name, count]) => {
        const cleaned = name.replace(/___/g, " · ").replace(/_/g, " ");
        const shortName = cleaned.length > 26 ? `${cleaned.slice(0, 24)}…` : cleaned;
        return { name: cleaned, shortName, count };
      });
  }, [metrics]);

  const historySeries = useMemo<HistoryChartDatum[]>(() => {
    if (!metrics?.history) return [];
    return metrics.history.map((item, idx) => ({
      idx: idx + 1,
      time: formatTimestampTime(item.timestamp),
      latencyMs: Number(item.latency_ms.toFixed(1)),
      confidencePct: Number(
        (item.confidence > 1 ? item.confidence : item.confidence * 100).toFixed(1),
      ),
      label: `${item.plant} · ${item.condition}`,
      cacheHit: item.cache_hit,
    }));
  }, [metrics]);

  const barChartHeight = Math.max(220, classDistribution.length * 44 + 44);

  return (
    <div className="space-y-6">
      {/* Header Toolbar */}
      <section className="flex flex-col justify-between gap-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:flex-row sm:items-center">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-emerald-600" />
            <h1 className="text-lg font-extrabold tracking-tight text-slate-900">
              Giám sát hiệu năng &amp; Drift mô hình
            </h1>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-slate-500">
            {metrics && (
              <>
                <span className="inline-flex items-center gap-1 font-mono">
                  <Server className="h-3.5 w-3.5 text-slate-400" />
                  {metrics.backend_type} · {metrics.model_version}
                </span>
                <span className="font-mono">Uptime: {formatUptime(metrics.uptime_seconds)}</span>
              </>
            )}
            {lastUpdated && (
              <span className="font-mono text-slate-400">Cập nhật lúc {lastUpdated}</span>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          {/* Auto refresh 5s toggle */}
          <button
            type="button"
            role="switch"
            aria-checked={autoRefresh}
            onClick={() => setAutoRefresh((prev) => !prev)}
            className={`inline-flex items-center gap-2 rounded-xl border px-3.5 py-2 text-xs font-semibold transition-all ${
              autoRefresh
                ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                : "border-slate-200 bg-slate-50 text-slate-600 hover:bg-slate-100"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                autoRefresh ? "animate-pulse bg-emerald-500" : "bg-slate-400"
              }`}
            />
            <span>Tự động làm mới (5s)</span>
          </button>

          <button
            type="button"
            onClick={() => void loadMetrics(false)}
            disabled={isRefreshing}
            className="inline-flex items-center gap-1.5 rounded-xl bg-emerald-600 px-3.5 py-2 text-xs font-bold text-white shadow-sm shadow-emerald-600/20 transition-all hover:bg-emerald-700 disabled:opacity-60"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? "animate-spin" : ""}`} />
            <span>Làm mới</span>
          </button>
        </div>
      </section>

      {error && (
        <div
          role="alert"
          className="flex items-start gap-3 rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800"
        >
          <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-rose-600" />
          <div>
            <p className="font-bold">Không thể tải dữ liệu giám sát</p>
            <p className="mt-0.5 text-xs text-rose-700">{error}</p>
          </div>
        </div>
      )}

      {/* 4 SaaS KPI Cards */}
      {loading && !metrics ? (
        <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[1, 2, 3, 4].map((idx) => (
            <div
              key={idx}
              className="h-28 animate-pulse rounded-2xl border border-slate-200 bg-white p-5"
            />
          ))}
        </section>
      ) : (
        metrics && (
          <>
            <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-500">Tổng Requests</span>
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-50 text-emerald-600">
                    <Activity className="h-4 w-4" />
                  </div>
                </div>
                <p className="mt-2 font-mono text-2xl font-bold text-slate-900">
                  {metrics.total_requests.toLocaleString("vi-VN")}
                </p>
                <p className="mt-1 font-mono text-[11px] text-slate-500">
                  {metrics.total_images_predicted.toLocaleString("vi-VN")} ảnh đã phân tích
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-500">Latency p50 / p95</span>
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100 text-slate-700">
                    <Clock className="h-4 w-4" />
                  </div>
                </div>
                <p className="mt-2 font-mono text-2xl font-bold text-slate-900">
                  {formatMs(metrics.latency_ms.p50)}{" "}
                  <span className="text-base font-medium text-slate-400">/</span>{" "}
                  <span className="text-emerald-700">{formatMs(metrics.latency_ms.p95)}</span>
                </p>
                <p className="mt-1 font-mono text-[11px] text-slate-500">
                  Avg {formatMs(metrics.latency_ms.avg)} · Max {formatMs(metrics.latency_ms.max)}
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-500">Cache Hit Rate</span>
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-sky-50 text-sky-600">
                    <DatabaseZap className="h-4 w-4" />
                  </div>
                </div>
                <p className="mt-2 font-mono text-2xl font-bold text-sky-700">
                  {formatConfidence(metrics.cache_hit_rate)}
                </p>
                <p className="mt-1 font-mono text-[11px] text-slate-500">
                  {metrics.cache_hits} lượt truy xuất bộ nhớ đệm
                </p>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-500">
                    Uncertain / Drift Rate
                  </span>
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-50 text-amber-600">
                    <AlertTriangle className="h-4 w-4" />
                  </div>
                </div>
                <p className="mt-2 font-mono text-2xl font-bold text-amber-700">
                  {formatConfidence(metrics.uncertain_rate)}
                </p>
                <p className="mt-1 font-mono text-[11px] text-slate-500">
                  {metrics.uncertain_count} ảnh dưới ngưỡng tin cậy
                </p>
              </div>
            </section>

            {/* 2 Recharts Visualizations */}
            <section className="grid grid-cols-1 gap-6 lg:grid-cols-12">
              {/* Horizontal Bar Chart: Top Predicted Classes */}
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm lg:col-span-5">
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-emerald-600" />
                    <h2 className="text-sm font-bold text-slate-900">
                      Phân bố nhóm bệnh dự đoán
                    </h2>
                  </div>
                  <span className="font-mono text-[11px] text-slate-400">
                    Top {classDistribution.length}
                  </span>
                </div>

                {classDistribution.length === 0 ? (
                  <div className="flex h-56 flex-col items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50/60 p-6 text-center">
                    <BarChart3 className="mb-2 h-6 w-6 text-slate-400" />
                    <p className="text-xs font-semibold text-slate-600">
                      Chưa có dữ liệu chẩn đoán
                    </p>
                    <p className="mt-1 text-[11px] text-slate-400">
                      Thực hiện vài lượt chẩn đoán để xem biểu đồ phân bố lớp bệnh.
                    </p>
                  </div>
                ) : (
                  <div style={{ height: barChartHeight }} className="w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={classDistribution}
                        layout="vertical"
                        margin={{ top: 4, right: 36, left: 4, bottom: 4 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
                        <XAxis
                          type="number"
                          allowDecimals={false}
                          tick={{ fill: "#64748B", fontSize: 11, fontFamily: "JetBrains Mono" }}
                          axisLine={{ stroke: "#E2E8F0" }}
                          tickLine={false}
                        />
                        <YAxis
                          type="category"
                          dataKey="shortName"
                          width={145}
                          tick={{ fill: "#334155", fontSize: 11, fontWeight: 500 }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <Tooltip
                          cursor={{ fill: "#F1F5F9" }}
                          content={<ClassBarTooltip />}
                        />
                        <Bar
                          dataKey="count"
                          barSize={24}
                          radius={[0, 6, 6, 0]}
                          fill="#059669"
                        >
                          <LabelList
                            dataKey="count"
                            position="right"
                            offset={8}
                            style={{
                              fill: "#334155",
                              fontSize: 11,
                              fontWeight: 700,
                              fontFamily: "JetBrains Mono, monospace",
                            }}
                          />
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>

              {/* Composed Chart: Latency & Confidence History */}
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm lg:col-span-7">
                <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-4 w-4 text-emerald-600" />
                    <h2 className="text-sm font-bold text-slate-900">
                      Biến động Latency (ms) &amp; Độ tin cậy (%)
                    </h2>
                  </div>
                  <div className="flex items-center gap-4 text-[11px] font-semibold">
                    <span className="flex items-center gap-1.5 text-emerald-700">
                      <span className="h-2.5 w-2.5 rounded-full bg-[#059669]" />
                      Latency (ms)
                    </span>
                    <span className="flex items-center gap-1.5 text-[#2563eb]">
                      <span className="h-2.5 w-2.5 rounded-full bg-[#2563eb]" />
                      Độ tin cậy (%)
                    </span>
                  </div>
                </div>

                {historySeries.length === 0 ? (
                  <div className="flex h-64 flex-col items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50/60 p-6 text-center">
                    <TrendingUp className="mb-2 h-6 w-6 text-slate-400" />
                    <p className="text-xs font-semibold text-slate-600">
                      Chưa có lịch sử suy luận theo thời gian
                    </p>
                    <p className="mt-1 text-[11px] text-slate-400">
                      Dữ liệu latency và độ tin cậy sẽ được ghi nhận tự động sau mỗi yêu cầu.
                    </p>
                  </div>
                ) : (
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart
                        data={historySeries}
                        margin={{ top: 8, right: 8, left: 0, bottom: 4 }}
                      >
                        <defs>
                          <linearGradient id="latencyGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#059669" stopOpacity={0.28} />
                            <stop offset="95%" stopColor="#059669" stopOpacity={0.02} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                        <XAxis
                          dataKey="time"
                          tick={{ fill: "#64748B", fontSize: 11, fontFamily: "JetBrains Mono" }}
                          axisLine={{ stroke: "#E2E8F0" }}
                          tickLine={false}
                        />
                        <YAxis
                          yAxisId="left"
                          padding={{ top: 10, bottom: 4 }}
                          tick={{ fill: "#059669", fontSize: 11, fontFamily: "JetBrains Mono" }}
                          axisLine={false}
                          tickLine={false}
                          unit="ms"
                          width={52}
                        />
                        <YAxis
                          yAxisId="right"
                          orientation="right"
                          domain={[0, 100]}
                          ticks={[0, 25, 50, 75, 100]}
                          padding={{ top: 10, bottom: 4 }}
                          tick={{ fill: "#2563eb", fontSize: 11, fontFamily: "JetBrains Mono" }}
                          axisLine={false}
                          tickLine={false}
                          unit="%"
                          width={48}
                        />
                        <Tooltip content={<HistoryTooltip />} />
                        <Area
                          yAxisId="left"
                          type="monotone"
                          dataKey="latencyMs"
                          stroke="#059669"
                          strokeWidth={2}
                          fill="url(#latencyGradient)"
                          dot={{ r: 4, stroke: "#059669", strokeWidth: 2, fill: "#ffffff" }}
                          activeDot={{ r: 6, stroke: "#059669", strokeWidth: 2, fill: "#ffffff" }}
                        />
                        <Line
                          yAxisId="right"
                          type="monotone"
                          dataKey="confidencePct"
                          stroke="#2563eb"
                          strokeWidth={2}
                          dot={{ r: 4, stroke: "#2563eb", strokeWidth: 2, fill: "#ffffff" }}
                          activeDot={{ r: 6, stroke: "#2563eb", strokeWidth: 2, fill: "#ffffff" }}
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            </section>
          </>
        )
      )}
    </div>
  );
}
