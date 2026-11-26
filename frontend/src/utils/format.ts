import type { Advisory } from "../types/api";

export function toImageDataUri(raw: string | null | undefined): string | null {
  if (!raw) return null;
  if (raw.startsWith("data:image/")) return raw;
  return `data:image/png;base64,${raw}`;
}

export function formatConfidence(value: number): string {
  const normalized = value > 1 ? value : value * 100;
  return `${normalized.toFixed(1)}%`;
}

export function toPercentNumber(value: number): number {
  const normalized = value > 1 ? value : value * 100;
  return Math.max(0, Math.min(100, normalized));
}

export function formatMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return "—";
  if (ms < 1) return `${ms.toFixed(2)} ms`;
  return `${Math.round(ms)} ms`;
}

export function formatUptime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}g ${m}p`;
  if (m > 0) return `${m}p ${s}s`;
  return `${s}s`;
}

export function getSeverityMeta(severity: Advisory["severity"]): {
  label: string;
  className: string;
} {
  switch (severity) {
    case "high":
      return {
        label: "Mức độ: Nghiêm trọng",
        className: "bg-rose-50 text-rose-700 border-rose-200",
      };
    case "medium":
      return {
        label: "Mức độ: Trung bình",
        className: "bg-amber-50 text-amber-700 border-amber-200",
      };
    case "none":
    default:
      return {
        label: "Mức độ: An toàn",
        className: "bg-emerald-50 text-emerald-700 border-emerald-200",
      };
  }
}

export function formatTimestampTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleTimeString("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function escapeCsvCell(value: string | number | boolean): string {
  const str = String(value ?? "");
  if (str.includes(",") || str.includes('"') || str.includes("\n")) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}
