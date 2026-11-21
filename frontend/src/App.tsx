import { useCallback, useEffect, useState } from "react";
import { BatchView } from "./components/BatchView";
import { DiagnoseView } from "./components/DiagnoseView";
import { MonitorView } from "./components/MonitorView";
import { Navbar } from "./components/Navbar";
import { fetchHealth, fetchModelInfo } from "./services/api";
import type { HealthApiResponse, ModelInfoApiResponse, NavTab } from "./types/api";

export function App() {
  const [activeTab, setActiveTab] = useState<NavTab>("diagnose");
  const [health, setHealth] = useState<HealthApiResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoApiResponse | null>(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState<boolean>(false);

  const loadServerStatus = useCallback(async () => {
    setIsCheckingHealth(true);
    try {
      const [healthRes, infoRes] = await Promise.allSettled([fetchHealth(), fetchModelInfo()]);
      setHealth(healthRes.status === "fulfilled" ? healthRes.value : null);
      if (infoRes.status === "fulfilled") {
        setModelInfo(infoRes.value);
      }
    } finally {
      setIsCheckingHealth(false);
    }
  }, []);

  useEffect(() => {
    void loadServerStatus();
    const timer = window.setInterval(() => {
      void loadServerStatus();
    }, 15000);
    return () => window.clearInterval(timer);
  }, [loadServerStatus]);

  return (
    <div className="min-h-screen bg-[#F8FAF9] text-[#0F172A]">
      <Navbar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        health={health}
        modelInfo={modelInfo}
        isCheckingHealth={isCheckingHealth}
        onRefreshStatus={() => void loadServerStatus()}
      />

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:py-8">
        {activeTab === "diagnose" && <DiagnoseView />}
        {activeTab === "batch" && <BatchView />}
        {activeTab === "monitor" && <MonitorView />}
      </main>
    </div>
  );
}

export default App;
