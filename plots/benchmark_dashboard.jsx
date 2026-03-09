import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, ReferenceLine,
} from "recharts";

/* ===== STENCIL DATA ===== */
const stencilRaw = {
  "2 GPUs": [
    { grid: "1024²", ipc_tp: 8.11, mpi_tp: 8.00, ipc_t: 12.93, mpi_t: 13.10 },
    { grid: "2048²", ipc_tp: 31.32, mpi_tp: 29.59, ipc_t: 13.39, mpi_t: 14.17 },
    { grid: "4096²", ipc_tp: 100.52, mpi_tp: 91.45, ipc_t: 16.69, mpi_t: 18.35 },
    { grid: "8192²", ipc_tp: 223.95, mpi_tp: 206.97, ipc_t: 29.97, mpi_t: 32.42 },
    { grid: "16384²", ipc_tp: 326.18, mpi_tp: 312.12, ipc_t: 82.30, mpi_t: 86.00 },
    { grid: "32768²", ipc_tp: 367.56, mpi_tp: 359.92, ipc_t: 292.12, mpi_t: 298.33 },
  ],
  "4 GPUs": [
    { grid: "1024²", ipc_tp: 6.83, mpi_tp: 6.22, ipc_t: 15.36, mpi_t: 16.86 },
    { grid: "2048²", ipc_tp: 28.51, mpi_tp: 23.15, ipc_t: 14.71, mpi_t: 18.12 },
    { grid: "4096²", ipc_tp: 99.76, mpi_tp: 77.26, ipc_t: 16.82, mpi_t: 21.72 },
    { grid: "8192²", ipc_tp: 262.88, mpi_tp: 216.57, ipc_t: 25.53, mpi_t: 30.99 },
    { grid: "16384²", ipc_tp: 535.68, mpi_tp: 431.05, ipc_t: 50.11, mpi_t: 62.28 },
    { grid: "32768²", ipc_tp: 691.41, mpi_tp: 642.64, ipc_t: 155.30, mpi_t: 167.08 },
  ],
};

/* ===== TRANSPOSE DATA ===== */
const transposeRaw = {
  "2 GPUs": [
    { grid: "1024²", ipc_rate: 389305, mpigpu_rate: 361894, staged_rate: 92365, ipc_t: 0.043, mpigpu_t: 0.046, staged_t: 0.182 },
    { grid: "2048²", ipc_rate: 612168, mpigpu_rate: 581289, staged_rate: 104727, ipc_t: 0.110, mpigpu_t: 0.115, staged_t: 0.641 },
    { grid: "4096²", ipc_rate: 727030, mpigpu_rate: 693650, staged_rate: 108390, ipc_t: 0.369, mpigpu_t: 0.387, staged_t: 2.477 },
    { grid: "8192²", ipc_rate: 769141, mpigpu_rate: 738579, staged_rate: 105626, ipc_t: 1.396, mpigpu_t: 1.454, staged_t: 10.165 },
    { grid: "16384²", ipc_rate: 772370, mpigpu_rate: 742027, staged_rate: 104282, ipc_t: 5.561, mpigpu_t: 5.788, staged_t: 41.186 },
  ],
  "4 GPUs": [
    { grid: "1024²", ipc_rate: 203647, mpigpu_rate: 197011, staged_rate: 75088, ipc_t: 0.082, mpigpu_t: 0.085, staged_t: 0.223 },
    { grid: "2048²", ipc_rate: 545046, mpigpu_rate: 535236, staged_rate: 106605, ipc_t: 0.123, mpigpu_t: 0.125, staged_t: 0.630 },
    { grid: "4096²", ipc_rate: 855314, mpigpu_rate: 849238, staged_rate: 100828, ipc_t: 0.314, mpigpu_t: 0.316, staged_t: 2.662 },
    { grid: "8192²", ipc_rate: 1025309, mpigpu_rate: 1022559, staged_rate: 121278, ipc_t: 1.047, mpigpu_t: 1.050, staged_t: 8.854 },
    { grid: "16384²", ipc_rate: 1064734, mpigpu_rate: 1076757, staged_rate: 101967, ipc_t: 4.034, mpigpu_t: 3.989, staged_t: 42.121 },
  ],
};

const enrichStencil = (arr) =>
  arr.map((d) => ({ ...d, speedup: +((d.mpi_t - d.ipc_t) / d.mpi_t * 100).toFixed(1) }));

const enrichTranspose = (arr) =>
  arr.map((d) => ({
    ...d,
    ipc_gb: +(d.ipc_rate / 1000).toFixed(1),
    mpigpu_gb: +(d.mpigpu_rate / 1000).toFixed(1),
    staged_gb: +(d.staged_rate / 1000).toFixed(1),
    ipc_vs_mpigpu: +((d.mpigpu_t - d.ipc_t) / d.mpigpu_t * 100).toFixed(1),
    ipc_vs_staged: +((d.staged_t - d.ipc_t) / d.staged_t * 100).toFixed(1),
  }));

const stencilData = { "2 GPUs": enrichStencil(stencilRaw["2 GPUs"]), "4 GPUs": enrichStencil(stencilRaw["4 GPUs"]) };
const transposeData = { "2 GPUs": enrichTranspose(transposeRaw["2 GPUs"]), "4 GPUs": enrichTranspose(transposeRaw["4 GPUs"]) };

const avg = (arr, key) => (arr.reduce((s, d) => s + d[key], 0) / arr.length).toFixed(1);
const best = (arr, key) => Math.max(...arr.map((d) => d[key])).toFixed(1);

const CustomTooltip = ({ active, payload, label, unit }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#1a1a2e", border: "1px solid rgba(255,255,255,0.12)",
      borderRadius: 8, padding: "10px 14px", fontSize: 13, color: "#e0e0e0",
      boxShadow: "0 4px 20px rgba(0,0,0,0.4)", fontFamily: "'JetBrains Mono', monospace",
    }}>
      <div style={{ fontWeight: 600, marginBottom: 6, color: "#fff" }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: p.color, display: "inline-block" }} />
          <span>{p.name}: <strong>{p.value} {unit}</strong></span>
        </div>
      ))}
    </div>
  );
};

const Card = ({ label, value, sub }) => (
  <div style={{
    background: "rgba(255,255,255,0.04)", borderRadius: 10, padding: "14px 16px",
    border: "1px solid rgba(255,255,255,0.06)",
  }}>
    <div style={{ fontSize: 11, color: "#6b7094", textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 4 }}>{label}</div>
    <div style={{ fontSize: 22, fontWeight: 700, color: "#fff" }}>{value}</div>
    <div style={{ fontSize: 11, color: "#525672", marginTop: 2 }}>{sub}</div>
  </div>
);

const TabBar = ({ tabs, active, onChange }) => (
  <div style={{ display: "flex", gap: 4, background: "rgba(255,255,255,0.03)", borderRadius: 8, padding: 3 }}>
    {tabs.map((t) => (
      <button key={t.key} onClick={() => onChange(t.key)} style={{
        flex: 1, padding: "8px 0", borderRadius: 6, border: "none", cursor: "pointer",
        fontSize: 12, fontWeight: 600, fontFamily: "inherit", letterSpacing: 0.5, transition: "all 0.2s",
        background: active === t.key ? "rgba(96,165,250,0.15)" : "transparent",
        color: active === t.key ? "#60a5fa" : "#6b7094",
      }}>{t.label}</button>
    ))}
  </div>
);

const GpuToggle = ({ value, onChange }) => (
  <div style={{ display: "flex", gap: 2, background: "rgba(255,255,255,0.04)", borderRadius: 8, padding: 3 }}>
    {["2 GPUs", "4 GPUs"].map((g) => (
      <button key={g} onClick={() => onChange(g)} style={{
        padding: "6px 14px", borderRadius: 6, border: "none", cursor: "pointer",
        fontSize: 12, fontWeight: 600, fontFamily: "inherit", transition: "all 0.2s",
        background: value === g ? "rgba(167,139,250,0.2)" : "transparent",
        color: value === g ? "#a78bfa" : "#6b7094",
      }}>{g}</button>
    ))}
  </div>
);

const ChartBox = ({ children }) => (
  <div style={{
    background: "rgba(255,255,255,0.02)", borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.06)", padding: "20px 16px 8px 4px",
  }}>{children}</div>
);

export default function Benchmark() {
  const [bench, setBench] = useState("transpose");
  const [stencilGpus, setStencilGpus] = useState("4 GPUs");
  const [stencilView, setStencilView] = useState("throughput");
  const [transGpus, setTransGpus] = useState("4 GPUs");
  const [transView, setTransView] = useState("rate");

  const sData = stencilData[stencilGpus];
  const tData = transposeData[transGpus];
  const peakTransRate = Math.max(...tData.map(d => d.ipc_rate));
  const peakTransGB = (peakTransRate / 1000).toFixed(0);

  return (
    <div style={{
      minHeight: "100vh", background: "#0f0f1a", color: "#e8e8ed",
      fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace", padding: "32px 28px",
    }}>
      <div style={{ maxWidth: 780, margin: "0 auto" }}>
        {/* Top selector */}
        <div style={{ display: "flex", gap: 2, background: "rgba(255,255,255,0.04)", borderRadius: 8, padding: 3, marginBottom: 24 }}>
          {[{ key: "stencil", label: "Stencil" }, { key: "transpose", label: "Transpose" }].map((b) => (
            <button key={b.key} onClick={() => setBench(b.key)} style={{
              flex: 1, padding: "10px 0", borderRadius: 6, border: "none", cursor: "pointer",
              fontSize: 13, fontWeight: 700, fontFamily: "inherit", letterSpacing: 0.5, transition: "all 0.2s",
              background: bench === b.key ? "rgba(167,139,250,0.2)" : "transparent",
              color: bench === b.key ? "#a78bfa" : "#6b7094",
            }}>{b.label}</button>
          ))}
        </div>

        {/* ===== STENCIL ===== */}
        {bench === "stencil" && (<>
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 3, color: "#6b7094", marginBottom: 6 }}>
              H200 • Stencil Computation
            </div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <h1 style={{ fontSize: 26, fontWeight: 700, margin: 0, letterSpacing: -0.5,
                background: "linear-gradient(135deg, #60a5fa, #a78bfa)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              }}>CUDA IPC vs MPI</h1>
              <GpuToggle value={stencilGpus} onChange={setStencilGpus} />
            </div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 24 }}>
            <Card label="Peak IPC" value={`${Math.max(...sData.map(d => d.ipc_tp))} B/s`} sub="32768²" />
            <Card label="Best Speedup" value={`${best(sData, 'speedup')}%`} sub="" />
            <Card label="Avg Speedup" value={`${avg(sData, 'speedup')}%`} sub="across sizes" />
          </div>
          <div style={{ marginBottom: 20 }}>
            <TabBar tabs={[
              { key: "throughput", label: "Throughput" }, { key: "time", label: "Time" }, { key: "speedup", label: "Speedup" },
            ]} active={stencilView} onChange={setStencilView} />
          </div>
          <ChartBox>
            {stencilView === "throughput" && (
              <ResponsiveContainer width="100%" height={360}>
                <LineChart data={sData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="grid" tick={{ fill: "#6b7094", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }} />
                  <YAxis tick={{ fill: "#6b7094", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                    label={{ value: "B cells/sec", angle: -90, position: "insideLeft", offset: 4, style: { fill: "#525672", fontSize: 11 } }} />
                  <Tooltip content={<CustomTooltip unit="B cells/s" />} />
                  <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                  <Line type="monotone" dataKey="ipc_tp" name="CUDA IPC" stroke="#60a5fa" strokeWidth={2.5} dot={{ r: 4, fill: "#60a5fa" }} />
                  <Line type="monotone" dataKey="mpi_tp" name="MPI" stroke="#f97316" strokeWidth={2.5} dot={{ r: 4, fill: "#f97316" }} strokeDasharray="6 3" />
                </LineChart>
              </ResponsiveContainer>
            )}
            {stencilView === "time" && (
              <ResponsiveContainer width="100%" height={360}>
                <BarChart data={sData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }} barCategoryGap="28%">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="grid" tick={{ fill: "#6b7094", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }} />
                  <YAxis tick={{ fill: "#6b7094", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                    label={{ value: "Time (ms)", angle: -90, position: "insideLeft", offset: 4, style: { fill: "#525672", fontSize: 11 } }} />
                  <Tooltip content={<CustomTooltip unit="ms" />} />
                  <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                  <Bar dataKey="ipc_t" name="CUDA IPC" fill="#60a5fa" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="mpi_t" name="MPI" fill="#f97316" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
            {stencilView === "speedup" && (
              <ResponsiveContainer width="100%" height={360}>
                <BarChart data={sData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="grid" tick={{ fill: "#6b7094", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }} />
                  <YAxis tick={{ fill: "#6b7094", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }} domain={[0, 'auto']}
                    label={{ value: "IPC speedup (%)", angle: -90, position: "insideLeft", offset: 4, style: { fill: "#525672", fontSize: 11 } }} />
                  <Tooltip content={<CustomTooltip unit="%" />} />
                  <ReferenceLine y={+avg(sData, 'speedup')} stroke="#a78bfa" strokeDasharray="5 5" strokeWidth={1.5}
                    label={{ value: `avg ${avg(sData, 'speedup')}%`, position: "right", fill: "#a78bfa", fontSize: 11 }} />
                  <Bar dataKey="speedup" name="IPC Speedup" fill="url(#sGrad)" radius={[4, 4, 0, 0]} />
                  <defs><linearGradient id="sGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#60a5fa" /><stop offset="100%" stopColor="#a78bfa" />
                  </linearGradient></defs>
                </BarChart>
              </ResponsiveContainer>
            )}
          </ChartBox>
        </>)}

        {/* ===== TRANSPOSE ===== */}
        {bench === "transpose" && (<>
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 3, color: "#6b7094", marginBottom: 6 }}>
              H200 • Matrix Transpose
            </div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <h1 style={{ fontSize: 26, fontWeight: 700, margin: 0, letterSpacing: -0.5,
                background: "linear-gradient(135deg, #34d399, #60a5fa)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              }}>IPC vs GPU-aware MPI vs Staged</h1>
              <GpuToggle value={transGpus} onChange={setTransGpus} />
            </div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 24 }}>
            <Card label="Peak IPC Rate" value={`${peakTransGB} GB/s`} sub="largest matrix" />
            <Card label="IPC vs GPU-MPI" value={`${avg(tData, 'ipc_vs_mpigpu')}%`} sub="avg speedup" />
            <Card label="IPC vs Staged" value={`${avg(tData, 'ipc_vs_staged')}%`} sub="avg speedup" />
          </div>
          <div style={{ marginBottom: 20 }}>
            <TabBar tabs={[
              { key: "rate", label: "Bandwidth" }, { key: "time", label: "Time" }, { key: "speedup", label: "Speedup" },
            ]} active={transView} onChange={setTransView} />
          </div>
          <ChartBox>
            {transView === "rate" && (
              <ResponsiveContainer width="100%" height={360}>
                <LineChart data={tData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="grid" tick={{ fill: "#6b7094", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }} />
                  <YAxis tick={{ fill: "#6b7094", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                    label={{ value: "GB/s", angle: -90, position: "insideLeft", offset: 4, style: { fill: "#525672", fontSize: 11 } }} />
                  <Tooltip content={<CustomTooltip unit="GB/s" />} />
                  <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                  <Line type="monotone" dataKey="ipc_gb" name="CUDA IPC" stroke="#34d399" strokeWidth={2.5} dot={{ r: 4, fill: "#34d399" }} />
                  <Line type="monotone" dataKey="mpigpu_gb" name="GPU-aware MPI" stroke="#60a5fa" strokeWidth={2.5} dot={{ r: 4, fill: "#60a5fa" }} strokeDasharray="6 3" />
                  <Line type="monotone" dataKey="staged_gb" name="Staged MPI" stroke="#f97316" strokeWidth={2.5} dot={{ r: 4, fill: "#f97316" }} strokeDasharray="3 3" />
                </LineChart>
              </ResponsiveContainer>
            )}
            {transView === "time" && (
              <ResponsiveContainer width="100%" height={360}>
                <BarChart data={tData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }} barCategoryGap="20%">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="grid" tick={{ fill: "#6b7094", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }} />
                  <YAxis tick={{ fill: "#6b7094", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                    label={{ value: "Time (ms)", angle: -90, position: "insideLeft", offset: 4, style: { fill: "#525672", fontSize: 11 } }} />
                  <Tooltip content={<CustomTooltip unit="ms" />} />
                  <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                  <Bar dataKey="ipc_t" name="CUDA IPC" fill="#34d399" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="mpigpu_t" name="GPU-aware MPI" fill="#60a5fa" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="staged_t" name="Staged MPI" fill="#f97316" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
            {transView === "speedup" && (
              <ResponsiveContainer width="100%" height={360}>
                <BarChart data={tData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }} barCategoryGap="28%">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="grid" tick={{ fill: "#6b7094", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }} />
                  <YAxis tick={{ fill: "#6b7094", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }} domain={[0, 100]}
                    label={{ value: "IPC speedup (%)", angle: -90, position: "insideLeft", offset: 4, style: { fill: "#525672", fontSize: 11 } }} />
                  <Tooltip content={<CustomTooltip unit="%" />} />
                  <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                  <Bar dataKey="ipc_vs_mpigpu" name="vs GPU-aware MPI" fill="url(#tGrad1)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="ipc_vs_staged" name="vs Staged MPI" fill="url(#tGrad2)" radius={[4, 4, 0, 0]} />
                  <defs>
                    <linearGradient id="tGrad1" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#34d399" /><stop offset="100%" stopColor="#60a5fa" />
                    </linearGradient>
                    <linearGradient id="tGrad2" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#f97316" /><stop offset="100%" stopColor="#ef4444" />
                    </linearGradient>
                  </defs>
                </BarChart>
              </ResponsiveContainer>
            )}
          </ChartBox>

          {/* Table */}
          <div style={{ marginTop: 20, overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                  {["Matrix", "IPC (GB/s)", "GPU-MPI (GB/s)", "Staged (GB/s)", "IPC (ms)", "vs GPU-MPI", "vs Staged"].map((h) => (
                    <th key={h} style={{ padding: "8px 10px", textAlign: "right", color: "#6b7094", fontWeight: 500 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tData.map((d, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                    <td style={{ padding: "7px 10px", textAlign: "right", color: "#e8e8ed", fontWeight: 600 }}>{d.grid}</td>
                    <td style={{ padding: "7px 10px", textAlign: "right", color: "#34d399" }}>{d.ipc_gb}</td>
                    <td style={{ padding: "7px 10px", textAlign: "right", color: "#60a5fa" }}>{d.mpigpu_gb}</td>
                    <td style={{ padding: "7px 10px", textAlign: "right", color: "#f97316" }}>{d.staged_gb}</td>
                    <td style={{ padding: "7px 10px", textAlign: "right", color: "#e8e8ed" }}>{d.ipc_t}</td>
                    <td style={{ padding: "7px 10px", textAlign: "right", color: "#34d399", fontWeight: 600 }}>{d.ipc_vs_mpigpu > 0 ? "+" : ""}{d.ipc_vs_mpigpu}%</td>
                    <td style={{ padding: "7px 10px", textAlign: "right", color: "#a78bfa", fontWeight: 600 }}>+{d.ipc_vs_staged}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>)}
      </div>
    </div>
  );
}
