import { useEffect, useMemo, useState } from 'react';
import { 
  BrainCircuit,
  Activity,
  ShieldCheck,
  Gauge,
  Bell,
  User,
  Command,
  Zap,
  Database,
  Cpu,
  Trophy,
  History,
  Terminal as TerminalIcon
} from 'lucide-react';
import { 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { useAgentData } from './hooks/useAgentData';

// ============================================================
// MonkeyType-inspired palette
// bg:       #323437  (dark charcoal)
// sub-bg:   #2c2e31  (slightly lighter)
// card-bg:  #1e2023  (darkest cards)
// accent:   #e2b714  (signature yellow)
// text:     #d1d0c5  (off-white)
// sub-text: #646669  (muted gray)
// graph:    #f7953b  (orange)
// ============================================================

const C = {
  bg:      '#323437',
  subBg:   '#2c2e31',
  cardBg:  '#1e2023',
  border:  '#3a3c40',
  accent:  '#e2b714',
  orange:  '#f7953b',
  text:    '#d1d0c5',
  sub:     '#646669',
  green:   '#44cf6e',
  red:     '#ca4754',
};

function App() {
  const { agentState, logs, connected, memoryHistory } = useAgentData();
  const [renderedMessage, setRenderedMessage] = useState('');

  const TYPING_SPEED_MS = 18;

  useEffect(() => {
    const text = agentState?.new_message ?? '';
    if (!text) { setRenderedMessage(''); return; }
    let i = 0;
    setRenderedMessage('');
    const iv = window.setInterval(() => {
      i++;
      setRenderedMessage(text.slice(0, i));
      if (i >= text.length) window.clearInterval(iv);
    }, TYPING_SPEED_MS);
    return () => window.clearInterval(iv);
  }, [agentState?.new_message]);

  const chartData = useMemo(() => logs, [logs]);

  const precision = useMemo(() => {
    if (!agentState?.step || !agentState.memory_count) return 0;
    return (agentState.correct_in_memory / agentState.memory_count) * 100;
  }, [agentState]);

  const recall = useMemo(() => {
    if (!agentState?.step || !agentState.total_relevant_seen) return 0;
    return (agentState.correct_in_memory / agentState.total_relevant_seen) * 100;
  }, [agentState]);

  return (
    <div style={{ background: C.bg, color: C.text, fontFamily: "'Roboto Mono', monospace" }}
      className="flex flex-col h-screen overflow-y-auto w-full">

      {/* ── NAV ── */}
      <nav style={{ background: C.subBg, borderBottom: `1px solid ${C.border}` }}
        className="h-16 px-6 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <div style={{ background: C.accent }} className="p-2 rounded-lg text-black">
            <BrainCircuit size={22} />
          </div>
          <div>
            <h1 className="text-lg font-black tracking-tighter leading-none italic uppercase" style={{ color: C.text }}>
              Long Horizon <span style={{ color: C.accent }}>Memory</span>
            </h1>
            <p className="text-[9px] font-bold uppercase tracking-[0.2em] mt-0.5" style={{ color: C.sub }}>
              Live AI Production Monitor v2.0
            </p>
          </div>
          <div className="hidden md:flex items-center gap-6 border-l ml-4 pl-6" style={{ borderColor: C.border }}>
            <div className="flex flex-col">
              <span className="text-[9px] font-bold uppercase tracking-widest" style={{ color: C.sub }}>Environment</span>
              <span className="text-xs font-bold" style={{ color: C.accent }}>{agentState?.task_name?.toUpperCase() || 'IDLE'}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[9px] font-bold uppercase tracking-widest" style={{ color: C.sub }}>Capacity</span>
              <span className="text-xs font-bold" style={{ color: C.text }}>{agentState?.memory_capacity || 8} Slots</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-[10px] font-black tracking-widest"
            style={{ background: C.cardBg, border: `1px solid ${C.border}` }}>
            <div className={`w-2 h-2 rounded-full`}
              style={{ background: connected ? C.green : C.red, boxShadow: connected ? `0 0 8px ${C.green}` : undefined }} />
            <span style={{ color: C.sub }}>{connected ? 'REAL-TIME SYNC' : 'DISCONNECTED'}</span>
          </div>
          <Bell size={18} style={{ color: C.sub }} className="cursor-pointer hover:opacity-80" />
          <User size={18} style={{ color: C.sub }} className="cursor-pointer hover:opacity-80" />
        </div>
      </nav>

      {/* ── MAIN ── */}
      <main className="flex-1 p-6 flex flex-col gap-6 max-w-[1600px] mx-auto w-full">

        {/* Metric Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard label="F1 PERFORMANCE" value={`${((agentState?.task_score || 0) * 100).toFixed(0)}%`}
            sub="Accuracy Metric" icon={<Trophy size={16}/>} color={C.accent} />
          <MetricCard label="PRECISION" value={`${precision.toFixed(0)}%`}
            sub="Reliability Rate" icon={<ShieldCheck size={16}/>} color={C.green} />
          <MetricCard label="RECALL" value={`${recall.toFixed(0)}%`}
            sub="Retention Rate" icon={<Activity size={16}/>} color="#3b82f6" />
          <MetricCard label="STEP REWARD" value={agentState?.reward?.toFixed(2) || '0.00'}
            sub="Current Feedback" icon={<Zap size={16}/>} color={C.orange} />
        </div>

        {/* Chart + Side Panels */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* Left: Chart + Input */}
          <div className="lg:col-span-8 flex flex-col gap-4">

            {/* Graph */}
            <div style={{ background: C.subBg, border: `1px solid ${C.border}` }}
              className="rounded-2xl p-6 relative flex flex-col shadow-xl">
              {/* top accent bar */}
              <div className="absolute top-0 left-0 w-full h-[3px] rounded-t-2xl"
                style={{ background: `linear-gradient(90deg, transparent, ${C.orange}, transparent)`, opacity: 0.7 }} />

              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2">
                  <Gauge size={18} style={{ color: C.orange }} />
                  <span className="text-[11px] font-black uppercase tracking-[0.3em]" style={{ color: C.sub }}>Reward Trajectory</span>
                </div>
                <div className="flex gap-4 text-[10px] font-bold" style={{ color: C.sub }}>
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full" style={{ background: C.orange }} /> STEP REWARD
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full" style={{ background: C.sub }} /> AVG SCORE
                  </div>
                </div>
              </div>

              <div style={{ height: 280, width: '100%' }}>
                <ResponsiveContainer width="99%" height={280} minWidth={100} minHeight={200}>
                  <AreaChart data={chartData} margin={{ top: 8, right: 8, left: -24, bottom: 0 }}>
                    <defs>
                      <linearGradient id="orangeGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={C.orange} stopOpacity={0.35} />
                        <stop offset="95%" stopColor={C.orange} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                    <XAxis dataKey="step" hide />
                    <YAxis hide domain={['auto', 'auto']} />
                    <Tooltip
                      contentStyle={{ background: C.cardBg, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 11 }}
                      itemStyle={{ color: C.orange }}
                    />
                    <Area type="monotone" dataKey="reward" stroke={C.orange} strokeWidth={3}
                      fillOpacity={1} fill="url(#orangeGrad)" animationDuration={800} />
                    <Area type="monotone" dataKey="fmt_reward" stroke={C.sub} strokeWidth={1}
                      strokeDasharray="4 4" fill="none" dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Input + Action row */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
              <div style={{ background: C.subBg, border: `1px solid ${C.border}` }}
                className="md:col-span-8 rounded-2xl p-6 flex flex-col gap-3 shadow-lg">
                <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-widest" style={{ color: C.sub }}>
                  <Cpu size={13} /> Input Perception Stream
                </div>
                <div className="text-base leading-relaxed font-medium min-h-[80px]" style={{ color: '#94a3b8' }}>
                  {renderedMessage
                    ? <span style={{ color: C.text }}>{renderedMessage}<span className="inline-block w-2 h-4 ml-1 animate-pulse" style={{ background: C.accent }} /></span>
                    : <span className="italic opacity-30">System idling. Waiting for observation...</span>
                  }
                </div>
              </div>

              <div style={{ background: C.subBg, border: `1px solid ${C.border}` }}
                className="md:col-span-4 rounded-2xl p-6 flex flex-col justify-between shadow-lg">
                <span className="text-[10px] font-black uppercase tracking-widest" style={{ color: C.sub }}>Active Action</span>
                <div className="flex flex-col gap-2 mt-3">
                  <span className="text-2xl font-black italic tracking-tighter"
                    style={{ color: agentState?.operation === 'noop' ? C.sub : C.accent }}>
                    {agentState?.operation?.toUpperCase() || 'STANDBY'}
                  </span>
                  <div className="h-1 w-full rounded-full overflow-hidden" style={{ background: C.border }}>
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: agentState?.step ? '100%' : 0 }}
                      transition={{ duration: 3 }}
                      className="h-full"
                      style={{ background: C.orange }}
                    />
                  </div>
                  <span className="text-[9px] font-bold uppercase" style={{ color: C.sub }}>
                    Step {agentState?.step || 0} · {agentState?.timestamp ? agentState.timestamp.slice(11, 19) : '--:--:--'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Right: Memory + Log */}
          <div className="lg:col-span-4 flex flex-col gap-4">

            {/* Storage Map */}
            <div style={{ background: C.subBg, border: `1px solid ${C.border}` }}
              className="rounded-2xl p-6 flex flex-col gap-5 shadow-lg">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-widest" style={{ color: C.sub }}>
                  <Database size={13} /> Storage Allocation
                </div>
                <span className="text-xs font-bold" style={{ color: C.accent }}>
                  {agentState?.memory_count || 0} / {agentState?.memory_capacity || 8}
                </span>
              </div>

              <div className="grid grid-cols-4 gap-2">
                {Array.from({ length: agentState?.memory_capacity || 8 }).map((_, idx) => (
                  <div key={idx} className="h-7 rounded"
                    style={{
                      background: idx < (agentState?.memory_count || 0) ? `${C.accent}25` : C.cardBg,
                      border: `1px solid ${idx < (agentState?.memory_count || 0) ? C.accent : C.border}`,
                      boxShadow: idx < (agentState?.memory_count || 0) ? `0 0 8px ${C.accent}44` : undefined,
                    }}
                  />
                ))}
              </div>

              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-widest" style={{ color: C.sub }}>
                  <TerminalIcon size={13} /> Buffer
                </div>
                <div className="rounded-lg p-3 text-[10px] leading-relaxed max-h-[130px] overflow-y-auto"
                  style={{ background: C.cardBg, border: `1px solid ${C.border}`, color: C.sub }}>
                  {agentState?.memory
                    ? <span style={{ color: C.text }}>{agentState.memory}</span>
                    : <span className="italic opacity-30">Buffer empty.</span>
                  }
                </div>
              </div>
            </div>

            {/* Event Log */}
            <div style={{ background: C.subBg, border: `1px solid ${C.border}` }}
              className="rounded-2xl p-6 flex flex-col gap-3 shadow-lg flex-1 max-h-[380px]">
              <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-widest" style={{ color: C.sub }}>
                <History size={13} /> Protocol History
              </div>
              <div className="flex flex-col gap-2 overflow-y-auto pr-1">
                <AnimatePresence mode="popLayout">
                  {logs.slice().reverse().map((log) => (
                    <motion.div
                      key={`${log.timestamp}-${log.step}`}
                      initial={{ opacity: 0, x: 16 }}
                      animate={{ opacity: 1, x: 0 }}
                      style={{ background: C.cardBg, border: `1px solid ${C.border}` }}
                      className="p-3 rounded-xl flex items-center justify-between"
                    >
                      <div className="flex flex-col gap-0.5">
                        <span className="text-[10px] font-black italic uppercase" style={{ color: C.accent }}>{log.operation}</span>
                        <span className="text-[9px] font-bold" style={{ color: C.sub }}>STEP {log.step} · {log.timestamp.slice(11, 19)}</span>
                      </div>
                      <span className="text-xs font-black" style={{ color: log.reward >= 0 ? C.green : C.red }}>
                        {log.reward >= 0 ? '+' : ''}{log.reward.toFixed(2)}
                      </span>
                    </motion.div>
                  ))}
                </AnimatePresence>
                {logs.length === 0 && (
                  <div className="text-[10px] italic text-center py-10 uppercase tracking-widest" style={{ color: C.sub }}>
                    Awaiting protocol execution...
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* ── FOOTER ── */}
      <footer style={{ background: C.subBg, borderTop: `1px solid ${C.border}` }}
        className="h-10 px-6 flex items-center justify-between opacity-60">
        <span className="text-[9px] font-bold uppercase tracking-widest" style={{ color: C.sub }}>
          Long Horizon Memory Env · Live Monitor
        </span>
        <div className="flex items-center gap-2" style={{ color: C.sub }}>
          <Command size={10} />
          <span className="text-[9px] font-bold uppercase tracking-widest">Meta Research · OpenEnv</span>
        </div>
      </footer>
    </div>
  );
}

function MetricCard({ label, value, sub, icon, color }: any) {
  return (
    <div style={{ background: '#2c2e31', border: '1px solid #3a3c40' }}
      className="p-5 rounded-2xl shadow-lg flex items-start justify-between relative overflow-hidden group transition-all duration-300"
      onMouseEnter={e => (e.currentTarget.style.borderColor = color + '66')}
      onMouseLeave={e => (e.currentTarget.style.borderColor = '#3a3c40')}
    >
      <div className="flex flex-col gap-1 z-10">
        <span className="text-[9px] font-black uppercase tracking-[0.2em]" style={{ color: '#646669' }}>{label}</span>
        <span className="text-3xl font-black tracking-tighter" style={{ color }}>{value}</span>
        <span className="text-[9px] font-bold uppercase tracking-widest mt-0.5" style={{ color: '#646669' }}>{sub}</span>
      </div>
      <div className="p-2 rounded-xl z-10 transition-colors" style={{ background: '#1e2023', color: '#646669' }}>
        {icon}
      </div>
    </div>
  );
}

export default App;
