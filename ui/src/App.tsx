import { useEffect, useMemo, useState } from 'react';
import { 
  BrainCircuit,
  Activity,
  ShieldCheck,
  Gauge,
  Bell,
  User,
  RotateCcw,
  Command,
  Zap,
  Database,
  Cpu,
  Trophy,
  History,
  Terminal as TerminalIcon
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
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

function App() {
  const { agentState, logs, connected, memoryHistory } = useAgentData();
  const [renderedMessage, setRenderedMessage] = useState('');
  const [lastOperation, setLastOperation] = useState<string | null>(null);

  // Demo typing speed control
  const TYPING_SPEED_MS = 15;

  useEffect(() => {
    const text = agentState?.new_message ?? '';
    if (!text) {
      setRenderedMessage('');
      return;
    }

    let i = 0;
    setRenderedMessage('');
    const interval = window.setInterval(() => {
      i += 1;
      setRenderedMessage(text.slice(0, i));
      if (i >= text.length) window.clearInterval(interval);
    }, TYPING_SPEED_MS);

    return () => window.clearInterval(interval);
  }, [agentState?.new_message]);

  const chartData = useMemo(() => logs, [logs]);

  // Metric Calculation helper
  const precision = useMemo(() => {
    if (!agentState?.step || !agentState.memory_count) return 0;
    return (agentState.correct_in_memory / agentState.memory_count) * 100;
  }, [agentState]);

  const recall = useMemo(() => {
    if (!agentState?.step || !agentState.total_relevant_seen) return 0;
    return (agentState.correct_in_memory / agentState.total_relevant_seen) * 100;
  }, [agentState]);

  return (
    <div className="min-h-screen w-full bg-[#050505] text-[#f8fafc] flex flex-col font-mono selection:bg-[#facc15] selection:text-black">
      
      {/* Top Navigation Bar */}
      <nav className="h-16 border-b border-[#1e293b] px-6 flex items-center justify-between bg-[#0a0a0a]/80 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3 group">
            <div className="p-2 bg-[#facc15] rounded-lg text-black group-hover:scale-110 transition-transform duration-300">
              <BrainCircuit size={24} />
            </div>
            <div>
              <h1 className="text-lg font-black tracking-tighter leading-none italic uppercase">
                Long Horizon <span className="text-[#facc15]">Memory</span>
              </h1>
              <p className="text-[10px] text-[#64748b] font-bold uppercase tracking-[0.2em] mt-1">Live AI Production Monitor v2.0</p>
            </div>
          </div>
          
          <div className="hidden md:flex items-center gap-6 border-l border-[#1e293b] ml-4 pl-6">
            <div className="flex flex-col">
              <span className="text-[9px] text-[#64748b] uppercase font-bold tracking-widest">Environment</span>
              <span className="text-xs font-bold text-[#facc15]">{agentState?.task_name?.toUpperCase() || 'IDLE'}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[9px] text-[#64748b] uppercase font-bold tracking-widest">Capacity</span>
              <span className="text-xs font-bold">{agentState?.memory_capacity || 8} Slots</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3 px-4 py-2 bg-[#111111] border border-[#1e293b] rounded-full shadow-inner">
            <div className={`w-2 h-2 rounded-full shadow-[0_0_10px] ${connected ? 'bg-green-500 shadow-green-500 animate-pulse' : 'bg-red-500 shadow-red-500'}`} />
            <span className="text-[10px] font-black tracking-widest text-[#94a3b8]">
              {connected ? 'REAL-TIME SYNC' : 'CONNECTION LOST'}
            </span>
          </div>
          <div className="flex gap-4 text-[#64748b]">
            <Bell size={20} className="hover:text-[#facc15] cursor-pointer transition-colors" />
            <User size={20} className="hover:text-[#facc15] cursor-pointer transition-colors" />
          </div>
        </div>
      </nav>

      <main className="flex-1 p-6 flex flex-col gap-6 max-w-[1600px] mx-auto w-full">
        
        {/* Top Metric Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard 
            label="F1 PERFORMANCE" 
            value={`${((agentState?.task_score || 0) * 100).toFixed(0)}%`} 
            subValue="Accuracy Metric"
            icon={<Trophy size={18} />}
            color="#facc15"
          />
          <MetricCard 
            label="PRECISION" 
            value={`${precision.toFixed(0)}%`} 
            subValue="Reliability Rate"
            icon={<ShieldCheck size={18} />}
            color="#10b981"
          />
          <MetricCard 
            label="RECALL" 
            value={`${recall.toFixed(0)}%`} 
            subValue="Retention Rate"
            icon={<Activity size={18} />}
            color="#3b82f6"
          />
          <MetricCard 
            label="STEP REWARD" 
            value={agentState?.reward?.toFixed(2) || '0.00'} 
            subValue="Current Feedback"
            icon={<Zap size={18} />}
            color="#f8fafc"
          />
        </div>

        {/* Chart & History Section */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Main Visualizer */}
          <div className="lg:col-span-8 flex flex-col gap-4">
            <div className="bg-[#0a0a0a] border border-[#1e293b] rounded-2xl p-6 relative overflow-hidden shadow-2xl group min-h-[400px] flex flex-col">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-[#facc15] to-transparent opacity-30" />
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2">
                  <Gauge className="text-[#facc15]" size={20} />
                  <span className="text-xs font-black uppercase tracking-[0.3em] text-[#64748b]">Reward Trajectory</span>
                </div>
                <div className="flex items-center gap-4 text-[10px] font-bold text-[#475569]">
                  <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-[#facc15]" /> STEP REWARD</div>
                  <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-[#64748b]" /> AVG SCORE</div>
                </div>
              </div>
              
              <div className="flex-1 w-full min-h-[250px] relative">
                <ResponsiveContainer width="99%" height={300} minWidth={100} minHeight={200}>
                  <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#facc15" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#facc15" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="step" hide />
                    <YAxis hide domain={['auto', 'auto']} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#111111', border: '1px solid #334155', borderRadius: '8px', fontSize: '10px' }}
                      itemStyle={{ color: '#facc15' }}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="reward" 
                      stroke="#facc15" 
                      strokeWidth={3} 
                      fillOpacity={1} 
                      fill="url(#rewardGradient)" 
                      animationDuration={1000}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="fmt_reward" 
                      stroke="#475569" 
                      strokeWidth={1} 
                      strokeDasharray="4 4"
                      dot={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Input Stream & Last Operation */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
              <div className="md:col-span-8 bg-[#0a0a0a] border border-[#1e293b] rounded-2xl p-6 flex flex-col gap-4 shadow-xl">
                <div className="flex items-center gap-2 text-[#64748b] text-[10px] font-black uppercase tracking-widest">
                  <Cpu size={14} />
                  <span>Input Perception Stream</span>
                </div>
                <div className="text-base sm:text-lg leading-relaxed text-[#94a3b8] font-medium min-h-[80px]">
                  {renderedMessage ? (
                    <span className="text-[#e2e8f0]">{renderedMessage}<span className="inline-block w-2 h-4 bg-[#facc15] ml-1 animate-pulse" /></span>
                  ) : (
                    <span className="italic opacity-40">System idling. Waiting for observation...</span>
                  )}
                </div>
              </div>
              <div className="md:col-span-4 bg-[#0a0a0a] border border-[#1e293b] rounded-2xl p-6 flex flex-col justify-between shadow-xl">
                <span className="text-[#64748b] text-[10px] font-black uppercase tracking-widest">Active Action</span>
                <div className="flex flex-col gap-1 mt-4">
                  <span className={`text-2xl font-black italic tracking-tighter ${agentState?.operation === 'noop' ? 'text-[#64748b]' : 'text-[#facc15]'}`}>
                    {agentState?.operation?.toUpperCase() || 'STANDBY'}
                  </span>
                  <div className="h-1 w-full bg-[#1e293b] rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: agentState?.step ? '100%' : 0 }}
                      transition={{ duration: 3 }}
                      className="h-full bg-[#facc15]" 
                    />
                  </div>
                  <span className="text-[9px] text-[#475569] font-bold uppercase mt-2">Next update in 3.0s</span>
                </div>
              </div>
            </div>
          </div>

          {/* Side Panels: Memory & History */}
          <div className="lg:col-span-4 flex flex-col gap-6">
            
            {/* Storage Map */}
            <div className="bg-[#0a0a0a] border border-[#1e293b] rounded-2xl p-6 flex flex-col gap-6 shadow-xl">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-[#64748b] text-[10px] font-black uppercase tracking-widest">
                  <Database size={14} />
                  <span>Storage Allocation</span>
                </div>
                <span className="text-xs font-bold text-[#facc15]">{agentState?.memory_count || 0} / {agentState?.memory_capacity || 8}</span>
              </div>
              
              <div className="grid grid-cols-4 gap-2">
                {Array.from({ length: agentState?.memory_capacity || 8 }).map((_, idx) => (
                  <div 
                    key={idx} 
                    className={`h-8 rounded border ${idx < (agentState?.memory_count || 0) 
                      ? 'bg-[#facc15]/20 border-[#facc15]/50 shadow-[0_0_10px_#facc1533]' 
                      : 'bg-[#111111] border-[#1e293b]'}`}
                  />
                ))}
              </div>

              <div className="flex flex-col gap-3 mt-2">
                <div className="flex items-center gap-2 text-[#64748b] text-[10px] font-black uppercase tracking-widest">
                  <TerminalIcon size={14} />
                  <span>Recent Buffer</span>
                </div>
                <div className="bg-black/50 rounded-lg p-3 text-[10px] leading-relaxed border border-[#1e293b] max-h-[150px] overflow-y-auto text-[#64748b] font-mono scrollbar-hide">
                  {agentState?.memory ? (
                    <div className="text-[#94a3b8]">{agentState.memory}</div>
                  ) : (
                    <div className="italic opacity-30">Buffer empty.</div>
                  )}
                </div>
              </div>
            </div>

            {/* Event Log */}
            <div className="bg-[#0a0a0a] border border-[#1e293b] rounded-2xl p-6 flex flex-col gap-4 shadow-xl flex-1 max-h-[400px]">
              <div className="flex items-center gap-2 text-[#64748b] text-[10px] font-black uppercase tracking-widest mb-2">
                <History size={14} />
                <span>Protocol History</span>
              </div>
              <div className="flex flex-col gap-2 overflow-y-auto pr-2 scrollbar-hide">
                <AnimatePresence mode="popLayout">
                  {logs.slice().reverse().map((log) => (
                    <motion.div 
                      key={log.timestamp + log.step}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="p-3 bg-[#111111] border border-[#1e293b] rounded-xl flex items-center justify-between group hover:border-[#facc15]/30 transition-colors"
                    >
                      <div className="flex flex-col gap-0.5">
                        <span className="text-[10px] font-black italic tracking-tighter text-[#facc15] uppercase">{log.operation}</span>
                        <span className="text-[9px] text-[#475569] font-bold">STEP {log.step} · {log.timestamp.slice(11, 19)}</span>
                      </div>
                      <div className="text-right">
                        <span className={`text-xs font-black ${log.reward >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                          {log.reward >= 0 ? '+' : ''}{log.reward.toFixed(2)}
                        </span>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
                {logs.length === 0 && <div className="text-[10px] text-[#475569] italic text-center py-10 uppercase tracking-widest">Awaiting protocol execution...</div>}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer Info */}
      <footer className="h-10 border-t border-[#1e293b] bg-[#0a0a0a] px-6 flex items-center justify-between opacity-50">
        <div className="flex items-center gap-4 text-[9px] font-bold text-[#64748b] uppercase tracking-widest">
          <div className="flex items-center gap-2">
            <kbd className="bg-[#1e293b] px-1.5 py-0.5 rounded text-[#94a3b8]">ALT</kbd>
            <span>+</span>
            <kbd className="bg-[#1e293b] px-1.5 py-0.5 rounded text-[#94a3b8]">S</kbd>
            <span className="ml-1">SYSTEM STATS</span>
          </div>
        </div>
        <div className="flex items-center gap-2 text-[9px] font-bold text-[#64748b] uppercase tracking-widest">
          <Command size={10} />
          <span>Meta Research - OpenEnv Division</span>
        </div>
      </footer>
    </div>
  );
}

function MetricCard({ label, value, subValue, icon, color }: any) {
  return (
    <div className="bg-[#0a0a0a] border border-[#1e293b] p-5 rounded-2xl shadow-xl flex items-start justify-between group hover:border-[#facc15]/40 transition-all duration-300 relative overflow-hidden">
      <div className="flex flex-col gap-1 relative z-10">
        <span className="text-[10px] font-black text-[#64748b] tracking-[0.2em] uppercase">{label}</span>
        <span className="text-3xl font-black tracking-tighter" style={{ color: color || '#f8fafc' }}>{value}</span>
        <span className="text-[9px] font-bold text-[#475569] uppercase tracking-widest mt-1">{subValue}</span>
      </div>
      <div className="p-2 bg-[#111111] rounded-xl text-[#64748b] group-hover:text-[#facc15] transition-colors relative z-10">
        {icon}
      </div>
      <div className="absolute bottom-0 right-0 w-24 h-24 bg-gradient-to-br from-transparent to-white/[0.02] -rotate-12 transform translate-x-12 translate-y-12" />
    </div>
  );
}

export default App;
