import React, { useState, useEffect } from 'react';
import {
  SlidersHorizontal,
  RotateCcw,
  Play,
  TrendingUp,
  Boxes,
  Clock,
  AlertTriangle,
  CheckCircle2,
  History,
  Trash2,
  ArrowRight,
  ShieldAlert,
  Layers,
  Info,
  Sparkles,
  Filter,
  ArrowUpRight,
  ArrowDownRight,
  ChevronRight,
  BarChart2,
  Zap,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  LineChart,
  Line,
} from 'recharts';
import { useFilters } from '../context/FilterContext';
import { simulateScenario } from '../services/api';
import { DataTable } from '../components/DataTable';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';

const SCENARIO_PRESETS = [
  {
    name: 'Demand Surge (+20%)',
    description: 'Simulates a 20% promotional demand spike across forecast weeks',
    demand: 20,
    stock: 0,
    leadTime: 0,
    icon: TrendingUp,
  },
  {
    name: 'Supply Chain Bottleneck (+14d)',
    description: 'Simulates two weeks of supplier transit delay',
    demand: 0,
    stock: 0,
    leadTime: 14,
    icon: Clock,
  },
  {
    name: 'Inventory Deficit (-50%)',
    description: 'Simulates sudden inventory shrinkage or unrecorded sales',
    demand: 0,
    stock: -50,
    leadTime: 0,
    icon: Boxes,
  },
  {
    name: 'Combined Supply Stress Test',
    description: '+25% demand jump with -30% stock deficit and +7d supplier delay',
    demand: 25,
    stock: -30,
    leadTime: 7,
    icon: AlertTriangle,
  },
  {
    name: 'Demand Contraction (-15%)',
    description: 'Simulates a post-holiday demand drop of 15%',
    demand: -15,
    stock: 0,
    leadTime: 0,
    icon: Zap,
  },
];

export const ScenarioSimulator = () => {
  const { filterOptions, selectedStore, selectedFamily, selectedModel } = useFilters();

  // Scenario Controls
  const [demandChange, setDemandChange] = useState(0);
  const [stockChange, setStockChange] = useState(0);
  const [leadTimeAdjustment, setLeadTimeAdjustment] = useState(0);
  const [storeFilter, setStoreFilter] = useState(selectedStore || '');
  const [familyFilter, setFamilyFilter] = useState(selectedFamily || '');
  const [modelFilter, setModelFilter] = useState(selectedModel || 'naive_forecast');

  // Simulation State
  const [simulationData, setSimulationData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Session History State (stored in memory for this user session)
  const [sessionHistory, setSessionHistory] = useState([]);

  // Sync with global filters on first load
  useEffect(() => {
    if (selectedStore) setStoreFilter(selectedStore);
    if (selectedFamily) setFamilyFilter(selectedFamily);
    if (selectedModel) setModelFilter(selectedModel);
  }, [selectedStore, selectedFamily, selectedModel]);

  // Execute simulation
  const handleSimulate = async (customParams = null) => {
    const params = customParams || {
      demand_change_pct: Number(demandChange),
      stock_change_pct: Number(stockChange),
      lead_time_adjustment_days: Number(leadTimeAdjustment),
      store_nbr: storeFilter ? Number(storeFilter) : null,
      family: familyFilter || null,
      model: modelFilter || 'naive_forecast',
    };

    setLoading(true);
    setError(null);

    try {
      const res = await simulateScenario(params);
      setSimulationData(res);

      // Append to session history (capped at last 10 entries)
      const historyItem = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        params,
        metrics: res.metrics,
        summary: res.summary_highlight,
      };

      setSessionHistory((prev) => [historyItem, ...prev.slice(0, 9)]);
    } catch (err) {
      setError(err.message || 'Failed to execute scenario simulation.');
    } finally {
      setLoading(false);
    }
  };

  // Initial simulation run on mount (baseline 0%, 0%, 0d)
  useEffect(() => {
    handleSimulate({
      demand_change_pct: 0,
      stock_change_pct: 0,
      lead_time_adjustment_days: 0,
      store_nbr: storeFilter ? Number(storeFilter) : null,
      family: familyFilter || null,
      model: modelFilter || 'naive_forecast',
    });
  }, []);

  // Reset controls to zero defaults
  const handleReset = () => {
    setDemandChange(0);
    setStockChange(0);
    setLeadTimeAdjustment(0);
    setStoreFilter('');
    setFamilyFilter('');
    setModelFilter('naive_forecast');
    setError(null);

    handleSimulate({
      demand_change_pct: 0,
      stock_change_pct: 0,
      lead_time_adjustment_days: 0,
      store_nbr: null,
      family: null,
      model: 'naive_forecast',
    });
  };

  // Apply a preset
  const applyPreset = (preset) => {
    setDemandChange(preset.demand);
    setStockChange(preset.stock);
    setLeadTimeAdjustment(preset.leadTime);

    handleSimulate({
      demand_change_pct: preset.demand,
      stock_change_pct: preset.stock,
      lead_time_adjustment_days: preset.leadTime,
      store_nbr: storeFilter ? Number(storeFilter) : null,
      family: familyFilter || null,
      model: modelFilter || 'naive_forecast',
    });
  };

  // Restore past scenario from history
  const restorePastScenario = (item) => {
    setDemandChange(item.params.demand_change_pct);
    setStockChange(item.params.stock_change_pct);
    setLeadTimeAdjustment(item.params.lead_time_adjustment_days);
    if (item.params.store_nbr) setStoreFilter(String(item.params.store_nbr));
    else setStoreFilter('');
    if (item.params.family) setFamilyFilter(item.params.family);
    else setFamilyFilter('');
    if (item.params.model) setModelFilter(item.params.model);

    handleSimulate(item.params);
  };

  const metrics = simulationData?.metrics;

  // Prepare Chart Data
  const orderComparisonData = metrics
    ? [
        {
          name: 'Suggested Order',
          Original: metrics.original_suggested_order_qty,
          Simulated: metrics.simulated_suggested_order_qty,
        },
        {
          name: 'Available Stock',
          Original: Math.round(metrics.original_available_stock),
          Simulated: Math.round(metrics.simulated_available_stock),
        },
      ]
    : [];

  const riskComparisonData = metrics
    ? [
        {
          risk: 'HIGH Risk',
          Original: metrics.original_risk_counts?.HIGH || 0,
          Simulated: metrics.simulated_risk_counts?.HIGH || 0,
        },
        {
          risk: 'MEDIUM Risk',
          Original: metrics.original_risk_counts?.MEDIUM || 0,
          Simulated: metrics.simulated_risk_counts?.MEDIUM || 0,
        },
        {
          risk: 'LOW Risk',
          Original: metrics.original_risk_counts?.LOW || 0,
          Simulated: metrics.simulated_risk_counts?.LOW || 0,
        },
      ]
    : [];

  const tableColumns = [
    { key: 'store_nbr', label: 'Store #' },
    { key: 'family', label: 'Product Family' },
    {
      key: 'orig_demand',
      label: 'Orig Demand',
      render: (val) => Number(val).toLocaleString(undefined, { maximumFractionDigits: 0 }),
    },
    {
      key: 'sim_demand',
      label: 'Sim Demand',
      render: (val) => Number(val).toLocaleString(undefined, { maximumFractionDigits: 0 }),
    },
    {
      key: 'orig_stock',
      label: 'Orig Stock',
      render: (val) => Number(val).toLocaleString(undefined, { maximumFractionDigits: 0 }),
    },
    {
      key: 'sim_stock',
      label: 'Sim Stock',
      render: (val) => Number(val).toLocaleString(undefined, { maximumFractionDigits: 0 }),
    },
    { key: 'adjusted_lead_time', label: 'Lead Time (d)' },
    {
      key: 'orig_suggested_order',
      label: 'Orig Order',
      render: (val) => Number(val).toLocaleString(),
    },
    {
      key: 'sim_suggested_order',
      label: 'Sim Order',
      render: (val) => (
        <span className="font-semibold text-brand-700">
          {Number(val).toLocaleString()}
        </span>
      ),
    },
    {
      key: 'order_diff',
      label: 'Order Delta',
      render: (val) => {
        const num = Number(val);
        if (num > 0) {
          return <span className="font-semibold text-rose-600">+{num.toLocaleString()}</span>;
        } else if (num < 0) {
          return <span className="font-semibold text-emerald-600">{num.toLocaleString()}</span>;
        }
        return <span className="text-slate-400">0</span>;
      },
    },
    {
      key: 'sim_risk',
      label: 'Sim Risk',
      render: (val, row) => {
        const isEscalated = row.risk_escalated;
        let badgeColor = 'bg-emerald-50 text-emerald-700 border-emerald-200';
        if (val === 'HIGH') badgeColor = 'bg-rose-50 text-rose-700 border-rose-200';
        if (val === 'MEDIUM') badgeColor = 'bg-amber-50 text-amber-700 border-amber-200';

        return (
          <div className="flex items-center gap-1.5">
            <span className={`px-2 py-0.5 rounded text-[11px] font-bold border ${badgeColor}`}>
              {val}
            </span>
            {isEscalated && (
              <span className="text-[10px] bg-rose-100 text-rose-800 px-1.5 py-0.2 rounded font-semibold animate-pulse">
                Escalated
              </span>
            )}
          </div>
        );
      },
    },
  ];

  return (
    <div className="space-y-6">
      {/* Page Header Card */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-600 text-white flex items-center justify-center shadow-sm shadow-brand-500/20">
            <SlidersHorizontal className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
              What-If Scenario Simulator
              <span className="inline-flex items-center px-2 py-0.5 text-xs font-semibold rounded-full bg-brand-50 text-brand-700 border border-brand-200/60">
                Phase 13
              </span>
            </h1>
            <p className="text-xs text-slate-500 mt-0.5">
              Simulate operational impacts of demand shocks, stock deficits, and supplier lead-time changes side-by-side with original plans.
            </p>
          </div>
        </div>

        {/* Global Action Buttons */}
        <div className="flex items-center gap-2.5">
          <button
            onClick={handleReset}
            disabled={loading}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors shadow-2xs disabled:opacity-50"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Controls</span>
          </button>
          <button
            onClick={() => handleSimulate()}
            disabled={loading}
            className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-brand-600 hover:bg-brand-700 rounded-lg transition-colors shadow-sm shadow-brand-500/20 disabled:opacity-50"
          >
            {loading ? <LoadingSpinner size="sm" /> : <Play className="w-3.5 h-3.5 fill-current" />}
            <span>Simulate Scenario</span>
          </button>
        </div>
      </div>

      {/* Preset Scenarios Carousel */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-2.5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-brand-600" />
            Quick Scenario Presets
          </span>
          <span className="text-[11px] text-slate-400">Click to apply predefined operational shock</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2.5">
          {SCENARIO_PRESETS.map((preset, idx) => {
            const Icon = preset.icon;
            return (
              <button
                key={idx}
                onClick={() => applyPreset(preset)}
                disabled={loading}
                className="flex items-start gap-2.5 p-3 text-left rounded-lg border border-slate-200 bg-slate-50/50 hover:bg-brand-50/50 hover:border-brand-200 transition-all group disabled:opacity-50"
              >
                <div className="w-7 h-7 rounded-md bg-white border border-slate-200 text-slate-600 flex items-center justify-center shrink-0 group-hover:text-brand-600 group-hover:border-brand-300 transition-colors mt-0.5">
                  <Icon className="w-3.5 h-3.5" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-bold text-slate-800 group-hover:text-brand-900 truncate">
                    {preset.name}
                  </div>
                  <p className="text-[10px] text-slate-500 line-clamp-2 mt-0.5">
                    {preset.description}
                  </p>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Interactive Simulator Controls Card */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-6">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-brand-600" />
            Scenario Control Parameters
          </h2>
          <span className="text-[11px] text-slate-500 font-mono">
            {metrics ? `${metrics.items_evaluated} FMCG items in scope` : 'Ready to simulate'}
          </span>
        </div>

        {/* Sliders Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* 1. Demand Change Slider */}
          <div className="space-y-2 bg-slate-50/70 p-4 rounded-xl border border-slate-200/80">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                <TrendingUp className="w-3.5 h-3.5 text-brand-600" />
                Demand Change (%)
              </label>
              <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded ${
                demandChange > 0
                  ? 'bg-rose-100 text-rose-700'
                  : demandChange < 0
                  ? 'bg-emerald-100 text-emerald-700'
                  : 'bg-slate-200 text-slate-700'
              }`}>
                {demandChange > 0 ? `+${demandChange}%` : `${demandChange}%`}
              </span>
            </div>
            <input
              type="range"
              min="-50"
              max="100"
              step="5"
              value={demandChange}
              onChange={(e) => setDemandChange(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-brand-600"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-400">
              <span>-50% (Slump)</span>
              <span>0% (Plan)</span>
              <span>+100% (Surge)</span>
            </div>
            <p className="text-[11px] text-slate-500 leading-snug">
              Scales 13-week forward forecast demand uniformly across all forward horizons.
            </p>
          </div>

          {/* 2. Current Stock Change Slider */}
          <div className="space-y-2 bg-slate-50/70 p-4 rounded-xl border border-slate-200/80">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                <Boxes className="w-3.5 h-3.5 text-brand-600" />
                Current Stock Change (%)
              </label>
              <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded ${
                stockChange < 0
                  ? 'bg-rose-100 text-rose-700'
                  : stockChange > 0
                  ? 'bg-emerald-100 text-emerald-700'
                  : 'bg-slate-200 text-slate-700'
              }`}>
                {stockChange > 0 ? `+${stockChange}%` : `${stockChange}%`}
              </span>
            </div>
            <input
              type="range"
              min="-75"
              max="100"
              step="5"
              value={stockChange}
              onChange={(e) => setStockChange(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-brand-600"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-400">
              <span>-75% (Deficit)</span>
              <span>0% (Stock)</span>
              <span>+100% (Surplus)</span>
            </div>
            <p className="text-[11px] text-slate-500 leading-snug">
              Scales available on-hand stock with a strict floor at 0 units (non-negative).
            </p>
          </div>

          {/* 3. Lead Time Adjustment Slider */}
          <div className="space-y-2 bg-slate-50/70 p-4 rounded-xl border border-slate-200/80">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-brand-600" />
                Lead-Time Adjustment (Days)
              </label>
              <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded ${
                leadTimeAdjustment > 0
                  ? 'bg-rose-100 text-rose-700'
                  : leadTimeAdjustment < 0
                  ? 'bg-emerald-100 text-emerald-700'
                  : 'bg-slate-200 text-slate-700'
              }`}>
                {leadTimeAdjustment > 0 ? `+${leadTimeAdjustment}d` : `${leadTimeAdjustment}d`}
              </span>
            </div>
            <input
              type="range"
              min="-7"
              max="30"
              step="1"
              value={leadTimeAdjustment}
              onChange={(e) => setLeadTimeAdjustment(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-brand-600"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-400">
              <span>-7d (Faster)</span>
              <span>0d (Contract)</span>
              <span>+30d (Delayed)</span>
            </div>
            <p className="text-[11px] text-slate-500 leading-snug">
              Adds or subtracts days from supplier lead time. Minimum lead time is 0 days.
            </p>
          </div>
        </div>

        {/* Filter Controls Bar */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2 border-t border-slate-100">
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Store Scope
            </label>
            <select
              value={storeFilter}
              onChange={(e) => setStoreFilter(e.target.value)}
              className="w-full text-xs bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:bg-white focus:ring-1 focus:ring-brand-500"
            >
              <option value="">All Stores (Network Total)</option>
              {filterOptions?.stores?.map((s) => (
                <option key={s.store_nbr} value={s.store_nbr}>
                  Store {s.store_nbr} ({s.city})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Product Family Scope
            </label>
            <select
              value={familyFilter}
              onChange={(e) => setFamilyFilter(e.target.value)}
              className="w-full text-xs bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:bg-white focus:ring-1 focus:ring-brand-500"
            >
              <option value="">All Product Families (33 Categories)</option>
              {filterOptions?.families?.map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Forecast Model Baseline
            </label>
            <select
              value={modelFilter}
              onChange={(e) => setModelFilter(e.target.value)}
              className="w-full text-xs bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:bg-white focus:ring-1 focus:ring-brand-500"
            >
              <option value="naive_forecast">Naive Baseline (11.61% WAPE - Rank 1)</option>
              <option value="lightgbm_forecast">LightGBM Recursive (12.27% WAPE - Rank 2)</option>
              <option value="seasonal_naive_forecast">Seasonal Naive Lag-52 (16.17% WAPE)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && <ErrorMessage message={error} />}

      {/* Summary Highlight Callout Card */}
      {simulationData && (
        <div className="bg-gradient-to-r from-brand-50 via-indigo-50/50 to-white rounded-xl border border-brand-200/80 p-5 shadow-sm space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-brand-800">
              <Zap className="w-4 h-4 text-brand-600" />
              Scenario Operational Impact Summary
            </div>
            <span className="text-[11px] font-mono text-slate-500">
              Scope: {storeFilter ? `Store ${storeFilter}` : 'Network-wide'} | {familyFilter || 'All Families'}
            </span>
          </div>
          <p className="text-sm font-medium text-slate-900 leading-relaxed">
            {simulationData.summary_highlight}
          </p>
        </div>
      )}

      {/* Side-by-Side Results KPI Cards */}
      {metrics && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {/* Card 1: 13-Week Forecast Demand */}
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-3">
            <div className="flex items-center justify-between text-xs text-slate-500 font-semibold uppercase tracking-wider">
              <span>Forecast Demand</span>
              <TrendingUp className="w-4 h-4 text-brand-600" />
            </div>
            <div className="space-y-1">
              <div className="flex justify-between items-baseline">
                <span className="text-xs text-slate-400">Original:</span>
                <span className="text-xs font-mono font-medium text-slate-600">
                  {metrics.original_forecast_demand.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between items-baseline">
                <span className="text-xs text-slate-900 font-bold">Simulated:</span>
                <span className="text-lg font-bold font-mono text-brand-700">
                  {metrics.simulated_forecast_demand.toLocaleString()}
                </span>
              </div>
            </div>
            <div className={`text-xs font-semibold flex items-center justify-between pt-2 border-t border-slate-100 ${
              metrics.demand_diff_units > 0
                ? 'text-rose-600'
                : metrics.demand_diff_units < 0
                ? 'text-emerald-600'
                : 'text-slate-400'
            }`}>
              <span>Delta:</span>
              <span>{metrics.demand_diff_units > 0 ? `+${metrics.demand_diff_units.toLocaleString()}` : metrics.demand_diff_units.toLocaleString()} ({metrics.demand_diff_pct > 0 ? `+${metrics.demand_diff_pct}%` : `${metrics.demand_diff_pct}%`})</span>
            </div>
          </div>

          {/* Card 2: Available Stock */}
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-3">
            <div className="flex items-center justify-between text-xs text-slate-500 font-semibold uppercase tracking-wider">
              <span>Available Stock</span>
              <Boxes className="w-4 h-4 text-brand-600" />
            </div>
            <div className="space-y-1">
              <div className="flex justify-between items-baseline">
                <span className="text-xs text-slate-400">Original:</span>
                <span className="text-xs font-mono font-medium text-slate-600">
                  {metrics.original_available_stock.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between items-baseline">
                <span className="text-xs text-slate-900 font-bold">Simulated:</span>
                <span className="text-lg font-bold font-mono text-brand-700">
                  {metrics.simulated_available_stock.toLocaleString()}
                </span>
              </div>
            </div>
            <div className={`text-xs font-semibold flex items-center justify-between pt-2 border-t border-slate-100 ${
              metrics.stock_diff_units < 0
                ? 'text-rose-600'
                : metrics.stock_diff_units > 0
                ? 'text-emerald-600'
                : 'text-slate-400'
            }`}>
              <span>Delta:</span>
              <span>{metrics.stock_diff_units > 0 ? `+${metrics.stock_diff_units.toLocaleString()}` : metrics.stock_diff_units.toLocaleString()} ({metrics.stock_diff_pct > 0 ? `+${metrics.stock_diff_pct}%` : `${metrics.stock_diff_pct}%`})</span>
            </div>
          </div>

          {/* Card 3: Supplier Lead Time */}
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-3">
            <div className="flex items-center justify-between text-xs text-slate-500 font-semibold uppercase tracking-wider">
              <span>Avg Lead Time</span>
              <Clock className="w-4 h-4 text-brand-600" />
            </div>
            <div className="space-y-1">
              <div className="flex justify-between items-baseline">
                <span className="text-xs text-slate-400">Original:</span>
                <span className="text-xs font-mono font-medium text-slate-600">
                  {metrics.original_lead_time} days
                </span>
              </div>
              <div className="flex justify-between items-baseline">
                <span className="text-xs text-slate-900 font-bold">Adjusted:</span>
                <span className="text-lg font-bold font-mono text-brand-700">
                  {metrics.adjusted_lead_time} days
                </span>
              </div>
            </div>
            <div className={`text-xs font-semibold flex items-center justify-between pt-2 border-t border-slate-100 ${
              metrics.lead_time_adjustment_days > 0
                ? 'text-rose-600'
                : metrics.lead_time_adjustment_days < 0
                ? 'text-emerald-600'
                : 'text-slate-400'
            }`}>
              <span>Adjustment:</span>
              <span>{metrics.lead_time_adjustment_days > 0 ? `+${metrics.lead_time_adjustment_days} days` : `${metrics.lead_time_adjustment_days} days`}</span>
            </div>
          </div>

          {/* Card 4: Suggested Replenishment Order */}
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-3">
            <div className="flex items-center justify-between text-xs text-slate-500 font-semibold uppercase tracking-wider">
              <span>Suggested Orders</span>
              <AlertTriangle className="w-4 h-4 text-brand-600" />
            </div>
            <div className="space-y-1">
              <div className="flex justify-between items-baseline">
                <span className="text-xs text-slate-400">Original:</span>
                <span className="text-xs font-mono font-medium text-slate-600">
                  {metrics.original_suggested_order_qty.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between items-baseline">
                <span className="text-xs text-slate-900 font-bold">Simulated:</span>
                <span className="text-lg font-bold font-mono text-brand-700">
                  {metrics.simulated_suggested_order_qty.toLocaleString()}
                </span>
              </div>
            </div>
            <div className={`text-xs font-semibold flex items-center justify-between pt-2 border-t border-slate-100 ${
              metrics.order_qty_diff_units > 0
                ? 'text-rose-600'
                : metrics.order_qty_diff_units < 0
                ? 'text-emerald-600'
                : 'text-slate-400'
            }`}>
              <span>Order Delta:</span>
              <span>{metrics.order_qty_diff_units > 0 ? `+${metrics.order_qty_diff_units.toLocaleString()}` : metrics.order_qty_diff_units.toLocaleString()} ({metrics.order_qty_diff_pct > 0 ? `+${metrics.order_qty_diff_pct}%` : `${metrics.order_qty_diff_pct}%`})</span>
            </div>
          </div>

          {/* Card 5: High Stockout Risk Items */}
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-3">
            <div className="flex items-center justify-between text-xs text-slate-500 font-semibold uppercase tracking-wider">
              <span>High-Risk Items</span>
              <ShieldAlert className="w-4 h-4 text-brand-600" />
            </div>
            <div className="space-y-1">
              <div className="flex justify-between items-baseline">
                <span className="text-xs text-slate-400">Original:</span>
                <span className="text-xs font-mono font-medium text-slate-600">
                  {metrics.original_risk_counts?.HIGH || 0} items
                </span>
              </div>
              <div className="flex justify-between items-baseline">
                <span className="text-xs text-slate-900 font-bold">Simulated:</span>
                <span className="text-lg font-bold font-mono text-rose-600">
                  {metrics.simulated_risk_counts?.HIGH || 0} items
                </span>
              </div>
            </div>
            <div className={`text-xs font-semibold flex items-center justify-between pt-2 border-t border-slate-100 ${
              metrics.high_risk_diff > 0
                ? 'text-rose-600'
                : metrics.high_risk_diff < 0
                ? 'text-emerald-600'
                : 'text-slate-400'
            }`}>
              <span>Risk Delta:</span>
              <span>{metrics.high_risk_diff > 0 ? `+${metrics.high_risk_diff} items` : `${metrics.high_risk_diff} items`}</span>
            </div>
          </div>
        </div>
      )}

      {/* Visualizations Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart 1: 13-Week Demand Trajectory (Original vs Simulated) */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                13-Week Demand Trajectory Comparison
              </h3>
              <p className="text-xs text-slate-500">
                Weekly forward demand under original plan vs. simulated shock
              </p>
            </div>
            <span className="text-xs font-semibold text-brand-700 bg-brand-50 px-2 py-0.5 rounded">
              {demandChange > 0 ? `+${demandChange}%` : `${demandChange}%`}
            </span>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={simulationData?.weekly_trajectory || []} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="week" tickFormatter={(w) => `Wk ${w}`} tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} tick={{ fontSize: 11 }} />
                <Tooltip
                  formatter={(val) => [Number(val).toLocaleString() + ' units', '']}
                  labelFormatter={(w) => `Forecast Horizon Week ${w}`}
                />
                <Legend />
                <Bar dataKey="original_demand" name="Original Demand" fill="#94a3b8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="simulated_demand" name="Simulated Demand" fill="#0284c7" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: Stockout Risk Distribution Comparison */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Stockout Risk Tier Migration
              </h3>
              <p className="text-xs text-slate-500">
                Number of store-family items in HIGH, MEDIUM, and LOW risk tiers
              </p>
            </div>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskComparisonData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="risk" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(val) => [val + ' items', '']} />
                <Legend />
                <Bar dataKey="Original" name="Original Plan" fill="#64748b" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Simulated" name="Simulated Plan" fill="#f43f5e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Item-Level Comparison Table */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              Top Affected Items Breakdown
            </h3>
            <p className="text-xs text-slate-500">
              Items exhibiting the highest replenishment order changes or risk escalation under this scenario
            </p>
          </div>
        </div>

        <DataTable
          columns={tableColumns}
          data={simulationData?.top_affected_items || []}
          pageSize={10}
          searchKey="family"
          searchPlaceholder="Filter items by product family..."
        />
      </div>

      {/* Session Scenario History */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <History className="w-4 h-4 text-brand-600" />
            <h3 className="text-sm font-bold text-slate-900">
              Session Simulation History
            </h3>
            <span className="text-xs text-slate-400">({sessionHistory.length} runs in current session)</span>
          </div>

          {sessionHistory.length > 0 && (
            <button
              onClick={() => setSessionHistory([])}
              className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-rose-600 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear History</span>
            </button>
          )}
        </div>

        {sessionHistory.length === 0 ? (
          <p className="text-xs text-slate-400 py-3 text-center">
            No scenario runs logged yet. Adjust the sliders and click "Simulate Scenario" to save runs here.
          </p>
        ) : (
          <div className="space-y-2">
            {sessionHistory.map((item) => (
              <div
                key={item.id}
                className="flex items-center justify-between p-3 rounded-lg border border-slate-200 bg-slate-50/50 hover:bg-slate-100 transition-colors text-xs"
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-slate-400">{item.timestamp}</span>
                    <span className="font-semibold text-slate-800">
                      Demand: {item.params.demand_change_pct > 0 ? `+${item.params.demand_change_pct}%` : `${item.params.demand_change_pct}%`},
                      Stock: {item.params.stock_change_pct > 0 ? `+${item.params.stock_change_pct}%` : `${item.params.stock_change_pct}%`},
                      Lead Time: {item.params.lead_time_adjustment_days > 0 ? `+${item.params.lead_time_adjustment_days}d` : `${item.params.lead_time_adjustment_days}d`}
                    </span>
                    {item.params.store_nbr && (
                      <span className="bg-slate-200 text-slate-700 px-1.5 py-0.5 rounded text-[10px] font-mono">
                        Store {item.params.store_nbr}
                      </span>
                    )}
                    {item.params.family && (
                      <span className="bg-slate-200 text-slate-700 px-1.5 py-0.5 rounded text-[10px] font-mono">
                        {item.params.family}
                      </span>
                    )}
                  </div>
                  <p className="text-slate-500 text-[11px] line-clamp-1">{item.summary}</p>
                </div>

                <div className="flex items-center gap-3">
                  <div className="text-right">
                    <span className="text-[10px] text-slate-400 block">Orders</span>
                    <span className="font-bold font-mono text-brand-700">
                      {item.metrics.simulated_suggested_order_qty.toLocaleString()}
                    </span>
                  </div>
                  <button
                    onClick={() => restorePastScenario(item)}
                    className="p-1.5 rounded-lg bg-white border border-slate-200 hover:border-brand-300 text-slate-600 hover:text-brand-600 transition-colors"
                    title="Reload this scenario"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Assumptions & Integrity Notice */}
      <div className="bg-slate-100/70 rounded-xl border border-slate-200 p-4 text-xs text-slate-600 space-y-1.5">
        <div className="flex items-center gap-1.5 font-bold text-slate-700">
          <Info className="w-4 h-4 text-brand-600 shrink-0" />
          <span>Simulation Integrity & Assumptions</span>
        </div>
        <ul className="list-disc list-inside space-y-0.5 text-slate-500 text-[11px]">
          {simulationData?.assumptions?.map((assump, i) => (
            <li key={i}>{assump}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};
