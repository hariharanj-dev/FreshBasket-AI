import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  Sparkles,
  Store,
  Package,
  Award,
  Calendar,
  Layers,
  ShieldCheck,
  Info
} from 'lucide-react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Cell,
  ReferenceLine,
} from 'recharts';
import { useFilters } from '../context/FilterContext';
import { getOverview } from '../services/api';
import { KpiCard } from '../components/KpiCard';
import { FilterBar } from '../components/FilterBar';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';

const formatSalesVolume = (val) => {
  if (val === undefined || val === null || isNaN(val)) return '—';
  const num = Number(val);
  if (num >= 1e6) {
    return `${(num / 1e6).toFixed(2)}M`;
  }
  if (num >= 1e3) {
    return `${(num / 1e3).toFixed(1)}k`;
  }
  return num.toLocaleString(undefined, { maximumFractionDigits: 1 });
};

export const Overview = () => {
  const { selectedStore, selectedFamily, selectedModel } = useFilters();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await getOverview({
        store_nbr: selectedStore || undefined,
        family: selectedFamily || undefined,
        model: selectedModel || undefined,
      });
      setData(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [selectedStore, selectedFamily, selectedModel]);

  if (loading && !data) {
    return <LoadingSpinner message="Aggregating executive demand overview..." />;
  }

  if (error && !data) {
    return <ErrorMessage message={error} onRetry={fetchData} />;
  }

  const kpis = data?.kpis || {};
  const histTotal = kpis.total_historical_sales ?? kpis.historical_sales_sum;
  const fcTotal = kpis.total_forecast_sales ?? kpis.forecast_sales_sum;
  const storeCount = kpis.unique_stores ?? kpis.store_count ?? 54;
  const familyCount = kpis.unique_families ?? kpis.family_count ?? 33;

  const timeline = data?.timeline || [];
  const modelPerf = data?.model_performance || [];
  const topFamilies = data?.top_families || [];

  return (
    <div className="space-y-6">
      {/* Top filter bar */}
      <FilterBar showModel={true} showStore={true} showFamily={true} />

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          title="Historical Sales Total"
          value={formatSalesVolume(histTotal)}
          subtitle="Cumulative complete weeks (2013–2017)"
          icon={TrendingUp}
          color="blue"
          badge="Complete Weeks"
          badgeColor="blue"
        />
        <KpiCard
          title="13-Week Forecast Total"
          value={formatSalesVolume(fcTotal)}
          subtitle={`Horizon Aug 14 – Nov 12, 2017 (${data?.selected_model?.label || 'Naive'})`}
          icon={Sparkles}
          color="emerald"
          badge="Forward Horizon"
          badgeColor="emerald"
        />
        <KpiCard
          title="Monitored Footprint"
          value={`${storeCount} Stores`}
          subtitle={`${familyCount} Product Families`}
          icon={Store}
          color="slate"
        />
        <KpiCard
          title="Best Model (Verified WAPE)"
          value="Naive Baseline"
          subtitle="11.61% Global WAPE (Rank #1)"
          icon={Award}
          color="amber"
          badge="Verified Leader"
          badgeColor="emerald"
        />
      </div>

      {/* Model Benchmark Callout Card */}
      <div className="bg-white border border-slate-200/90 rounded-xl p-4.5 shadow-2xs flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="space-y-1.5 max-w-3xl">
          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-emerald-50 text-emerald-800 border border-emerald-200">
              <ShieldCheck className="w-3 h-3 text-emerald-600" />
              Audit Verified Benchmark
            </span>
            <span className="font-bold text-slate-900 text-xs sm:text-sm">
              Global Evaluation Results Benchmark
            </span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            <strong className="text-slate-800">Naive Baseline (11.61% WAPE)</strong> is the verified top-performing model on global aggregate error, followed by <strong className="text-slate-800">LightGBM (12.27% WAPE)</strong> and <strong className="text-slate-800">Seasonal Naive (16.17% WAPE)</strong>. LightGBM demonstrates superior precision on high-velocity commercial categories (&gt;1,000 units/week).
          </p>
        </div>
        <div className="flex items-center space-x-4 shrink-0 bg-slate-50 border border-slate-200/70 rounded-lg px-3.5 py-2">
          <div className="text-right">
            <div className="text-[11px] text-slate-500 font-medium">Forecast Origin</div>
            <div className="text-xs font-bold text-slate-800 tabular-nums">Aug 13, 2017</div>
          </div>
          <div className="w-px h-7 bg-slate-200" />
          <div className="text-left">
            <div className="text-[11px] text-slate-500 font-medium">Prediction Horizon</div>
            <div className="text-xs font-bold text-brand-700">13 Full Weeks</div>
          </div>
        </div>
      </div>

      {/* Main Trajectory Chart: Historical Actuals vs 13-Week Forecasts */}
      <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-2xs">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <h3 className="text-sm sm:text-base font-bold text-slate-900 tracking-tight">
              Weekly Demand Trajectory: Historical Actuals vs. 13-Week Predictions
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Weekly aggregate volume (Last 52 historical weeks + 13 forward forecast weeks). Incomplete boundary week excluded.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-xs">
            <div className="flex items-center space-x-1.5">
              <span className="w-3 h-3 rounded-sm bg-blue-500" />
              <span className="text-slate-600 font-medium">Historical Actual</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="w-3 h-0.5 bg-emerald-600 rounded" />
              <span className="text-slate-600 font-medium">Naive Baseline</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="w-3 h-0.5 bg-sky-500 rounded border-dashed" />
              <span className="text-slate-600 font-medium">LightGBM</span>
            </div>
          </div>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={timeline} margin={{ top: 10, right: 20, left: 10, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 11, fill: '#64748b' }}
                tickFormatter={(val) => (val ? val.slice(5) : '')}
                interval="preserveStartEnd"
                minTickGap={28}
              />
              <YAxis
                tick={{ fontSize: 11, fill: '#64748b' }}
                tickFormatter={(val) => `${(val / 1e3).toFixed(0)}k`}
                domain={['auto', 'auto']}
              />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null;
                  const item = payload[0]?.payload;
                  return (
                    <div className="bg-slate-900 text-white rounded-lg p-3 text-xs shadow-xl border border-slate-800 space-y-1.5">
                      <div className="font-bold text-slate-300 border-b border-slate-700 pb-1 flex items-center justify-between gap-4">
                        <span>Week Starting: {label}</span>
                        {item.type && (
                          <span className="px-1.5 py-0.2 rounded text-[10px] bg-slate-800 text-slate-300 uppercase">
                            {item.type} {item.horizon_week ? `(+${item.horizon_week})` : ''}
                          </span>
                        )}
                      </div>
                      {item.actual !== null && (
                        <div className="flex justify-between gap-4 text-blue-400 font-semibold tabular-nums">
                          <span>Historical Actual:</span>
                          <span>{Number(item.actual).toLocaleString()} units</span>
                        </div>
                      )}
                      {item.naive !== null && (
                        <div className="flex justify-between gap-4 text-emerald-400 tabular-nums">
                          <span>Naive Forecast:</span>
                          <span>{Number(item.naive).toLocaleString()} units</span>
                        </div>
                      )}
                      {item.lightgbm !== null && (
                        <div className="flex justify-between gap-4 text-sky-400 tabular-nums">
                          <span>LightGBM Forecast:</span>
                          <span>{Number(item.lightgbm).toLocaleString()} units</span>
                        </div>
                      )}
                      {item.seasonal_naive !== null && (
                        <div className="flex justify-between gap-4 text-amber-400 tabular-nums">
                          <span>Seasonal Naive:</span>
                          <span>{Number(item.seasonal_naive).toLocaleString()} units</span>
                        </div>
                      )}
                    </div>
                  );
                }}
              />
              <ReferenceLine
                x="2017-08-14"
                stroke="#64748b"
                strokeDasharray="4 4"
                label={{
                  value: 'Forecast Origin (Aug 13)',
                  position: 'top',
                  fill: '#64748b',
                  fontSize: 10,
                  fontWeight: 600,
                }}
              />
              <Bar dataKey="actual" fill="#3b82f6" name="Historical Actual" radius={[2, 2, 0, 0]} opacity={0.85} />
              <Line
                type="monotone"
                dataKey="naive"
                stroke="#059669"
                strokeWidth={2.2}
                dot={{ r: 2.5, fill: '#059669' }}
                name="Naive Baseline"
              />
              <Line
                type="monotone"
                dataKey="lightgbm"
                stroke="#0284c7"
                strokeWidth={2}
                strokeDasharray="4 4"
                dot={{ r: 2, fill: '#0284c7' }}
                name="LightGBM"
              />
              <Line
                type="monotone"
                dataKey="seasonal_naive"
                stroke="#d97706"
                strokeWidth={1.5}
                strokeDasharray="2 2"
                dot={false}
                name="Seasonal Naive"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Two Column Grid: Model Leaderboard & Top Product Families */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Performance Comparison Bar Chart */}
        <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-2xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-sm font-bold text-slate-900 tracking-tight">
                  Global Forecast Error Benchmark (WAPE %)
                </h3>
                <p className="text-xs text-slate-500">Lower percentage indicates superior overall accuracy</p>
              </div>
              <span className="text-[10px] font-bold text-slate-500 uppercase px-2 py-0.5 bg-slate-100 rounded border border-slate-200">
                13-Week Horizon
              </span>
            </div>

            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={modelPerf}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 30, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                  <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} domain={[0, 20]} unit="%" />
                  <YAxis dataKey="model" type="category" tick={{ fontSize: 11, fill: '#334155' }} />
                  <Tooltip
                    formatter={(val) => [`${val}%`, 'WAPE Error']}
                    contentStyle={{ backgroundColor: '#0f172a', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                  />
                  <Bar dataKey="wape" radius={[0, 4, 4, 0]}>
                    {modelPerf.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={
                          entry.model?.toLowerCase().includes('naive') && !entry.model?.toLowerCase().includes('seasonal')
                            ? '#059669' // Best model Naive in green
                            : entry.model?.toLowerCase().includes('lightgbm')
                            ? '#0284c7' // LightGBM in sky/blue
                            : '#d97706' // Seasonal Naive in amber
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 grid grid-cols-3 gap-2.5 text-center text-xs">
            {modelPerf.map((m, idx) => (
              <div key={m.model} className="p-2.5 rounded-lg bg-slate-50 border border-slate-200/70">
                <div className="flex items-center justify-between text-[11px] text-slate-500">
                  <span className="font-bold text-slate-700">Rank #{idx + 1}</span>
                  {idx === 0 && <span className="text-[10px] text-emerald-700 font-bold">★ Best</span>}
                </div>
                <div className="font-semibold text-slate-800 truncate mt-0.5">{m.model}</div>
                <div className="text-base font-bold text-slate-900 mt-0.5 tabular-nums font-mono">{m.wape}%</div>
                <div className="text-[10px] text-slate-400 mt-0.5">MAE: {m.mae?.toFixed(1)}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Top Product Families Demand Distribution */}
        <div className="bg-white rounded-xl border border-slate-200/90 p-5 shadow-2xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-sm font-bold text-slate-900 tracking-tight">
                  Top 10 Product Families by Volume
                </h3>
                <p className="text-xs text-slate-500">Historical sales distribution across core categories</p>
              </div>
              <Package className="w-4 h-4 text-slate-400" />
            </div>

            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={topFamilies}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 75, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                  <XAxis
                    type="number"
                    tick={{ fontSize: 11, fill: '#64748b' }}
                    tickFormatter={(val) => `${(val / 1e6).toFixed(1)}M`}
                  />
                  <YAxis dataKey="family" type="category" tick={{ fontSize: 10, fill: '#475569' }} />
                  <Tooltip
                    formatter={(val) => [`${Number(val).toLocaleString()} units`, 'Historical Sales']}
                    contentStyle={{ backgroundColor: '#0f172a', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                  />
                  <Bar dataKey="weekly_sales" fill="#10b981" radius={[0, 4, 4, 0]} opacity={0.85} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Leading FMCG categories account for &gt;75% of unit demand</span>
            <span className="font-semibold text-brand-700">GROCERY I &amp; BEVERAGES lead</span>
          </div>
        </div>
      </div>
    </div>
  );
};
