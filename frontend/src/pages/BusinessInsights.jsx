import React, { useState, useEffect } from 'react';
import {
  Lightbulb,
  TrendingUp,
  TrendingDown,
  Activity,
  AlertTriangle,
  ShieldAlert,
  Sparkles,
  HelpCircle,
  Package,
  Store,
  Layers,
  ArrowRight,
  Info,
  CheckCircle2,
  Calendar,
  Clock,
  Download
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  PieChart,
  Pie,
  Legend,
} from 'recharts';
import { useFilters } from '../context/FilterContext';
import {
  getInsightsOverview,
  getInsightsDemandTrends,
  getInsightsProductOpportunities,
  getInsightsInventoryRisks,
} from '../services/api';
import { KpiCard } from '../components/KpiCard';
import { FilterBar } from '../components/FilterBar';
import { DataTable } from '../components/DataTable';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';

const PRIORITY_BADGES = {
  high: { label: 'High Priority', bg: 'bg-red-100 text-red-800 border-red-300' },
  medium: { label: 'Medium Priority', bg: 'bg-amber-100 text-amber-800 border-amber-300' },
  low: { label: 'Low Priority', bg: 'bg-emerald-100 text-emerald-800 border-emerald-300' },
  info: { label: 'Model Intelligence', bg: 'bg-blue-100 text-blue-800 border-blue-300' },
};

const TREND_COLORS = {
  Rising: '#059669',
  Declining: '#ef4444',
  Stable: '#3b82f6',
  'Insufficient data': '#94a3b8',
};

export const BusinessInsights = () => {
  const { selectedStore, selectedFamily, selectedModel, dataMode } = useFilters();
  const [activeTab, setActiveTab] = useState('executive'); // 'executive', 'trends', 'forecast', 'inventory'
  const [overviewData, setOverviewData] = useState(null);
  const [trendsData, setTrendsData] = useState(null);
  const [forecastData, setForecastData] = useState(null);
  const [inventoryData, setInventoryData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchAllInsights = async () => {
    try {
      setLoading(true);
      setError(null);
      const params = {
        store_nbr: selectedStore || undefined,
        family: selectedFamily || undefined,
        model: selectedModel || undefined,
      };

      const [ovRes, trRes, fcRes, invRes] = await Promise.all([
        getInsightsOverview(params),
        getInsightsDemandTrends({ store_nbr: selectedStore || undefined, family: selectedFamily || undefined, window_weeks: 12 }),
        getInsightsProductOpportunities(params),
        getInsightsInventoryRisks(params),
      ]);

      setOverviewData(ovRes);
      setTrendsData(trRes);
      setForecastData(fcRes);
      setInventoryData(invRes);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllInsights();
  }, [selectedStore, selectedFamily, selectedModel, dataMode]);

  const insightCards = overviewData?.insight_cards || [];
  const trendSummary = overviewData?.trend_summary || { rising: 0, declining: 0, stable: 0, insufficient: 0, total: 0 };
  const allTrends = trendsData?.trends || [];
  const topFamilies = forecastData?.top_demand_families || [];
  const modelDivergence = forecastData?.model_divergence || [];
  const horizonUncertainty = forecastData?.horizon_uncertainty || [];
  const topStores = forecastData?.top_demand_stores || [];

  const trendPieData = [
    { name: 'Rising Trend', value: trendSummary.rising, color: TREND_COLORS.Rising },
    { name: 'Declining Trend', value: trendSummary.declining, color: TREND_COLORS.Declining },
    { name: 'Stable Demand', value: trendSummary.stable, color: TREND_COLORS.Stable },
    ...(trendSummary.insufficient > 0
      ? [{ name: 'Insufficient Data', value: trendSummary.insufficient, color: TREND_COLORS['Insufficient data'] }]
      : []),
  ];

  const trendTableColumns = [
    { key: 'family', header: 'Product Family', sortable: true },
    ...(selectedStore || allTrends[0]?.store_nbr !== null
      ? [{ key: 'store_nbr', header: 'Store', sortable: true, render: (val) => val ? `Store ${val}` : 'All' }]
      : []),
    {
      key: 'trend_class',
      header: 'Trend Classification',
      sortable: true,
      render: (val, row) => (
        <span
          className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold border ${
            val === 'Rising'
              ? 'bg-emerald-100 text-emerald-800 border-emerald-300'
              : val === 'Declining'
              ? 'bg-red-100 text-red-800 border-red-300'
              : val === 'Stable'
              ? 'bg-blue-100 text-blue-800 border-blue-200'
              : 'bg-slate-100 text-slate-700 border-slate-300'
          }`}
        >
          {val === 'Rising' ? '↗ Rising' : val === 'Declining' ? '↘ Declining' : val === 'Stable' ? '→ Stable' : '— Insufficient'}
        </span>
      ),
    },
    {
      key: 'pct_change',
      header: '% Change',
      sortable: true,
      align: 'right',
      render: (val) => {
        const num = Number(val);
        const color = num > 0 ? 'text-emerald-700 font-bold' : num < 0 ? 'text-red-700 font-bold' : 'text-slate-600';
        return <span className={color}>{num > 0 ? `+${num.toFixed(1)}%` : `${num.toFixed(1)}%`}</span>;
      },
    },
    {
      key: 'slope',
      header: 'OLS Slope (β)',
      sortable: true,
      align: 'right',
      render: (val) => `${Number(val) > 0 ? '+' : ''}${Number(val).toFixed(1)} u/wk`,
    },
    {
      key: 'p_value',
      header: 'p-value (t-test)',
      sortable: true,
      align: 'right',
      render: (val, row) => (
        <span className={row.is_significant ? 'text-emerald-700 font-bold' : 'text-slate-500'}>
          {Number(val).toFixed(4)} {row.is_significant ? '★' : ''}
        </span>
      ),
    },
    {
      key: 'recent_mean_weekly_sales',
      header: 'Recent 4-Wk Avg',
      sortable: true,
      align: 'right',
      render: (val) => Number(val).toLocaleString(undefined, { maximumFractionDigits: 1 }),
    },
    {
      key: 'prior_mean_weekly_sales',
      header: 'Prior 8-Wk Avg',
      sortable: true,
      align: 'right',
      render: (val) => Number(val).toLocaleString(undefined, { maximumFractionDigits: 1 }),
    },
    {
      key: 'reason',
      header: 'Statistical Context & Audit Reason',
      sortable: false,
      render: (val) => <span className="text-[11px] text-slate-500 max-w-xs block leading-tight">{val}</span>,
    },
  ];

  const divergenceColumns = [
    { key: 'family', header: 'Product Family', sortable: true },
    {
      key: 'naive_units',
      header: 'Naive Baseline (Units)',
      sortable: true,
      align: 'right',
      render: (val) => Number(val).toLocaleString(),
    },
    {
      key: 'lightgbm_units',
      header: 'LightGBM (Units)',
      sortable: true,
      align: 'right',
      render: (val) => Number(val).toLocaleString(),
    },
    {
      key: 'difference_units',
      header: 'Variance (Units)',
      sortable: true,
      align: 'right',
      render: (val) => (
        <span className={Number(val) >= 0 ? 'text-blue-700 font-semibold' : 'text-amber-700 font-semibold'}>
          {Number(val) > 0 ? `+${Number(val).toLocaleString()}` : Number(val).toLocaleString()}
        </span>
      ),
    },
    {
      key: 'difference_pct',
      header: 'Variance %',
      sortable: true,
      align: 'right',
      render: (val) => (
        <span className={`font-bold ${Math.abs(Number(val)) >= 15 ? 'text-blue-600' : 'text-slate-700'}`}>
          {Number(val) > 0 ? `+${Number(val).toFixed(1)}%` : `${Number(val).toFixed(1)}%`}
        </span>
      ),
    },
    {
      key: 'divergence_level',
      header: 'Divergence Level',
      sortable: true,
      render: (val) => (
        <span
          className={`px-2 py-0.5 rounded text-xs font-semibold ${
            val === 'Substantial' ? 'bg-blue-100 text-blue-800' : 'bg-slate-100 text-slate-700'
          }`}
        >
          {val}
        </span>
      ),
    },
  ];

  const vulnerableColumns = [
    {
      key: 'store_nbr',
      header: 'Store',
      sortable: true,
      render: (val) => <span className="font-semibold text-slate-900">Store {val}</span>,
    },
    { key: 'family', header: 'Product Family', sortable: true },
    {
      key: 'current_stock',
      header: 'Current Stock',
      sortable: true,
      align: 'right',
      render: (val) => Number(val).toLocaleString(),
    },
    {
      key: 'expected_lt_demand',
      header: 'Lead Demand',
      sortable: true,
      align: 'right',
      render: (val) => Number(val).toLocaleString(),
    },
    {
      key: 'reorder_point',
      header: 'Reorder Point (ROP)',
      sortable: true,
      align: 'right',
      render: (val) => Number(val).toLocaleString(),
    },
    {
      key: 'suggested_replenishment_qty',
      header: 'Suggested Reorder',
      sortable: true,
      align: 'right',
      render: (val) => <span className="font-bold text-red-600">+{Number(val).toLocaleString()} units</span>,
    },
    {
      key: 'supplier_lead_time_days',
      header: 'Lead Time',
      sortable: true,
      align: 'right',
      render: (val) => `${val ?? 7}d`,
    },
    {
      key: 'supplier_name',
      header: 'Supplier',
      sortable: true,
      render: (val) => <span className="text-slate-600 truncate max-w-[120px] inline-block">{val ?? 'Standard'}</span>,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Top Filter Bar */}
      <FilterBar showModel={true} showStore={true} showFamily={true} />

      {loading && !overviewData ? (
        <LoadingSpinner message="Synthesizing AI demand intelligence and business recommendations..." />
      ) : error && !overviewData ? (
        <ErrorMessage message={error} onRetry={fetchAllInsights} />
      ) : (
        <>
          {/* Top Level Summary KPIs */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <KpiCard
              title="Demand Accelerations"
              value={`${trendSummary.rising} Categories`}
              subtitle="Statistically verified upward trends"
              icon={TrendingUp}
              color="emerald"
              badge="Rising"
              badgeColor="emerald"
            />
            <KpiCard
              title="Demand Softenings"
              value={`${trendSummary.declining} Categories`}
              subtitle="Observed downward velocity"
              icon={TrendingDown}
              color="red"
              badge="Declining"
              badgeColor="red"
            />
            <KpiCard
              title="Stable Categories"
              value={`${trendSummary.stable} Categories`}
              subtitle="Within steady state bounds"
              icon={Activity}
              color="blue"
              badge="Stable"
              badgeColor="blue"
            />
            <KpiCard
              title="Critical Stockout Risks"
              value={`${overviewData?.inventory_risk_highlights?.high_risk_count ?? 0} SKUs`}
              subtitle="Stock below expected lead demand"
              icon={ShieldAlert}
              color="amber"
              badge="Urgent PO"
              badgeColor="amber"
            />
          </div>

          {/* Navigation Sub-Tabs */}
          <div className="flex flex-wrap items-center justify-between border-b border-slate-200 gap-3 pb-3">
            <div className="flex items-center space-x-1.5 sm:space-x-2">
              <button
                onClick={() => setActiveTab('executive')}
                className={`px-3 py-2 text-xs font-bold rounded-lg transition-colors flex items-center space-x-1.5 ${
                  activeTab === 'executive'
                    ? 'bg-brand-600 text-white shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <Lightbulb className="w-3.5 h-3.5" />
                <span>Executive Insights ({insightCards.length})</span>
              </button>
              <button
                onClick={() => setActiveTab('trends')}
                className={`px-3 py-2 text-xs font-bold rounded-lg transition-colors flex items-center space-x-1.5 ${
                  activeTab === 'trends'
                    ? 'bg-brand-600 text-white shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <TrendingUp className="w-3.5 h-3.5" />
                <span>Demand Trends ({trendSummary.total})</span>
              </button>
              <button
                onClick={() => setActiveTab('forecast')}
                className={`px-3 py-2 text-xs font-bold rounded-lg transition-colors flex items-center space-x-1.5 ${
                  activeTab === 'forecast'
                    ? 'bg-brand-600 text-white shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Forecast Opportunities &amp; Divergence</span>
              </button>
              <button
                onClick={() => setActiveTab('inventory')}
                className={`px-3 py-2 text-xs font-bold rounded-lg transition-colors flex items-center space-x-1.5 ${
                  activeTab === 'inventory'
                    ? 'bg-brand-600 text-white shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <ShieldAlert className="w-3.5 h-3.5" />
                <span>Inventory Risks ({overviewData?.inventory_risk_highlights?.high_risk_count ?? 0})</span>
              </button>
            </div>

            <div className="text-xs text-slate-500 font-medium hidden md:block">
              Comparison Window: <span className="font-semibold text-slate-700">{overviewData?.comparison_period}</span>
            </div>
          </div>

          {/* TAB 1: EXECUTIVE INSIGHT CARDS */}
          {activeTab === 'executive' && (
            <div className="space-y-5">
              {insightCards.map((card) => {
                const badge = PRIORITY_BADGES[card.priority] || PRIORITY_BADGES.info;
                return (
                  <div
                    key={card.id}
                    className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs space-y-4 hover:border-slate-300 transition-colors"
                  >
                    {/* Header */}
                    <div className="flex flex-wrap items-start justify-between gap-2 border-b border-slate-100 pb-3">
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-0.5 rounded-full text-[11px] font-bold border ${badge.bg}`}>
                            {badge.label}
                          </span>
                          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                            {card.category}
                          </span>
                        </div>
                        <h3 className="text-base font-bold text-slate-900 mt-1">{card.title}</h3>
                      </div>
                      <div className="text-right text-[11px] text-slate-400">
                        <span className="font-medium text-slate-600">Period:</span> {card.comparison_period}
                      </div>
                    </div>

                    {/* Conclusion Statement */}
                    <div className="text-sm font-semibold text-slate-800 leading-relaxed bg-slate-50/70 p-3 rounded-lg border border-slate-100">
                      {card.conclusion}
                    </div>

                    {/* Supporting Data Grid */}
                    <div>
                      <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2">
                        Supporting Verified Metrics
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2.5">
                        {Object.entries(card.supporting_data).map(([key, val]) => (
                          <div key={key} className="bg-slate-50 rounded-lg p-2.5 border border-slate-100">
                            <div className="text-[10px] text-slate-500 uppercase tracking-tight truncate">
                              {key.replace(/_/g, ' ')}
                            </div>
                            <div className="text-xs font-bold text-slate-900 mt-0.5 truncate">
                              {typeof val === 'boolean' ? (val ? 'Yes (p < 0.05)' : 'No (p >= 0.05)') : String(val)}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Actionable Recommendation Callout */}
                    <div className="bg-emerald-50/70 border border-emerald-200 rounded-lg p-3.5 flex items-start space-x-2.5">
                      <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0 mt-0.5" />
                      <div>
                        <span className="text-xs font-bold text-emerald-900 block">Recommended Action:</span>
                        <p className="text-xs text-emerald-800 mt-0.5 leading-relaxed">{card.recommended_action}</p>
                      </div>
                    </div>

                    {/* Methodological Limitations / Data Caveats */}
                    <div className="bg-slate-50 border border-slate-200/80 rounded-lg p-3 flex items-start space-x-2 text-[11px] text-slate-500">
                      <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
                      <div>
                        <span className="font-semibold text-slate-700">Limitations &amp; Data Assumptions: </span>
                        <span>{card.limitations}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* TAB 2: DEMAND TRENDS */}
          {activeTab === 'trends' && (
            <div className="space-y-6">
              {/* Methodological Banner */}
              <div className="bg-slate-900 text-slate-100 rounded-xl p-5 shadow-sm space-y-2">
                <div className="flex items-center space-x-2 text-brand-400">
                  <Activity className="w-4 h-4" />
                  <span className="text-xs font-bold uppercase tracking-wider">
                    Statistically Documented Trend Methodology
                  </span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed max-w-4xl">
                  {trendsData?.methodology} Only complete historical weeks are evaluated; partial weeks (such as the 2-day August 14 boundary) are strictly excluded. Categories are classified as <strong>Rising</strong> or <strong>Declining</strong> with statistical significance established via two-tailed Student's t-test ($p &lt; 0.05$). Series with high weekly variance or fewer than 4 non-zero sales observations are transparently flagged.
                </p>
              </div>

              {/* Trend Distribution Chart + Breakdown */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">
                  <h4 className="text-sm font-bold text-slate-900 mb-1">Trend Classification Distribution</h4>
                  <p className="text-xs text-slate-500 mb-3">Breakdown across evaluated FMCG categories</p>
                  <div className="h-52">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={trendPieData}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius={45}
                          outerRadius={70}
                          paddingAngle={3}
                        >
                          {trendPieData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip
                          formatter={(val, name) => [`${val} series`, name]}
                          contentStyle={{ backgroundColor: '#0f172a', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                        />
                        <Legend iconType="circle" wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 p-5 shadow-2xs flex flex-col justify-between">
                  <div>
                    <h4 className="text-sm font-bold text-slate-900 mb-1">Summary of Category Velocity Shifts</h4>
                    <p className="text-xs text-slate-500 mb-3">Comparing recent 4 weeks vs prior 8 weeks baseline</p>
                    <div className="grid grid-cols-3 gap-3 text-center mb-4">
                      <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-100">
                        <div className="text-xs font-semibold text-emerald-800">Rising Demand</div>
                        <div className="text-2xl font-bold text-emerald-700 mt-0.5">{trendSummary.rising}</div>
                        <div className="text-[10px] text-emerald-600 mt-1">Accelerating velocity</div>
                      </div>
                      <div className="p-3 rounded-lg bg-red-50 border border-red-100">
                        <div className="text-xs font-semibold text-red-800">Declining Demand</div>
                        <div className="text-2xl font-bold text-red-700 mt-0.5">{trendSummary.declining}</div>
                        <div className="text-[10px] text-red-600 mt-1">Softening velocity</div>
                      </div>
                      <div className="p-3 rounded-lg bg-blue-50 border border-blue-100">
                        <div className="text-xs font-semibold text-blue-800">Stable Demand</div>
                        <div className="text-2xl font-bold text-blue-700 mt-0.5">{trendSummary.stable}</div>
                        <div className="text-[10px] text-blue-600 mt-1">Steady state (&lt;±5%)</div>
                      </div>
                    </div>
                  </div>

                  <div className="text-xs text-slate-500 pt-2 border-t border-slate-100 flex items-center justify-between">
                    <span>Evaluated Complete Window: <strong>{trendsData?.comparison_period}</strong></span>
                    <span className="font-semibold text-brand-700">Strictly verified historical actuals</span>
                  </div>
                </div>
              </div>

              {/* Granular Demand Trends Table */}
              <DataTable
                title="Historical Demand Trend Classifier"
                columns={trendTableColumns}
                data={allTrends}
                searchKey="family"
                searchPlaceholder="Search product family..."
                pageSize={11}
                exportFilename="freshbasket_demand_trends.csv"
              />
            </div>
          )}

          {/* TAB 3: FORECAST OPPORTUNITIES & DIVERGENCE */}
          {activeTab === 'forecast' && (
            <div className="space-y-6">
              {/* Top Projected Demand Families Chart */}
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h3 className="text-base font-bold text-slate-900">
                      Projected 13-Week Demand Concentration (Top 10 Families)
                    </h3>
                    <p className="text-xs text-slate-500">
                      Forward volume projections for Aug 14 to Nov 12, 2017 ({forecastData?.selected_model_label || 'Naive'})
                    </p>
                  </div>
                  <Package className="w-4 h-4 text-slate-400" />
                </div>

                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={topFamilies} margin={{ top: 10, right: 20, left: 10, bottom: 25 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                      <XAxis dataKey="family" tick={{ fontSize: 9, fill: '#475569' }} angle={-35} textAnchor="end" />
                      <YAxis tick={{ fontSize: 10 }} tickFormatter={(val) => `${(val / 1e6).toFixed(1)}M`} />
                      <Tooltip
                        formatter={(val) => [`${Number(val).toLocaleString()} units`, 'Projected Demand']}
                        contentStyle={{ backgroundColor: '#0f172a', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                      />
                      <Bar dataKey="selected_units" fill="#059669" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Horizon Uncertainty Decay Curve */}
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h3 className="text-sm font-bold text-slate-900">
                      Prediction Uncertainty / Inter-Model Dispersion (Weeks 1 to 13)
                    </h3>
                    <p className="text-xs text-slate-500">
                      Quantified by Coefficient of Variation (CV %) across Naive, LightGBM, and Seasonal Naive
                    </p>
                  </div>
                  <Clock className="w-4 h-4 text-slate-400" />
                </div>

                <div className="h-60">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={horizonUncertainty} margin={{ top: 10, right: 20, left: 10, bottom: 25 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                      <XAxis dataKey="horizon_week" tick={{ fontSize: 10 }} tickFormatter={(val) => `Wk +${val}`} />
                      <YAxis tick={{ fontSize: 10 }} unit="%" />
                      <Tooltip
                        formatter={(val) => [`${Number(val).toFixed(2)}%`, 'Model Dispersion (CV)']}
                        contentStyle={{ backgroundColor: '#0f172a', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                      />
                      <Line
                        type="monotone"
                        dataKey="uncertainty_cv_pct"
                        stroke="#8b5cf6"
                        strokeWidth={2.5}
                        dot={{ r: 3.5, fill: '#8b5cf6' }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Model Divergence Table: Naive vs LightGBM */}
              <DataTable
                title="Model Divergence Analysis (LightGBM vs Naive Baseline)"
                columns={divergenceColumns}
                data={modelDivergence}
                searchKey="family"
                searchPlaceholder="Search product family..."
                pageSize={10}
                exportFilename="freshbasket_model_divergence.csv"
              />
            </div>
          )}

          {/* TAB 4: INVENTORY RISKS */}
          {activeTab === 'inventory' && (
            <div className="space-y-6">
              {/* Demo Mode / Data Source Alert */}
              <div
                className={`p-4 rounded-xl border text-xs flex items-start space-x-3 ${
                  inventoryData?.is_demo_data
                    ? 'bg-amber-50 border-amber-300 text-amber-900'
                    : 'bg-emerald-50 border-emerald-300 text-emerald-900'
                }`}
              >
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-amber-700" />
                <div className="space-y-1">
                  <span className="font-bold text-sm block">Inventory Data Source Indicator</span>
                  <p>{inventoryData?.demo_warning}</p>
                </div>
              </div>

              {/* Vulnerable Items Table */}
              <DataTable
                title="Priority High-Risk SKUs (Stock Below Lead-Time Demand)"
                columns={vulnerableColumns}
                data={inventoryData?.critical_vulnerable_items || []}
                searchKey="family"
                searchPlaceholder="Search by product family or store..."
                pageSize={10}
                exportFilename="freshbasket_critical_stockout_risks.csv"
              />

              {/* Long Lead-Time Exposures */}
              <DataTable
                title="Extended Supplier Lead-Time Vulnerabilities (Lead Time >= 14 Days)"
                columns={vulnerableColumns}
                data={inventoryData?.long_lead_time_vulnerabilities || []}
                searchKey="family"
                searchPlaceholder="Search long lead-time items..."
                pageSize={10}
                exportFilename="freshbasket_long_lead_time_vulnerabilities.csv"
              />
            </div>
          )}
        </>
      )}
    </div>
  );
};
