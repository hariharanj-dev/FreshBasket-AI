import React, { useState, useEffect, useRef } from 'react';
import {
  Boxes,
  Upload,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Download,
  RotateCcw,
  ShieldAlert,
  Package,
  RefreshCw
} from 'lucide-react';
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
} from 'recharts';
import { useFilters } from '../context/FilterContext';
import {
  getInventorySummary,
  getInventoryRecommendations,
  uploadInventoryCSV,
  resetInventory,
  downloadReplenishmentCSV,
} from '../services/api';
import { KpiCard } from '../components/KpiCard';
import { FilterBar } from '../components/FilterBar';
import { DataTable } from '../components/DataTable';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';
import { EmptyState } from '../components/EmptyState';
import { ErrorBoundary } from '../components/ErrorBoundary';

const RISK_COLORS = {
  HIGH: '#ef4444',
  MEDIUM: '#f59e0b',
  LOW: '#10b981',
  UNKNOWN: '#64748b',
};

export const InventoryPlanning = () => {
  const {
    selectedStore,
    setSelectedStore,
    selectedFamily,
    setSelectedFamily,
    selectedModel,
    dataMode,
    setDataMode,
    refreshMetadata,
  } = useFilters();

  const [selectedRisk, setSelectedRisk] = useState('');
  const [summary, setSummary] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Upload state
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const fileInputRef = useRef(null);

  const fetchInventoryData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [sumRes, recRes] = await Promise.all([
        getInventorySummary({
          model: selectedModel || undefined,
          store_nbr: selectedStore || undefined,
          family: selectedFamily || undefined,
        }),
        getInventoryRecommendations({
          model: selectedModel || undefined,
          store_nbr: selectedStore || undefined,
          family: selectedFamily || undefined,
          risk_level: selectedRisk || undefined,
          limit: 1000,
        }),
      ]);
      setSummary(sumRes);
      setRecommendations(recRes.records || []);
    } catch (err) {
      setError(err.message || 'Unable to retrieve inventory data from server.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchInventoryData();
  }, [selectedStore, selectedFamily, selectedModel, selectedRisk, dataMode]);

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setUploading(true);
      setUploadError(null);
      setUploadSuccess(null);
      const res = await uploadInventoryCSV(file);
      setUploadSuccess(res.message);
      setDataMode('CUSTOM');
      await refreshMetadata();
      await fetchInventoryData();
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleResetDemo = async () => {
    try {
      setLoading(true);
      await resetInventory();
      setDataMode('DEMO');
      setUploadSuccess('Reset to baseline demo inventory successfully.');
      setUploadError(null);
      await refreshMetadata();
      await fetchInventoryData();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadCsv = () => {
    downloadReplenishmentCSV({
      model: selectedModel || undefined,
      store_nbr: selectedStore || undefined,
      family: selectedFamily || undefined,
      risk_level: selectedRisk || undefined,
    });
  };

  const riskPieData = summary
    ? [
        { name: 'High Stockout Risk', value: summary.high_risk_count ?? 0, color: RISK_COLORS.HIGH },
        { name: 'Medium Risk Buffer', value: summary.medium_risk_count ?? 0, color: RISK_COLORS.MEDIUM },
        { name: 'Low Risk (Healthy)', value: summary.low_risk_count ?? 0, color: RISK_COLORS.LOW },
        ...(summary.unknown_risk_count > 0
          ? [{ name: 'Unknown / Invalid', value: summary.unknown_risk_count, color: RISK_COLORS.UNKNOWN }]
          : []),
      ]
    : [];

  const tableColumns = [
    {
      key: 'stockout_risk',
      header: 'Risk Level',
      sortable: true,
      render: (val) => {
        const bg =
          val === 'HIGH'
            ? 'bg-red-100 text-red-800 border-red-200'
            : val === 'MEDIUM'
            ? 'bg-amber-100 text-amber-800 border-amber-200'
            : val === 'LOW'
            ? 'bg-emerald-100 text-emerald-800 border-emerald-200'
            : 'bg-slate-100 text-slate-800 border-slate-200';
        return (
          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold border ${bg}`}>
            {val === 'HIGH' ? '🔴 HIGH' : val === 'MEDIUM' ? '🟡 MEDIUM' : val === 'LOW' ? '🟢 LOW' : '⚪ UNKNOWN'}
          </span>
        );
      },
    },
    {
      key: 'store_nbr',
      header: 'Store #',
      sortable: true,
      render: (val) => <span className="font-semibold text-slate-900">Store {val}</span>,
    },
    { key: 'family', header: 'Product Family', sortable: true },
    {
      key: 'current_stock',
      header: 'Current Stock',
      sortable: true,
      align: 'right',
      render: (val) => (val !== null && val !== undefined && !isNaN(val) ? Number(val).toLocaleString() : '—'),
    },
    {
      key: 'expected_lt_demand',
      header: 'Lead-Time Demand',
      sortable: true,
      align: 'right',
      render: (val) => (val !== null && val !== undefined && !isNaN(val) ? Number(val).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '—'),
    },
    {
      key: 'safety_stock',
      header: 'Safety Stock',
      sortable: true,
      align: 'right',
      render: (val) => (val !== null && val !== undefined && !isNaN(val) ? Number(val).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '—'),
    },
    {
      key: 'reorder_point',
      header: 'Reorder Point (ROP)',
      sortable: true,
      align: 'right',
      render: (val) => (val !== null && val !== undefined && !isNaN(val) ? Number(val).toLocaleString(undefined, { maximumFractionDigits: 1 }) : '—'),
    },
    {
      key: 'suggested_replenishment_qty',
      header: 'Suggested Order Qty',
      sortable: true,
      align: 'right',
      render: (val) => (
        <span className={`font-bold ${Number(val) > 0 ? 'text-red-600' : 'text-slate-500'}`}>
          {val !== null && val !== undefined && !isNaN(val) ? `${Number(val).toLocaleString()} units` : '0 units'}
        </span>
      ),
    },
    {
      key: 'supplier_lead_time_days',
      header: 'Lead Time',
      sortable: true,
      align: 'right',
      render: (val, row) => `${val ?? row?.lead_time_days ?? 7}d`,
    },
    {
      key: 'supplier_name',
      header: 'Supplier',
      sortable: true,
      render: (val, row) => (
        <span className="text-slate-600 truncate max-w-[140px] inline-block" title={val ?? row?.primary_supplier}>
          {val ?? row?.primary_supplier ?? 'Standard Supplier'}
        </span>
      ),
    },
  ];

  return (
    <div className="space-y-6">
      {/* Filters with Risk Tier Selector */}
      <FilterBar
        showModel={true}
        showStore={true}
        showFamily={true}
        showRisk={true}
        selectedRisk={selectedRisk}
        onRiskChange={setSelectedRisk}
      />

      {/* Data Source Mode & CSV Upload Panel */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center space-x-2.5">
              <span
                className={`px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider ${
                  dataMode === 'CUSTOM'
                    ? 'bg-amber-100 text-amber-800 border border-amber-300'
                    : 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                }`}
              >
                {dataMode === 'CUSTOM' ? '⚡ Active: Uploaded Store CSV' : '🟢 Active: Baseline Demo Inventory'}
              </span>
              <span className="text-xs text-slate-500">
                {dataMode === 'CUSTOM'
                  ? 'Calculations driven by user-uploaded stock balances'
                  : 'Synthesized retail inventory positions (54 stores × 33 families)'}
              </span>
            </div>
            <p className="text-xs text-slate-600 max-w-2xl leading-relaxed">
              Upload custom store stock levels to recalculate reorder points (ROP), lead-time demand, stockout risk tiers, and discrete replenishment units.
            </p>
          </div>

          <div className="flex items-center space-x-2 shrink-0">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".csv"
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="inline-flex items-center space-x-2 px-3.5 py-2 bg-brand-600 hover:bg-brand-700 text-white text-xs font-semibold rounded-lg transition-colors shadow-2xs disabled:opacity-50 cursor-pointer"
            >
              <Upload className={`w-3.5 h-3.5 ${uploading ? 'animate-bounce' : ''}`} />
              <span>{uploading ? 'Validating CSV...' : 'Upload Inventory CSV'}</span>
            </button>

            {dataMode === 'CUSTOM' && (
              <button
                onClick={handleResetDemo}
                className="inline-flex items-center space-x-1.5 px-3.5 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold rounded-lg transition-colors border border-slate-200 cursor-pointer"
                title="Reset to demo baseline"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset Demo</span>
              </button>
            )}

            <button
              onClick={handleDownloadCsv}
              className="inline-flex items-center space-x-1.5 px-3.5 py-2 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold rounded-lg transition-colors border border-slate-200 shadow-2xs cursor-pointer"
              title="Download replenishment plan CSV"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export CSV</span>
            </button>
          </div>
        </div>

        {/* Upload feedback */}
        {uploadSuccess && (
          <div className="mt-3.5 p-3 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center space-x-2">
            <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-600" />
            <span>{uploadSuccess}</span>
          </div>
        )}
        {uploadError && (
          <div className="mt-3.5 p-3 rounded-lg bg-red-50 border border-red-200 text-red-800 text-xs flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 shrink-0 text-red-600" />
            <span>{uploadError}</span>
          </div>
        )}
      </div>

      {/* Recoverable Error Banner if summary already loaded */}
      {error && summary && (
        <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
            <span>Failed to refresh latest inventory parameters: {error}</span>
          </div>
          <button
            onClick={fetchInventoryData}
            className="inline-flex items-center space-x-1 px-2.5 py-1 bg-amber-600 text-white rounded-md hover:bg-amber-700 font-semibold text-[11px]"
          >
            <RefreshCw className="w-3 h-3" />
            <span>Retry</span>
          </button>
        </div>
      )}

      {/* Main Content Area */}
      {loading && !summary ? (
        <LoadingSpinner message="Evaluating stockout risk and replenishment parameters..." />
      ) : error && !summary ? (
        <ErrorMessage
          title="Inventory Service Temporarily Unavailable"
          message={error}
          onRetry={fetchInventoryData}
        />
      ) : (
        <>
          {/* Risk Scorecard KPIs */}
          <ErrorBoundary title="Risk Scorecard Error">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              <KpiCard
                title="High Stockout Risk"
                value={summary?.high_risk_count ?? 0}
                subtitle="Stock < Lead Demand"
                icon={ShieldAlert}
                color="red"
                badge="Urgent"
                badgeColor="red"
              />
              <KpiCard
                title="Medium Risk"
                value={summary?.medium_risk_count ?? 0}
                subtitle="Buffer Breached"
                icon={AlertTriangle}
                color="amber"
                badge="Caution"
                badgeColor="amber"
              />
              <KpiCard
                title="Low Risk (Healthy)"
                value={summary?.low_risk_count ?? 0}
                subtitle="Stock >= Reorder Point"
                icon={CheckCircle2}
                color="emerald"
              />
              <KpiCard
                title="Monitored Items"
                value={summary?.total_items ?? 0}
                subtitle="Store-family series"
                icon={Boxes}
                color="slate"
              />
              <KpiCard
                title="Lead-Time Demand"
                value={
                  summary?.total_expected_lt_demand !== undefined
                    ? `${(summary.total_expected_lt_demand / 1e3).toFixed(0)}k`
                    : '0'
                }
                subtitle="Aggregated demand"
                icon={Clock}
                color="blue"
              />
              <KpiCard
                title="Suggested Orders"
                value={
                  summary?.total_suggested_replenishment_qty !== undefined
                    ? `${(summary.total_suggested_replenishment_qty / 1e3).toFixed(0)}k`
                    : '0'
                }
                subtitle="Total replenishment units"
                icon={Package}
                color="purple"
              />
            </div>
          </ErrorBoundary>

          {/* Risk Distribution & Business Insights Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Risk Distribution Donut Chart */}
            <ErrorBoundary title="Risk Distribution Chart Error">
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">
                <h4 className="text-sm font-bold text-slate-900 mb-1">
                  Stockout Risk Tier Distribution
                </h4>
                <p className="text-xs text-slate-500 mb-3">
                  Classification based on lead-time demand and safety stock
                </p>

                <div className="h-56">
                  {riskPieData.length > 0 && riskPieData.some((d) => d.value > 0) ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={riskPieData}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius={50}
                          outerRadius={75}
                          paddingAngle={3}
                        >
                          {riskPieData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip
                          formatter={(val, name) => [`${val} items`, name]}
                          contentStyle={{
                            backgroundColor: '#0f172a',
                            borderRadius: '8px',
                            color: '#fff',
                            fontSize: '12px',
                          }}
                        />
                        <Legend iconType="circle" wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                      </PieChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-slate-400">
                      No distribution data available for active filters.
                    </div>
                  )}
                </div>
              </div>
            </ErrorBoundary>

            {/* Rule-Based Executive Insights Panel */}
            <ErrorBoundary title="Executive Insights Error">
              <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 p-5 shadow-2xs flex flex-col justify-between">
                <div>
                  <h4 className="text-sm font-bold text-slate-900 mb-1">
                    Automated Replenishment Heuristics &amp; Insights
                  </h4>
                  <p className="text-xs text-slate-500 mb-4 font-mono">
                    {'Deterministic reorder formulas: ROP = D̂_LT + SS, Q = ⌈max(0, ROP - Stock)⌉'}
                  </p>

                  <div className="space-y-2.5">
                    {(summary?.business_insights || []).length > 0 ? (
                      summary.business_insights.map((insight, idx) => (
                        <div
                          key={idx}
                          className="p-3 rounded-lg bg-slate-50 border border-slate-100 text-xs text-slate-700 flex items-start space-x-2.5"
                        >
                          <span className="w-1.5 h-1.5 rounded-full bg-brand-600 mt-1.5 shrink-0" />
                          <span className="leading-relaxed">{insight}</span>
                        </div>
                      ))
                    ) : (
                      <p className="text-xs text-slate-400 italic">No automated insights generated for active selection.</p>
                    )}
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
                  <span>Calculations strictly powered by Python backend heuristics</span>
                  <span className="font-semibold text-brand-700">Zero frontend calculations</span>
                </div>
              </div>
            </ErrorBoundary>
          </div>

          {/* Replenishment Recommendations Table */}
          <ErrorBoundary title="Recommendations Table Error">
            {recommendations.length === 0 ? (
              <EmptyState
                title="No Inventory Recommendations Found"
                message="No store-family records match the current filter selection. Try choosing 'All Risk Tiers' or clearing filters."
                onClearFilters={() => {
                  setSelectedStore('');
                  setSelectedFamily('');
                  setSelectedRisk('');
                }}
              />
            ) : (
              <DataTable
                title="Item-Level Replenishment Recommendations"
                columns={tableColumns}
                data={recommendations}
                searchKey="family"
                searchPlaceholder="Search by product family or store..."
                pageSize={15}
                onExportCsv={handleDownloadCsv}
                exportFilename="freshbasket_replenishment_recommendations.csv"
              />
            )}
          </ErrorBoundary>
        </>
      )}
    </div>
  );
};
