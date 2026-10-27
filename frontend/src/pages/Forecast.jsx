import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  Calendar,
  Layers,
  Download,
  AlertCircle,
  HelpCircle,
  TrendingUp,
  Clock
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
} from 'recharts';
import { useFilters } from '../context/FilterContext';
import { getForecasts } from '../services/api';
import { KpiCard } from '../components/KpiCard';
import { FilterBar } from '../components/FilterBar';
import { DataTable } from '../components/DataTable';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';

export const Forecast = () => {
  const { selectedStore, selectedFamily, selectedModel } = useFilters();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await getForecasts({
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

  const weeklyForecast = data?.weekly_forecast || [];
  const meta = data?.metadata || {};
  const totalUnits = data?.total_forecast_units || 0;

  const tableColumns = [
    {
      key: 'horizon_week',
      header: 'Horizon Week',
      sortable: true,
      render: (val) => (
        <span className="font-bold text-brand-700 bg-brand-50 px-2 py-0.5 rounded-full text-xs">
          Week +{val}
        </span>
      ),
    },
    { key: 'week_start_date', header: 'Week Start (Mon)', sortable: true },
    { key: 'week_end_date', header: 'Week End (Sun)', sortable: true },
    {
      key: 'selected_forecast',
      header: `Selected (${meta.selected_model_label || 'Naive'})`,
      sortable: true,
      align: 'right',
      render: (val) => (
        <span className="font-bold text-slate-900">
          {Number(val).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}
        </span>
      ),
    },
    {
      key: 'naive_forecast',
      header: 'Naive Baseline',
      sortable: true,
      align: 'right',
      render: (val) => (
        <span className="text-slate-600">
          {Number(val).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}
        </span>
      ),
    },
    {
      key: 'lightgbm_forecast',
      header: 'LightGBM (Recursive)',
      sortable: true,
      align: 'right',
      render: (val) => (
        <span className="text-blue-600">
          {Number(val).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}
        </span>
      ),
    },
    {
      key: 'seasonal_naive_forecast',
      header: 'Seasonal Naive (Lag 52)',
      sortable: true,
      align: 'right',
      render: (val) => (
        <span className="text-amber-600">
          {Number(val).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}
        </span>
      ),
    },
  ];

  return (
    <div className="space-y-6">
      {/* Filters */}
      <FilterBar showModel={true} showStore={true} showFamily={true} />

      {loading && !data ? (
        <LoadingSpinner message="Retrieving 13-week forward predictions..." />
      ) : error && !data ? (
        <ErrorMessage message={error} onRetry={fetchData} />
      ) : (
        <>
          {/* Horizon Metadata Header Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <KpiCard
              title="13-Week Cumulative Demand"
              value={`${(totalUnits / 1e3).toFixed(1)}k units`}
              subtitle={`Model: ${meta.selected_model_label || 'Naive'}`}
              icon={Sparkles}
              color="emerald"
            />
            <KpiCard
              title="Forecast Origin Date"
              value="Aug 13, 2017"
              subtitle="End of last verified full historical week"
              icon={Clock}
              color="blue"
            />
            <KpiCard
              title="Prediction Window"
              value="13 Full Weeks"
              subtitle="Aug 14, 2017 – Nov 12, 2017"
              icon={Calendar}
              color="purple"
            />
            <KpiCard
              title="Active Forecasting Model"
              value={meta.selected_model_label?.split(' ')[0] || 'Naive'}
              subtitle={
                meta.selected_model === 'naive_forecast'
                  ? 'Rank #1 Verified (11.61% WAPE)'
                  : meta.selected_model === 'lightgbm_forecast'
                  ? 'Rank #2 (12.27% WAPE)'
                  : 'Rank #3 (16.17% WAPE)'
              }
              icon={Layers}
              color="amber"
              badge={meta.selected_model === 'naive_forecast' ? 'Best WAPE' : undefined}
            />
          </div>

          {/* Model Trajectory Chart */}
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
              <div>
                <h3 className="text-base font-bold text-slate-900">
                  13-Week Multi-Model Demand Predictions (Aug 14 – Nov 12, 2017)
                </h3>
                <p className="text-xs text-slate-500">
                  Comparison between Naive, LightGBM Recursive, and Seasonal Naive (Lag 52) across the 13-week planning horizon.
                </p>
              </div>
              <div className="flex items-center space-x-4 text-xs">
                <div className="flex items-center space-x-1.5">
                  <span className="w-3 h-3 rounded-full bg-emerald-600" />
                  <span className="text-slate-700 font-medium">Naive Baseline</span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <span className="w-3 h-3 rounded-full bg-blue-600" />
                  <span className="text-slate-700 font-medium">LightGBM (Recursive)</span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <span className="w-3 h-3 rounded-full bg-amber-500" />
                  <span className="text-slate-700 font-medium">Seasonal Naive</span>
                </div>
              </div>
            </div>

            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={weeklyForecast} margin={{ top: 10, right: 20, left: 10, bottom: 25 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis
                    dataKey="week_start_date"
                    tick={{ fontSize: 11, fill: '#64748b' }}
                    tickFormatter={(val, idx) => `Wk +${idx + 1}`}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: '#64748b' }}
                    tickFormatter={(val) => `${(val / 1e3).toFixed(0)}k`}
                  />
                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (!active || !payload?.length) return null;
                      const item = payload[0]?.payload;
                      return (
                        <div className="bg-slate-900 text-white rounded-lg p-3 text-xs shadow-xl border border-slate-800 space-y-1.5">
                          <div className="font-bold text-slate-300 border-b border-slate-700 pb-1">
                            Week +{item.horizon_week}: {item.week_start_date} to {item.week_end_date}
                          </div>
                          <div className="flex justify-between gap-4 text-emerald-400 font-semibold">
                            <span>Naive Baseline:</span>
                            <span>{Number(item.naive_forecast).toLocaleString()} units</span>
                          </div>
                          <div className="flex justify-between gap-4 text-blue-400 font-semibold">
                            <span>LightGBM (Recursive):</span>
                            <span>{Number(item.lightgbm_forecast).toLocaleString()} units</span>
                          </div>
                          <div className="flex justify-between gap-4 text-amber-400">
                            <span>Seasonal Naive (Lag 52):</span>
                            <span>{Number(item.seasonal_naive_forecast).toLocaleString()} units</span>
                          </div>
                        </div>
                      );
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="naive_forecast"
                    stroke="#059669"
                    strokeWidth={2.5}
                    dot={{ r: 4, fill: '#059669' }}
                    name="Naive Baseline"
                  />
                  <Line
                    type="monotone"
                    dataKey="lightgbm_forecast"
                    stroke="#2563eb"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={{ r: 3.5, fill: '#2563eb' }}
                    name="LightGBM"
                  />
                  <Line
                    type="monotone"
                    dataKey="seasonal_naive_forecast"
                    stroke="#d97706"
                    strokeWidth={1.5}
                    strokeDasharray="2 2"
                    dot={{ r: 2.5, fill: '#d97706' }}
                    name="Seasonal Naive"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* 13-Week Detailed Tabular Forecast */}
          <DataTable
            title="Thirteen-Week Forecast Matrix"
            columns={tableColumns}
            data={weeklyForecast}
            searchKey="week_start_date"
            searchPlaceholder="Search week dates..."
            pageSize={13}
            exportFilename="freshbasket_13_week_predictions.csv"
          />

          {/* Business Guidance Note */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs text-slate-600 flex items-start space-x-3">
            <HelpCircle className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-semibold text-slate-800">
                Methodological Note on 13-Week Forecast Inputs:
              </span>
              <p>
                All 13-week forecast numbers represent pure forward predictions made from origin date <code>2017-08-13</code>. No future sales or actual observations beyond August 13 were provided to models during inference. In recursive autoregressive models (LightGBM), forecasted outputs are chained as input features for subsequent horizon steps.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
