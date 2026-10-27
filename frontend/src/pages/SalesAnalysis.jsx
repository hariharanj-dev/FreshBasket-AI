import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  Calendar,
  DollarSign,
  Download,
  BarChart2,
  Filter,
  Layers
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
} from 'recharts';
import { useFilters } from '../context/FilterContext';
import { getSales } from '../services/api';
import { KpiCard } from '../components/KpiCard';
import { FilterBar } from '../components/FilterBar';
import { DataTable } from '../components/DataTable';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';

export const SalesAnalysis = () => {
  const { selectedStore, selectedFamily, filterOptions } = useFilters();
  const [dateRange, setDateRange] = useState({ start: '', end: '' });
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await getSales({
        store_nbr: selectedStore || undefined,
        family: selectedFamily || undefined,
        start_date: dateRange.start || undefined,
        end_date: dateRange.end || undefined,
        exclude_partial_weeks: true,
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
  }, [selectedStore, selectedFamily, dateRange.start, dateRange.end]);

  const summary = data?.summary || {};
  const timeSeries = data?.time_series || [];
  const topStores = data?.top_stores || [];
  const topFamilies = data?.top_families || [];
  const monthlySummary = data?.monthly_summary || [];

  const tableColumns = [
    { key: 'date', header: 'Week Starting Date', sortable: true },
    {
      key: 'weekly_sales',
      header: 'Weekly Sales Units',
      sortable: true,
      align: 'right',
      render: (val) => Number(val).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 }),
    },
    {
      key: 'moving_avg_4wk',
      header: '4-Wk Moving Avg',
      sortable: true,
      align: 'right',
      render: (val) => Number(val).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 }),
    },
    {
      key: 'onpromotion_items',
      header: 'Promotion Items',
      sortable: true,
      align: 'right',
      render: (val) => Number(val).toLocaleString(),
    },
    {
      key: 'transactions',
      header: 'Store Transactions',
      sortable: true,
      align: 'right',
      render: (val) => Number(val).toLocaleString(),
    },
  ];

  return (
    <div className="space-y-6">
      {/* Filters with Date Range */}
      <FilterBar
        showModel={false}
        showStore={true}
        showFamily={true}
        showDateRange={true}
        startDate={dateRange.start}
        endDate={dateRange.end}
        onDateChange={({ start, end }) => setDateRange({ start, end })}
      />

      {loading && !data ? (
        <LoadingSpinner message="Calculating weekly historical aggregations..." />
      ) : error && !data ? (
        <ErrorMessage message={error} onRetry={fetchData} />
      ) : (
        <>
          {/* Summary KPIs */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <KpiCard
              title="Aggregated Sales Volume"
              value={summary.total_sales ? `${(summary.total_sales / 1e6).toFixed(2)}M units` : '0 units'}
              subtitle={`Filter: Store ${summary.store_filter} | Family: ${summary.family_filter}`}
              icon={DollarSign}
              color="emerald"
            />
            <KpiCard
              title="Observed Complete Weeks"
              value={`${summary.weeks_count ?? 0} Weeks`}
              subtitle="Excluding partial cutoff week (Aug 14, 2017)"
              icon={Calendar}
              color="blue"
            />
            <KpiCard
              title="Average Weekly Velocity"
              value={summary.avg_weekly_sales ? `${Number(summary.avg_weekly_sales).toLocaleString(undefined, { maximumFractionDigits: 0 })} units/wk` : '0'}
              subtitle="Mean weekly run rate across selected scope"
              icon={TrendingUp}
              color="purple"
            />
          </div>

          {/* Time Series Chart: Weekly Sales + 4-Week Moving Average */}
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
              <div>
                <h3 className="text-base font-bold text-slate-900">
                  Historical Weekly Sales Trend &amp; 4-Week Moving Average
                </h3>
                <p className="text-xs text-slate-500">
                  Smoothed moving average reveals underlying seasonal patterns and demand baseline
                </p>
              </div>
              <div className="flex items-center space-x-3 text-xs">
                <div className="flex items-center space-x-1.5">
                  <span className="w-3 h-3 rounded bg-emerald-500 opacity-60" />
                  <span className="text-slate-600 font-medium">Weekly Sales</span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <span className="w-3 h-0.5 bg-blue-600 font-bold" />
                  <span className="text-slate-600 font-medium">4-Wk Moving Avg</span>
                </div>
              </div>
            </div>

            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={timeSeries} margin={{ top: 10, right: 20, left: 10, bottom: 25 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 11, fill: '#64748b' }}
                    tickFormatter={(val) => val ? val.slice(0, 7) : ''}
                    interval="preserveStartEnd"
                    minTickGap={40}
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
                        <div className="bg-slate-900 text-white rounded-lg p-3 text-xs shadow-xl border border-slate-800 space-y-1">
                          <div className="font-bold text-slate-300 border-b border-slate-700 pb-1">
                            Week: {label}
                          </div>
                          <div className="flex justify-between gap-4 text-emerald-400 font-semibold">
                            <span>Weekly Sales:</span>
                            <span>{Number(item.weekly_sales).toLocaleString()} units</span>
                          </div>
                          <div className="flex justify-between gap-4 text-blue-400">
                            <span>4-Wk Moving Avg:</span>
                            <span>{Number(item.moving_avg_4wk).toLocaleString()} units</span>
                          </div>
                          {item.onpromotion_items > 0 && (
                            <div className="flex justify-between gap-4 text-amber-300">
                              <span>Promotion Items:</span>
                              <span>{Number(item.onpromotion_items).toLocaleString()}</span>
                            </div>
                          )}
                          {item.transactions > 0 && (
                            <div className="flex justify-between gap-4 text-slate-300">
                              <span>Transactions:</span>
                              <span>{Number(item.transactions).toLocaleString()}</span>
                            </div>
                          )}
                        </div>
                      );
                    }}
                  />
                  <Bar
                    dataKey="weekly_sales"
                    fill="#10b981"
                    opacity={0.65}
                    radius={[2, 2, 0, 0]}
                    name="Weekly Sales"
                  />
                  <Line
                    type="monotone"
                    dataKey="moving_avg_4wk"
                    stroke="#2563eb"
                    strokeWidth={2.5}
                    dot={false}
                    name="4-Wk Moving Avg"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Breakdown Charts: Top Stores & Monthly Seasonality */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Top Stores by Volume */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-bold text-slate-900">
                  Sales by Store (Top 15 Branches)
                </h4>
                <span className="text-xs text-slate-400">Filtered Window</span>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={topStores} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                    <XAxis
                      type="number"
                      tick={{ fontSize: 11 }}
                      tickFormatter={(val) => `${(val / 1e3).toFixed(0)}k`}
                    />
                    <YAxis dataKey="store_label" type="category" tick={{ fontSize: 10 }} />
                    <Tooltip
                      formatter={(val) => [`${Number(val).toLocaleString()} units`, 'Sales Volume']}
                      contentStyle={{ backgroundColor: '#0f172a', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                    />
                    <Bar dataKey="weekly_sales" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Monthly Trend */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-bold text-slate-900">
                  Monthly Aggregated Sales (Recent 24 Months)
                </h4>
                <span className="text-xs text-slate-400">Macro Trajectory</span>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={monthlySummary} margin={{ top: 5, right: 20, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                    <XAxis dataKey="year_month" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" />
                    <YAxis
                      tick={{ fontSize: 10 }}
                      tickFormatter={(val) => `${(val / 1e3).toFixed(0)}k`}
                    />
                    <Tooltip
                      formatter={(val) => [`${Number(val).toLocaleString()} units`, 'Monthly Sales']}
                      contentStyle={{ backgroundColor: '#0f172a', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                    />
                    <Bar dataKey="weekly_sales" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Granular Weekly Sales Table */}
          <DataTable
            title="Weekly Sales Records"
            columns={tableColumns}
            data={timeSeries}
            searchKey="date"
            searchPlaceholder="Search by date (YYYY-MM-DD)..."
            pageSize={10}
            exportFilename="freshbasket_historical_weekly_sales.csv"
          />
        </>
      )}
    </div>
  );
};
