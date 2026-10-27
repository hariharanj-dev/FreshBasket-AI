import React, { useState, useEffect } from 'react';
import {
  Store,
  MapPin,
  TrendingUp,
  BarChart3,
  Building,
  Award,
  Layers
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from 'recharts';
import { useFilters } from '../context/FilterContext';
import { getStores } from '../services/api';
import { KpiCard } from '../components/KpiCard';
import { FilterBar } from '../components/FilterBar';
import { DataTable } from '../components/DataTable';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';

export const StorePerformance = () => {
  const { selectedStore, selectedModel } = useFilters();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await getStores({
        store_nbr: selectedStore || undefined,
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
  }, [selectedStore, selectedModel]);

  const storeRankings = data?.store_rankings || [];
  const typeSummary = data?.type_summary || [];

  const tableColumns = [
    {
      key: 'store_nbr',
      header: 'Store #',
      sortable: true,
      render: (val) => <span className="font-bold text-slate-900">Store {val}</span>,
    },
    { key: 'city', header: 'City', sortable: true },
    { key: 'state', header: 'State / Province', sortable: true },
    {
      key: 'type',
      header: 'Store Type',
      sortable: true,
      render: (val) => (
        <span className="px-2 py-0.5 rounded-md bg-slate-100 text-slate-700 font-semibold text-xs border border-slate-200">
          Type {val}
        </span>
      ),
    },
    { key: 'cluster', header: 'Cluster', sortable: true },
    {
      key: 'actual_sales_volume',
      header: 'Evaluation Sales',
      sortable: true,
      align: 'right',
      render: (val) => Number(val).toLocaleString(undefined, { maximumFractionDigits: 0 }),
    },
    {
      key: 'wape',
      header: 'WAPE %',
      sortable: true,
      align: 'right',
      render: (val) => {
        const num = Number(val);
        const color = num < 11 ? 'text-emerald-700 font-bold' : num < 16 ? 'text-blue-700' : 'text-amber-700';
        return <span className={color}>{num.toFixed(2)}%</span>;
      },
    },
    {
      key: 'mae',
      header: 'MAE',
      sortable: true,
      align: 'right',
      render: (val) => Number(val).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 }),
    },
    {
      key: 'rmse',
      header: 'RMSE',
      sortable: true,
      align: 'right',
      render: (val) => Number(val).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 }),
    },
  ];

  return (
    <div className="space-y-6">
      {/* Filters */}
      <FilterBar showModel={true} showStore={true} showFamily={false} />

      {loading && !data ? (
        <LoadingSpinner message="Calculating store performance metrics across 54 retail branches..." />
      ) : error && !data ? (
        <ErrorMessage message={error} onRetry={fetchData} />
      ) : (
        <>
          {/* Summary KPIs */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <KpiCard
              title="Monitored Retail Network"
              value={`${storeRankings.length} Branches`}
              subtitle="54 active retail stores across Ecuador"
              icon={Store}
              color="emerald"
            />
            <KpiCard
              title="Top Performing Branch (Accuracy)"
              value={storeRankings[0] ? `Store ${storeRankings[0].store_nbr}` : '—'}
              subtitle={storeRankings[0] ? `${storeRankings[0].city} (${storeRankings[0].wape}% WAPE)` : '—'}
              icon={Award}
              color="blue"
              badge="Lowest Error"
            />
            <KpiCard
              title="Store Formats Tracked"
              value={typeSummary.length ? `${typeSummary.length} Formats (A–E)` : '5 Formats'}
              subtitle="Covering Hypermarkets, Supermarkets, Express stores"
              icon={Building}
              color="purple"
            />
          </div>

          {/* Store Type Format Summary */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {typeSummary.map((t) => (
              <div
                key={t.type}
                className="bg-white rounded-xl border border-slate-200 p-3.5 shadow-2xs text-center"
              >
                <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                  Type {t.type}
                </div>
                <div className="text-xl font-bold text-slate-900 mt-0.5">
                  {t.store_count} <span className="text-xs font-normal text-slate-500">stores</span>
                </div>
                <div className="text-xs text-brand-700 font-semibold mt-1">
                  Avg WAPE: {t.avg_wape}%
                </div>
              </div>
            ))}
          </div>

          {/* Top 20 Stores by Volume Chart */}
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-base font-bold text-slate-900">
                  Store Network Performance (Top 20 Branches by Volume)
                </h3>
                <p className="text-xs text-slate-500">
                  Historical evaluation sales volume and branch accuracy
                </p>
              </div>
              <BarChart3 className="w-4 h-4 text-slate-400" />
            </div>

            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={storeRankings.slice(0, 20)}
                  margin={{ top: 10, right: 20, left: 10, bottom: 25 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis
                    dataKey="store_nbr"
                    tick={{ fontSize: 10, fill: '#475569' }}
                    tickFormatter={(val) => `Store ${val}`}
                  />
                  <YAxis
                    tick={{ fontSize: 10 }}
                    tickFormatter={(val) => `${(val / 1e3).toFixed(0)}k`}
                  />
                  <Tooltip
                    formatter={(val, name) => [
                      name === 'actual_sales_volume'
                        ? `${Number(val).toLocaleString()} units`
                        : `${val}%`,
                      name === 'actual_sales_volume' ? 'Sales Volume' : 'WAPE Error',
                    ]}
                    contentStyle={{ backgroundColor: '#0f172a', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                  />
                  <Bar dataKey="actual_sales_volume" fill="#3b82f6" radius={[4, 4, 0, 0]} name="actual_sales_volume" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Store Table */}
          <DataTable
            title="54-Store Performance Leaderboard"
            columns={tableColumns}
            data={storeRankings}
            searchKey="city"
            searchPlaceholder="Search store by city or state..."
            pageSize={10}
            exportFilename="freshbasket_store_performance_rankings.csv"
          />
        </>
      )}
    </div>
  );
};
