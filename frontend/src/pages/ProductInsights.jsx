import React, { useState, useEffect } from 'react';
import {
  Package,
  TrendingUp,
  Award,
  Layers,
  CheckCircle2,
  BarChart3,
  Percent
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
import { getProducts } from '../services/api';
import { KpiCard } from '../components/KpiCard';
import { FilterBar } from '../components/FilterBar';
import { DataTable } from '../components/DataTable';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';

export const ProductInsights = () => {
  const { selectedModel } = useFilters();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await getProducts({ model: selectedModel || undefined });
      setData(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [selectedModel]);

  const familyAccuracy = data?.family_accuracy || [];
  const familyVolumes = data?.family_volumes || [];
  const lgbmLeads = data?.categories_where_lgbm_leads || [];

  const tableColumns = [
    { key: 'family', header: 'Product Family', sortable: true },
    { key: 'model', header: 'Evaluated Model', sortable: true },
    {
      key: 'wape',
      header: 'WAPE %',
      sortable: true,
      align: 'right',
      render: (val) => {
        const num = Number(val);
        const color = num < 12 ? 'text-emerald-700 font-bold' : num < 20 ? 'text-amber-700' : 'text-red-700';
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
    {
      key: 'rmsle',
      header: 'RMSLE',
      sortable: true,
      align: 'right',
      render: (val) => (val !== null && val !== undefined ? Number(val).toFixed(4) : '—'),
    },
  ];

  return (
    <div className="space-y-6">
      {/* Top Filter Bar */}
      <FilterBar showModel={true} showStore={false} showFamily={false} />

      {loading && !data ? (
        <LoadingSpinner message="Calculating family-level error metrics and sales shares..." />
      ) : error && !data ? (
        <ErrorMessage message={error} onRetry={fetchData} />
      ) : (
        <>
          {/* Summary KPIs */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <KpiCard
              title="Evaluated FMCG Categories"
              value="33 Families"
              subtitle="Covering Grocery, Beverages, Perishables, Personal Care"
              icon={Package}
              color="emerald"
            />
            <KpiCard
              title="LightGBM Leading Categories"
              value={`${lgbmLeads.length} Categories`}
              subtitle="Where LightGBM outperformed Naive Baseline"
              icon={Award}
              color="blue"
              badge="High Velocity"
            />
            <KpiCard
              title="Top Category By Volume"
              value={familyVolumes[0]?.family || 'GROCERY I'}
              subtitle={`${familyVolumes[0] ? (familyVolumes[0].weekly_sales / 1e6).toFixed(1) : '0'}M cumulative historical units`}
              icon={TrendingUp}
              color="purple"
            />
          </div>

          {/* LightGBM Category Wins Callout */}
          {lgbmLeads.length > 0 && (
            <div className="bg-blue-50/75 border border-blue-200 rounded-xl p-4.5">
              <div className="flex items-center space-x-2 mb-2">
                <span className="w-2 h-2 rounded-full bg-blue-600" />
                <h4 className="text-xs font-bold uppercase tracking-wider text-blue-900">
                  Commercial Deep Dive: Categories Where LightGBM Outperforms Naive
                </h4>
              </div>
              <p className="text-xs text-blue-800 leading-relaxed mb-3">
                While Naive Baseline wins on global aggregate WAPE due to low-volume long-tail products, LightGBM achieves superior accuracy across high-velocity categories influenced by promotions:
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 text-xs">
                {lgbmLeads.slice(0, 8).map((cat) => (
                  <div key={cat.family} className="bg-white rounded-lg p-2 border border-blue-100 shadow-2xs">
                    <div className="font-bold text-slate-800 truncate">{cat.family}</div>
                    <div className="flex items-center justify-between mt-1 text-[11px]">
                      <span className="text-emerald-700 font-semibold">{cat.lightgbm_wape}%</span>
                      <span className="text-slate-400">vs {cat.naive_wape}%</span>
                      <span className="text-blue-600 font-bold">-{cat.improvement_pts} pts</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Family Sales Distribution Chart */}
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-base font-bold text-slate-900">
                  Product Family Historical Demand Ranking (Top 20)
                </h3>
                <p className="text-xs text-slate-500">
                  Total unit volume across all 54 retail branches
                </p>
              </div>
              <BarChart3 className="w-4 h-4 text-slate-400" />
            </div>

            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={familyVolumes.slice(0, 20)}
                  margin={{ top: 10, right: 20, left: 10, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis
                    dataKey="family"
                    tick={{ fontSize: 9, fill: '#475569' }}
                    angle={-45}
                    textAnchor="end"
                    interval={0}
                  />
                  <YAxis
                    tick={{ fontSize: 10 }}
                    tickFormatter={(val) => `${(val / 1e6).toFixed(1)}M`}
                  />
                  <Tooltip
                    formatter={(val) => [`${Number(val).toLocaleString()} units`, 'Sales Volume']}
                    contentStyle={{ backgroundColor: '#0f172a', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                  />
                  <Bar dataKey="weekly_sales" fill="#059669" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Detailed 33-Family Performance Table */}
          <DataTable
            title="Product Family Accuracy Matrix"
            columns={tableColumns}
            data={familyAccuracy}
            searchKey="family"
            searchPlaceholder="Search product family..."
            pageSize={11}
            exportFilename="freshbasket_product_family_metrics.csv"
          />
        </>
      )}
    </div>
  );
};
