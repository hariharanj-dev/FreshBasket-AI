import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  Award,
  AlertTriangle,
  Info,
  CheckCircle2,
  TrendingDown,
  Layers,
  HelpCircle,
  Clock
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { getEvaluation } from '../services/api';
import { KpiCard } from '../components/KpiCard';
import { DataTable } from '../components/DataTable';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';

export const ModelEvaluation = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await getEvaluation();
      setData(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const globalMatrix = data?.global_matrix || [];
  const horizonDecay = data?.horizon_decay || [];
  const rmsleAudit = data?.rmsle_audit || {};

  const matrixColumns = [
    {
      key: 'rank',
      header: 'Rank',
      sortable: true,
      render: (val) => (
        <span
          className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
            val === 1
              ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
              : val === 2
              ? 'bg-blue-100 text-blue-800 border border-blue-200'
              : 'bg-amber-100 text-amber-800 border border-amber-200'
          }`}
        >
          #{val}
        </span>
      ),
    },
    {
      key: 'model_label',
      header: 'Forecasting Model',
      sortable: true,
      render: (val, row) => (
        <div>
          <span className="font-bold text-slate-900">{val}</span>
          {row.rank === 1 && (
            <span className="ml-2 px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 text-[10px] font-bold">
              Best Overall (WAPE)
            </span>
          )}
        </div>
      ),
    },
    {
      key: 'wape',
      header: 'WAPE %',
      sortable: true,
      align: 'right',
      render: (val, row) => (
        <span className={`font-bold ${row.rank === 1 ? 'text-emerald-700 text-sm' : 'text-slate-800'}`}>
          {Number(val).toFixed(2)}%
        </span>
      ),
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
      {loading && !data ? (
        <LoadingSpinner message="Loading validated model evaluation metrics..." />
      ) : error && !data ? (
        <ErrorMessage message={error} onRetry={fetchData} />
      ) : (
        <>
          {/* Top KPI Cards for Global Matrix */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <KpiCard
              title="Verified Best Model (WAPE)"
              value="Naive Baseline"
              subtitle="11.61% WAPE | MAE: 328.7 | RMSE: 1,052.2"
              icon={Award}
              color="emerald"
              badge="Rank #1"
              badgeColor="emerald"
            />
            <KpiCard
              title="Machine Learning Runner-Up"
              value="LightGBM"
              subtitle="12.27% WAPE | MAE: 347.3 | RMSE: 1,085.6"
              icon={BarChart3}
              color="blue"
              badge="Rank #2"
              badgeColor="blue"
            />
            <KpiCard
              title="Seasonal Baseline"
              value="Seasonal Naive"
              subtitle="16.17% WAPE (52-week lag baseline)"
              icon={Clock}
              color="amber"
              badge="Rank #3"
              badgeColor="amber"
            />
          </div>

          {/* Model Integrity & Objective Disclaimer */}
          <div className="bg-amber-50/70 border border-amber-200 rounded-xl p-4.5 flex items-start space-x-3.5">
            <Info className="w-5 h-5 text-amber-700 shrink-0 mt-0.5" />
            <div className="space-y-1 text-xs text-amber-900 leading-relaxed">
              <span className="font-bold text-sm block">
                Evaluation Objective &amp; Model Selection Transparency
              </span>
              <p>
                In retail forecasting, <strong>lower error metrics indicate better empirical precision</strong>, but business suitability depends on the specific operational objective. <strong>Naive Baseline is the verified Rank #1 model</strong> across global aggregate volume (11.61% WAPE vs LightGBM's 12.27%). LightGBM should not be falsely presented as the overall lowest error model; however, LightGBM excels at modeling promotional lifts and high-velocity commercial categories.
              </p>
            </div>
          </div>

          {/* Global Leaderboard Table */}
          <DataTable
            title="Global Model Evaluation Matrix (13-Week Test Period)"
            columns={matrixColumns}
            data={globalMatrix}
            pageSize={5}
            exportFilename="freshbasket_model_evaluation_matrix.csv"
          />

          {/* Horizon Decay Curve: Week 1 to 13 Error Trajectory */}
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-2xs">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
              <div>
                <h3 className="text-base font-bold text-slate-900">
                  Horizon Error Decay: Week-by-Week WAPE % (Weeks 1 to 13)
                </h3>
                <p className="text-xs text-slate-500">
                  Observing error progression as the forecasting horizon extends forward from the origin
                </p>
              </div>
              <div className="flex items-center space-x-4 text-xs">
                <div className="flex items-center space-x-1.5">
                  <span className="w-3 h-0.5 bg-emerald-600 font-bold" />
                  <span className="text-slate-700 font-medium">Naive WAPE</span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <span className="w-3 h-0.5 bg-blue-600 font-bold" />
                  <span className="text-slate-700 font-medium">LightGBM WAPE</span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <span className="w-3 h-0.5 bg-amber-600 font-bold" />
                  <span className="text-slate-700 font-medium">Seasonal Naive WAPE</span>
                </div>
              </div>
            </div>

            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={horizonDecay} margin={{ top: 10, right: 20, left: 10, bottom: 25 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis
                    dataKey="horizon_week"
                    tick={{ fontSize: 11, fill: '#64748b' }}
                    tickFormatter={(val) => `Wk +${val}`}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: '#64748b' }}
                    domain={[5, 25]}
                    unit="%"
                  />
                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (!active || !payload?.length) return null;
                      const item = payload[0]?.payload;
                      return (
                        <div className="bg-slate-900 text-white rounded-lg p-3 text-xs shadow-xl border border-slate-800 space-y-1">
                          <div className="font-bold text-slate-300 border-b border-slate-700 pb-1">
                            Forecast Horizon: Week +{label}
                          </div>
                          <div className="flex justify-between gap-4 text-emerald-400 font-semibold">
                            <span>Naive WAPE:</span>
                            <span>{item.Naive_wape}%</span>
                          </div>
                          <div className="flex justify-between gap-4 text-blue-400 font-semibold">
                            <span>LightGBM WAPE:</span>
                            <span>{item.LightGBM_wape}%</span>
                          </div>
                          <div className="flex justify-between gap-4 text-amber-400">
                            <span>Seasonal Naive WAPE:</span>
                            <span>{item['Seasonal Naive_wape']}%</span>
                          </div>
                        </div>
                      );
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="Naive_wape"
                    stroke="#059669"
                    strokeWidth={2.5}
                    dot={{ r: 4, fill: '#059669' }}
                    name="Naive Baseline"
                  />
                  <Line
                    type="monotone"
                    dataKey="LightGBM_wape"
                    stroke="#2563eb"
                    strokeWidth={2}
                    dot={{ r: 3.5, fill: '#2563eb' }}
                    name="LightGBM"
                  />
                  <Line
                    type="monotone"
                    dataKey="Seasonal Naive_wape"
                    stroke="#d97706"
                    strokeWidth={1.5}
                    strokeDasharray="3 3"
                    dot={{ r: 2.5, fill: '#d97706' }}
                    name="Seasonal Naive"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Technical RMSLE Audit Callout */}
          <div className="bg-slate-900 text-slate-100 rounded-xl p-5 shadow-md">
            <div className="flex items-center space-x-2 text-brand-400 mb-2">
              <HelpCircle className="w-4 h-4" />
              <span className="text-xs font-bold uppercase tracking-wider">
                Technical Audit: Explanation of the RMSLE Metric Discrepancy
              </span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed mb-4">
              {rmsleAudit.explanation}
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-3 border-t border-slate-800 text-center">
              <div className="bg-slate-800/80 rounded-lg p-3">
                <div className="text-[11px] text-slate-400">Commercial Volume LightGBM RMSLE</div>
                <div className="text-lg font-bold text-emerald-400 mt-0.5">
                  {rmsleAudit.commercial_volume_rmsle?.lightgbm ?? '0.2318'}
                </div>
              </div>
              <div className="bg-slate-800/80 rounded-lg p-3">
                <div className="text-[11px] text-slate-400">Commercial Volume Naive RMSLE</div>
                <div className="text-lg font-bold text-slate-200 mt-0.5">
                  {rmsleAudit.commercial_volume_rmsle?.naive ?? '0.2583'}
                </div>
              </div>
              <div className="bg-slate-800/80 rounded-lg p-3">
                <div className="text-[11px] text-slate-400">Commercial Revenue Advantage</div>
                <div className="text-lg font-bold text-blue-400 mt-0.5">
                  +{rmsleAudit.commercial_volume_rmsle?.lgbm_advantage_pct ?? '10.3'}% Accuracy
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
