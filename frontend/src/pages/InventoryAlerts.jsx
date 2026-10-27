import React, { useState, useEffect } from 'react';
import {
  AlertTriangle,
  ShieldAlert,
  Store,
  Package,
  Download,
  FileText,
  Mail,
  Camera,
  History,
  CheckCircle2,
  ExternalLink,
  ChevronRight,
  Printer
} from 'lucide-react';
import { useFilters } from '../context/FilterContext';
import {
  getInventoryAlerts,
  downloadReplenishmentCSV,
  getExecutiveReport,
  getEmailDraft,
  captureAlertSnapshot,
  getAlertHistory,
} from '../services/api';
import { KpiCard } from '../components/KpiCard';
import { FilterBar } from '../components/FilterBar';
import { DataTable } from '../components/DataTable';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';

export const InventoryAlerts = () => {
  const { selectedStore, selectedFamily, selectedModel, dataMode } = useFilters();
  const [selectedRisk, setSelectedRisk] = useState('');
  const [activeTab, setActiveTab] = useState('triage'); // 'triage', 'executive_report', 'email_draft', 'history'
  const [alertsData, setAlertsData] = useState(null);
  const [executiveReportText, setExecutiveReportText] = useState('');
  const [emailDraftText, setEmailDraftText] = useState('');
  const [alertHistory, setAlertHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [snapshotMsg, setSnapshotMsg] = useState(null);

  const fetchAlerts = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await getInventoryAlerts({
        model: selectedModel || undefined,
        store_nbr: selectedStore || undefined,
        family: selectedFamily || undefined,
        risk_level: selectedRisk || undefined,
      });
      setAlertsData(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchExecutiveReport = async () => {
    try {
      const text = await getExecutiveReport({
        model: selectedModel || undefined,
        store_nbr: selectedStore || undefined,
        family: selectedFamily || undefined,
      });
      setExecutiveReportText(text);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchEmailDraft = async () => {
    try {
      const text = await getEmailDraft({
        model: selectedModel || undefined,
        store_nbr: selectedStore || undefined,
        family: selectedFamily || undefined,
      });
      setEmailDraftText(text);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchHistory = async () => {
    try {
      const hist = await getAlertHistory();
      setAlertHistory(hist.history || []);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchAlerts();
    if (activeTab === 'executive_report') fetchExecutiveReport();
    if (activeTab === 'email_draft') fetchEmailDraft();
    if (activeTab === 'history') fetchHistory();
  }, [selectedStore, selectedFamily, selectedModel, selectedRisk, activeTab, dataMode]);

  const handleCaptureSnapshot = async () => {
    try {
      setSnapshotMsg('Saving snapshot to persistent log...');
      const res = await captureAlertSnapshot(selectedModel);
      setSnapshotMsg(res.message);
      if (activeTab === 'history') fetchHistory();
      setTimeout(() => setSnapshotMsg(null), 5000);
    } catch (err) {
      setSnapshotMsg(`Error: ${err.message}`);
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

  const alertItems = alertsData?.prioritized_alerts || [];
  const storeSummary = alertsData?.store_summary || [];

  const alertColumns = [
    {
      key: 'stockout_risk',
      header: 'Risk Level',
      sortable: true,
      render: (val) => (
        <span
          className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold ${
            val === 'HIGH'
              ? 'bg-red-100 text-red-800 border border-red-300'
              : 'bg-amber-100 text-amber-800 border border-amber-300'
          }`}
        >
          {val === 'HIGH' ? '🔴 HIGH RISK' : '🟡 MEDIUM RISK'}
        </span>
      ),
    },
    {
      key: 'store_nbr',
      header: 'Store #',
      sortable: true,
      render: (val) => <span className="font-bold text-slate-900">Store {val}</span>,
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
      render: (val) => Number(val).toLocaleString(undefined, { maximumFractionDigits: 1 }),
    },
    {
      key: 'suggested_replenishment_qty',
      header: 'Replenishment Deficit',
      sortable: true,
      align: 'right',
      render: (val) => (
        <span className="font-bold text-red-600">
          +{Number(val).toLocaleString()} units
        </span>
      ),
    },
    {
      key: 'lead_time_days',
      header: 'Lead Time',
      sortable: true,
      align: 'right',
      render: (val) => `${val ?? 7} days`,
    },
    {
      key: 'primary_supplier',
      header: 'Supplier Contact',
      sortable: true,
      render: (val) => <span className="text-slate-600 truncate max-w-[140px] inline-block">{val ?? 'Standard'}</span>,
    },
  ];

  const storeSummaryColumns = [
    {
      key: 'store_nbr',
      header: 'Store Branch',
      sortable: true,
      render: (val) => <span className="font-bold text-slate-900">Store {val}</span>,
    },
    {
      key: 'high_risk_skus',
      header: 'High Risk SKUs',
      sortable: true,
      align: 'right',
      render: (val) => (
        <span className={`font-bold ${Number(val) > 0 ? 'text-red-600' : 'text-slate-400'}`}>
          {val}
        </span>
      ),
    },
    {
      key: 'medium_risk_skus',
      header: 'Medium Risk SKUs',
      sortable: true,
      align: 'right',
      render: (val) => (
        <span className={`font-semibold ${Number(val) > 0 ? 'text-amber-600' : 'text-slate-400'}`}>
          {val}
        </span>
      ),
    },
    {
      key: 'total_replenishment_units',
      header: 'Total Replenishment Units',
      sortable: true,
      align: 'right',
      render: (val) => (
        <span className="font-bold text-slate-900">
          {Number(val).toLocaleString()}
        </span>
      ),
    },
    {
      key: 'total_lead_time_demand',
      header: 'Total Lead Demand',
      sortable: true,
      align: 'right',
      render: (val) => Number(val).toLocaleString(undefined, { maximumFractionDigits: 0 }),
    },
  ];

  return (
    <div className="space-y-6">
      {/* Top Filter Bar */}
      <FilterBar
        showModel={true}
        showStore={true}
        showFamily={true}
        showRisk={true}
        selectedRisk={selectedRisk}
        onRiskChange={setSelectedRisk}
      />

      {/* Triage Navigation Tabs */}
      <div className="flex flex-wrap items-center justify-between border-b border-slate-200 gap-3 pb-3">
        <div className="flex items-center space-x-1.5 sm:space-x-2">
          <button
            onClick={() => setActiveTab('triage')}
            className={`px-3 py-2 text-xs font-bold rounded-lg transition-colors ${
              activeTab === 'triage'
                ? 'bg-brand-600 text-white shadow-2xs'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
            }`}
          >
            Alert Triage Center
          </button>
          <button
            onClick={() => {
              setActiveTab('executive_report');
              fetchExecutiveReport();
            }}
            className={`px-3 py-2 text-xs font-bold rounded-lg transition-colors flex items-center space-x-1.5 ${
              activeTab === 'executive_report'
                ? 'bg-brand-600 text-white shadow-2xs'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Executive Briefing</span>
          </button>
          <button
            onClick={() => {
              setActiveTab('email_draft');
              fetchEmailDraft();
            }}
            className={`px-3 py-2 text-xs font-bold rounded-lg transition-colors flex items-center space-x-1.5 ${
              activeTab === 'email_draft'
                ? 'bg-brand-600 text-white shadow-2xs'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
            }`}
          >
            <Mail className="w-3.5 h-3.5" />
            <span>Supplier Email Draft</span>
          </button>
          <button
            onClick={() => {
              setActiveTab('history');
              fetchHistory();
            }}
            className={`px-3 py-2 text-xs font-bold rounded-lg transition-colors flex items-center space-x-1.5 ${
              activeTab === 'history'
                ? 'bg-brand-600 text-white shadow-2xs'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
            }`}
          >
            <History className="w-3.5 h-3.5" />
            <span>Audit History</span>
          </button>
        </div>

        {/* Action Buttons: Snapshot + Downloads */}
        <div className="flex items-center space-x-2 shrink-0">
          <button
            onClick={handleCaptureSnapshot}
            className="inline-flex items-center space-x-1.5 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-800 text-xs font-semibold rounded-lg transition-colors border border-slate-200"
            title="Log audit snapshot of current risk state"
          >
            <Camera className="w-3.5 h-3.5 text-slate-600" />
            <span>Log Snapshot</span>
          </button>

          <button
            onClick={handleDownloadCsv}
            className="inline-flex items-center space-x-1.5 px-3 py-1.5 bg-brand-600 hover:bg-brand-700 text-white text-xs font-semibold rounded-lg transition-colors shadow-2xs"
            title="Download replenishment order CSV"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {snapshotMsg && (
        <div className="p-3 rounded-lg bg-blue-50 border border-blue-200 text-blue-800 text-xs flex items-center space-x-2">
          <CheckCircle2 className="w-4 h-4 text-blue-600 shrink-0" />
          <span>{snapshotMsg}</span>
        </div>
      )}

      {loading && !alertsData ? (
        <LoadingSpinner message="Scanning inventory positions for stockout risks..." />
      ) : error && !alertsData ? (
        <ErrorMessage message={error} onRetry={fetchAlerts} />
      ) : (
        <>
          {/* Top Triage KPIs */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <KpiCard
              title="High Risk Items"
              value={alertsData?.high_risk_count ?? 0}
              subtitle="Projected stockout before restock"
              icon={ShieldAlert}
              color="red"
              badge="Urgent Reorder"
              badgeColor="red"
            />
            <KpiCard
              title="Medium Risk Items"
              value={alertsData?.medium_risk_count ?? 0}
              subtitle="Safety buffer breached"
              icon={AlertTriangle}
              color="amber"
              badge="Watchlist"
              badgeColor="amber"
            />
            <KpiCard
              title="Stores Affected"
              value={`${alertsData?.stores_affected ?? 0} / 54`}
              subtitle="Branches with urgent replenishment needs"
              icon={Store}
              color="purple"
            />
            <KpiCard
              title="Product Families Affected"
              value={`${alertsData?.families_affected ?? 0} / 33`}
              subtitle="FMCG categories requiring PO issuance"
              icon={Package}
              color="blue"
            />
          </div>

          {/* TAB 1: Alert Triage Center */}
          {activeTab === 'triage' && (
            <div className="space-y-6">
              {/* Prioritized Alert Feed */}
              <DataTable
                title="Priority Stockout Alert Feed (High & Medium Risk)"
                columns={alertColumns}
                data={alertItems}
                searchKey="family"
                searchPlaceholder="Search alert items by product family..."
                pageSize={12}
                onExportCsv={handleDownloadCsv}
                exportFilename="freshbasket_stockout_alerts.csv"
              />

              {/* Store-Wise Risk Rollup */}
              <DataTable
                title="Store-Level Risk Rollup (54 Branches)"
                columns={storeSummaryColumns}
                data={storeSummary}
                searchKey="store_nbr"
                searchPlaceholder="Search by store number..."
                pageSize={10}
                exportFilename="freshbasket_store_risk_exposure.csv"
              />
            </div>
          )}

          {/* TAB 2: Executive Briefing Markdown Document */}
          {activeTab === 'executive_report' && (
            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-2xs space-y-4">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <div>
                  <h3 className="text-base font-bold text-slate-900">
                    FreshBasket AI — Executive Inventory &amp; Stockout Risk Briefing
                  </h3>
                  <p className="text-xs text-slate-500">
                    Real-time generated markdown briefing generated by backend business logic
                  </p>
                </div>
                <button
                  onClick={() => {
                    const blob = new Blob([executiveReportText], { type: 'text/markdown' });
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.setAttribute('download', 'freshbasket_executive_briefing.md');
                    document.body.appendChild(link);
                    link.click();
                    link.remove();
                  }}
                  className="inline-flex items-center space-x-1.5 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold rounded-lg"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download .md</span>
                </button>
              </div>

              <div className="bg-slate-50 rounded-xl p-5 border border-slate-200 max-h-[600px] overflow-y-auto font-mono text-xs text-slate-800 leading-relaxed whitespace-pre-wrap">
                {executiveReportText || 'Loading executive briefing from backend...'}
              </div>
            </div>
          )}

          {/* TAB 3: Supplier Staging Email Draft */}
          {activeTab === 'email_draft' && (
            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-2xs space-y-4">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <div>
                  <h3 className="text-base font-bold text-slate-900">
                    Staging Procurement Email Draft
                  </h3>
                  <p className="text-xs text-slate-500">
                    Pre-composed communication for supply chain buyers and supplier dispatch
                  </p>
                </div>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(emailDraftText);
                    setSnapshotMsg('Email draft copied to clipboard!');
                    setTimeout(() => setSnapshotMsg(null), 3000);
                  }}
                  className="inline-flex items-center space-x-1.5 px-3 py-1.5 bg-brand-600 hover:bg-brand-700 text-white text-xs font-semibold rounded-lg"
                >
                  <Mail className="w-3.5 h-3.5" />
                  <span>Copy to Clipboard</span>
                </button>
              </div>

              <div className="bg-slate-50 rounded-xl p-5 border border-slate-200 max-h-[600px] overflow-y-auto font-mono text-xs text-slate-800 leading-relaxed whitespace-pre-wrap">
                {emailDraftText || 'Loading email draft from backend...'}
              </div>
            </div>
          )}

          {/* TAB 4: Audit History */}
          {activeTab === 'history' && (
            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-2xs space-y-4">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <div>
                  <h3 className="text-base font-bold text-slate-900">
                    Alert Audit History Log
                  </h3>
                  <p className="text-xs text-slate-500">
                    Persistent timestamps of recorded alert events (stored in <code>data/inventory/alert_history.csv</code>)
                  </p>
                </div>
                <span className="text-xs font-semibold text-slate-500">
                  Total snapshots: {alertHistory.length}
                </span>
              </div>

              {alertHistory.length === 0 ? (
                <div className="py-8 text-center text-xs text-slate-400">
                  No previous snapshots recorded. Click "Log Snapshot" to create the first entry.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left border-collapse">
                    <thead>
                      <tr className="bg-slate-50 border-b border-slate-200 font-bold text-slate-500 text-[11px] uppercase">
                        <th className="px-3 py-2.5">Snapshot Time (UTC)</th>
                        <th className="px-3 py-2.5">Store</th>
                        <th className="px-3 py-2.5">Family</th>
                        <th className="px-3 py-2.5">Risk</th>
                        <th className="px-3 py-2.5 text-right">Current Stock</th>
                        <th className="px-3 py-2.5 text-right">Replenishment Qty</th>
                        <th className="px-3 py-2.5">Data Mode</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 text-slate-700">
                      {alertHistory.slice(-50).reverse().map((row, idx) => (
                        <tr key={idx} className="hover:bg-slate-50">
                          <td className="px-3 py-2 font-mono text-[11px] text-slate-500">{row.snapshot_timestamp}</td>
                          <td className="px-3 py-2 font-semibold">Store {row.store_nbr}</td>
                          <td className="px-3 py-2">{row.family}</td>
                          <td className="px-3 py-2">
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                row.stockout_risk === 'HIGH'
                                  ? 'bg-red-100 text-red-800'
                                  : 'bg-amber-100 text-amber-800'
                              }`}
                            >
                              {row.stockout_risk}
                            </span>
                          </td>
                          <td className="px-3 py-2 text-right">{row.current_stock}</td>
                          <td className="px-3 py-2 text-right font-bold text-red-600">
                            {row.suggested_replenishment_qty}
                          </td>
                          <td className="px-3 py-2 text-slate-500 text-[11px]">{row.data_source_mode}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
};
