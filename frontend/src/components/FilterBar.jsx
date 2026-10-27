import React from 'react';
import { Filter, X, ChevronDown, Calendar } from 'lucide-react';
import { useFilters } from '../context/FilterContext';

export const FilterBar = ({
  showModel = true,
  showStore = true,
  showFamily = true,
  showRisk = false,
  selectedRisk = '',
  onRiskChange = () => {},
  startDate = '',
  endDate = '',
  onDateChange = () => {},
  showDateRange = false,
  extraControls = null,
}) => {
  const {
    filterOptions,
    selectedStore,
    setSelectedStore,
    selectedFamily,
    setSelectedFamily,
    selectedModel,
    setSelectedModel,
    resetFilters,
  } = useFilters();

  const hasActiveFilters =
    Boolean(selectedStore) ||
    Boolean(selectedFamily) ||
    (selectedModel !== 'naive_forecast' && showModel) ||
    Boolean(selectedRisk) ||
    Boolean(startDate) ||
    Boolean(endDate);

  const activeCount = [
    Boolean(selectedStore),
    Boolean(selectedFamily),
    selectedModel !== 'naive_forecast' && showModel,
    Boolean(selectedRisk),
    Boolean(startDate || endDate),
  ].filter(Boolean).length;

  return (
    <div className="bg-white rounded-xl border border-slate-200/90 p-3 sm:p-3.5 mb-6 shadow-2xs">
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* Controls Grid */}
        <div className="flex flex-wrap items-center gap-2.5 flex-1 min-w-0">
          <div className="flex items-center space-x-1.5 text-xs font-semibold text-slate-500 uppercase tracking-wider pr-1">
            <Filter className="w-3.5 h-3.5 text-brand-600 shrink-0" />
            <span>Filters</span>
            {activeCount > 0 && (
              <span className="w-5 h-5 rounded-full bg-brand-100 text-brand-800 text-[10px] font-bold flex items-center justify-center">
                {activeCount}
              </span>
            )}
          </div>

          {/* Store Selector */}
          {showStore && (
            <div className="relative min-w-[170px] flex-1 sm:flex-initial">
              <label htmlFor="store-filter" className="sr-only">Store Filter</label>
              <select
                id="store-filter"
                value={selectedStore}
                onChange={(e) => setSelectedStore(e.target.value ? Number(e.target.value) : '')}
                className="w-full appearance-none bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 pr-8 text-xs font-medium text-slate-800 hover:bg-slate-100/80 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-colors cursor-pointer"
              >
                <option value="">All Stores (54)</option>
                {filterOptions.stores.map((s) => (
                  <option key={s.store_nbr} value={s.store_nbr}>
                    Store {s.store_nbr} — {s.city} ({s.type})
                  </option>
                ))}
              </select>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
            </div>
          )}

          {/* Family Selector */}
          {showFamily && (
            <div className="relative min-w-[190px] flex-1 sm:flex-initial">
              <label htmlFor="family-filter" className="sr-only">Family Filter</label>
              <select
                id="family-filter"
                value={selectedFamily}
                onChange={(e) => setSelectedFamily(e.target.value)}
                className="w-full appearance-none bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 pr-8 text-xs font-medium text-slate-800 hover:bg-slate-100/80 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-colors cursor-pointer"
              >
                <option value="">All Product Families (33)</option>
                {filterOptions.families.map((fam) => (
                  <option key={fam} value={fam}>
                    {fam}
                  </option>
                ))}
              </select>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
            </div>
          )}

          {/* Model Selector */}
          {showModel && (
            <div className="relative min-w-[210px] flex-1 sm:flex-initial">
              <label htmlFor="model-filter" className="sr-only">Model Filter</label>
              <select
                id="model-filter"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full appearance-none bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 pr-8 text-xs font-medium text-slate-800 hover:bg-slate-100/80 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-colors cursor-pointer"
              >
                {filterOptions.models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label} ({m.wape}% WAPE){m.is_best ? ' ★ Best' : ''}
                  </option>
                ))}
              </select>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
            </div>
          )}

          {/* Optional Risk Tier Filter */}
          {showRisk && (
            <div className="relative min-w-[150px] flex-1 sm:flex-initial">
              <label htmlFor="risk-filter" className="sr-only">Risk Tier Filter</label>
              <select
                id="risk-filter"
                value={selectedRisk}
                onChange={(e) => onRiskChange(e.target.value)}
                className="w-full appearance-none bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 pr-8 text-xs font-medium text-slate-800 hover:bg-slate-100/80 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-colors cursor-pointer"
              >
                <option value="">All Risk Tiers</option>
                <option value="HIGH">High Risk (Critical)</option>
                <option value="MEDIUM">Medium Risk (Warning)</option>
                <option value="LOW">Low Risk (Healthy)</option>
              </select>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
            </div>
          )}

          {/* Optional Date Range Filter */}
          {showDateRange && (
            <div className="flex items-center space-x-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg px-2 py-1">
              <Calendar className="w-3.5 h-3.5 text-slate-400 shrink-0" />
              <input
                type="date"
                value={startDate}
                min={filterOptions.date_range?.min_date}
                max={filterOptions.date_range?.max_date}
                onChange={(e) => onDateChange({ start: e.target.value, end: endDate })}
                className="bg-transparent text-xs text-slate-800 focus:outline-none cursor-pointer"
                aria-label="Start Date"
              />
              <span className="text-slate-400 text-xs">—</span>
              <input
                type="date"
                value={endDate}
                min={filterOptions.date_range?.min_date}
                max={filterOptions.date_range?.max_date}
                onChange={(e) => onDateChange({ start: startDate, end: e.target.value })}
                className="bg-transparent text-xs text-slate-800 focus:outline-none cursor-pointer"
                aria-label="End Date"
              />
            </div>
          )}

          {extraControls}
        </div>

        {/* Clear Filters Button */}
        {hasActiveFilters && (
          <button
            onClick={() => {
              resetFilters();
              if (showRisk) onRiskChange('');
              if (showDateRange) onDateChange({ start: '', end: '' });
            }}
            className="inline-flex items-center space-x-1.5 text-xs font-semibold text-slate-500 hover:text-red-600 px-2.5 py-1.5 rounded-lg border border-slate-200 hover:border-red-200 hover:bg-red-50/60 transition-colors"
          >
            <X className="w-3.5 h-3.5" />
            <span>Reset Filters</span>
          </button>
        )}
      </div>
    </div>
  );
};
