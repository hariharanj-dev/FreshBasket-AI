import React from 'react';
import { PackageOpen, X } from 'lucide-react';

export const EmptyState = ({
  title = 'No records match your criteria',
  message = 'Try clearing or relaxing your active store, product family, or risk-tier filters.',
  onClearFilters,
}) => {
  return (
    <div className="bg-white rounded-xl border border-dashed border-slate-300 p-12 text-center my-6">
      <div className="w-12 h-12 rounded-2xl bg-slate-100 flex items-center justify-center text-slate-400 mx-auto mb-3">
        <PackageOpen className="w-6 h-6" />
      </div>
      <h3 className="text-sm font-bold text-slate-800">{title}</h3>
      <p className="text-xs text-slate-500 mt-1 max-w-md mx-auto">{message}</p>
      {onClearFilters && (
        <button
          onClick={onClearFilters}
          className="mt-4 inline-flex items-center space-x-1.5 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold rounded-lg transition-colors"
        >
          <X className="w-3.5 h-3.5" />
          <span>Clear Filters</span>
        </button>
      )}
    </div>
  );
};
