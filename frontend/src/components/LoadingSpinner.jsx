import React from 'react';
import { Loader2 } from 'lucide-react';

export const LoadingSpinner = ({ message = 'Loading FreshBasket AI data...' }) => {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-4">
      <div className="w-12 h-12 rounded-2xl bg-brand-50 flex items-center justify-center text-brand-600 mb-3 animate-pulse">
        <Loader2 className="w-6 h-6 animate-spin text-brand-600" />
      </div>
      <p className="text-sm font-semibold text-slate-700">{message}</p>
      <p className="text-xs text-slate-400 mt-1">Connecting to FastAPI analytics engine</p>
    </div>
  );
};
