import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

export const ErrorMessage = ({
  title = 'Failed to load data',
  message,
  onRetry,
}) => {
  return (
    <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-red-900 my-6 shadow-2xs">
      <div className="flex items-start space-x-3.5">
        <div className="p-2 bg-red-100 rounded-lg text-red-600 shrink-0">
          <AlertCircle className="w-5 h-5" />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-bold text-red-900">{title}</h3>
          <p className="text-xs text-red-700 mt-1 leading-relaxed break-words">
            {message || 'Unable to communicate with the FastAPI backend. Make sure the backend server is running at http://localhost:8000.'}
          </p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-3.5 inline-flex items-center space-x-1.5 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-xs font-semibold rounded-lg transition-colors shadow-2xs"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry Request</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
