import React, { useState, useMemo } from 'react';
import { Search, ChevronLeft, ChevronRight, ArrowUpDown, ArrowUp, ArrowDown, Download, X } from 'lucide-react';

export const DataTable = ({
  columns,
  data = [],
  searchKey = '',
  searchPlaceholder = 'Search records...',
  pageSize = 10,
  onExportCsv,
  exportFilename = 'export.csv',
  title,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortColumn, setSortColumn] = useState(null);
  const [sortDirection, setSortDirection] = useState('asc'); // 'asc' or 'desc'
  const [currentPage, setCurrentPage] = useState(1);

  // Filter by search
  const filteredData = useMemo(() => {
    if (!searchTerm.trim()) return data;
    const lower = searchTerm.toLowerCase();
    return data.filter((row) => {
      if (searchKey && row[searchKey] !== undefined) {
        return String(row[searchKey]).toLowerCase().includes(lower);
      }
      return Object.values(row).some((val) =>
        String(val ?? '').toLowerCase().includes(lower)
      );
    });
  }, [data, searchTerm, searchKey]);

  // Sort
  const sortedData = useMemo(() => {
    if (!sortColumn) return filteredData;
    return [...filteredData].sort((a, b) => {
      const aVal = a[sortColumn];
      const bVal = b[sortColumn];

      if (aVal === bVal) return 0;
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;

      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
      }
      return sortDirection === 'asc'
        ? String(aVal).localeCompare(String(bVal))
        : String(bVal).localeCompare(String(aVal));
    });
  }, [filteredData, sortColumn, sortDirection]);

  // Pagination
  const totalPages = Math.ceil(sortedData.length / pageSize) || 1;
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return sortedData.slice(start, start + pageSize);
  }, [sortedData, currentPage, pageSize]);

  const handleSort = (colKey) => {
    if (sortColumn === colKey) {
      if (sortDirection === 'asc') {
        setSortDirection('desc');
      } else {
        setSortColumn(null);
        setSortDirection('asc');
      }
    } else {
      setSortColumn(colKey);
      setSortDirection('asc');
    }
  };

  const handleLocalCsvExport = () => {
    if (onExportCsv) {
      onExportCsv();
      return;
    }
    if (!data.length) return;
    const headers = columns.map((c) => c.header || c.key).join(',');
    const rows = sortedData.map((row) =>
      columns
        .map((c) => {
          const val = row[c.key];
          const text = val === null || val === undefined ? '' : String(val);
          return text.includes(',') ? `"${text.replace(/"/g, '""')}"` : text;
        })
        .join(',')
    );
    const csvContent = [headers, ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', exportFilename);
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200/90 overflow-hidden shadow-2xs">
      {/* Header Controls */}
      <div className="px-4 sm:px-5 py-3 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3 bg-white">
        <div className="flex items-center space-x-2">
          {title && <h4 className="text-sm font-bold text-slate-800 tracking-tight">{title}</h4>}
          <span className="text-xs text-slate-400 font-medium tabular-nums">
            ({sortedData.length} {sortedData.length === 1 ? 'record' : 'records'})
          </span>
        </div>

        <div className="flex items-center space-x-2.5">
          {/* Search Box */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              placeholder={searchPlaceholder}
              className="pl-8 pr-7 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-700 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 w-44 sm:w-56 transition-all"
            />
            {searchTerm && (
              <button
                onClick={() => {
                  setSearchTerm('');
                  setCurrentPage(1);
                }}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 p-0.5"
                title="Clear search"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>

          {/* Export CSV button */}
          <button
            onClick={handleLocalCsvExport}
            className="inline-flex items-center space-x-1.5 px-3 py-1.5 bg-white hover:bg-slate-50 border border-slate-200 hover:border-slate-300 text-slate-700 text-xs font-semibold rounded-lg transition-colors shadow-2xs"
            title="Export displayed records to CSV"
          >
            <Download className="w-3.5 h-3.5 text-slate-500" />
            <span className="hidden sm:inline">Export CSV</span>
          </button>
        </div>
      </div>

      {/* Table View */}
      <div className="overflow-x-auto max-h-[580px] overflow-y-auto">
        <table className="w-full text-left border-collapse">
          <thead className="sticky top-0 z-10 bg-slate-50/95 backdrop-blur-sm border-b border-slate-200">
            <tr className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
              {columns.map((col) => {
                const isSorted = sortColumn === col.key;
                return (
                  <th
                    key={col.key}
                    onClick={() => col.sortable !== false && handleSort(col.key)}
                    className={`px-4 py-3 select-none ${col.className || ''} ${
                      col.sortable !== false
                        ? 'cursor-pointer hover:bg-slate-100 hover:text-slate-800 transition-colors'
                        : ''
                    }`}
                  >
                    <div className={`flex items-center space-x-1.5 ${col.align === 'right' ? 'justify-end' : ''}`}>
                      <span>{col.header}</span>
                      {col.sortable !== false && (
                        <span className="text-slate-400">
                          {isSorted ? (
                            sortDirection === 'asc' ? (
                              <ArrowUp className="w-3.5 h-3.5 text-brand-600" />
                            ) : (
                              <ArrowDown className="w-3.5 h-3.5 text-brand-600" />
                            )
                          ) : (
                            <ArrowUpDown className="w-3 h-3 opacity-40 hover:opacity-100" />
                          )}
                        </span>
                      )}
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-xs text-slate-700">
            {paginatedData.length === 0 ? (
              <tr>
                <td
                  colSpan={columns.length}
                  className="px-4 py-12 text-center text-slate-400 font-medium"
                >
                  No matching records found.
                </td>
              </tr>
            ) : (
              paginatedData.map((row, idx) => (
                <tr
                  key={row.id || idx}
                  className="hover:bg-slate-50/90 transition-colors even:bg-slate-50/30"
                >
                  {columns.map((col) => (
                    <td
                      key={col.key}
                      className={`px-4 py-2.5 ${col.className || ''} ${
                        col.align === 'right' ? 'text-right tabular-nums' : ''
                      }`}
                    >
                      {col.render ? col.render(row[col.key], row) : (row[col.key] ?? '—')}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      {totalPages > 1 && (
        <div className="px-4 sm:px-5 py-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500 bg-white">
          <div className="tabular-nums">
            Showing {(currentPage - 1) * pageSize + 1} to{' '}
            {Math.min(currentPage * pageSize, sortedData.length)} of {sortedData.length} records
          </div>
          <div className="flex items-center space-x-1.5">
            <button
              onClick={() => setCurrentPage((p) => Math.max(p - 1, 1))}
              disabled={currentPage === 1}
              className="p-1.5 rounded-lg border border-slate-200 hover:bg-slate-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              aria-label="Previous Page"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="px-2.5 py-1 rounded bg-slate-50 border border-slate-200 font-semibold text-slate-700 tabular-nums text-xs">
              {currentPage} / {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(p + 1, totalPages))}
              disabled={currentPage === totalPages}
              className="p-1.5 rounded-lg border border-slate-200 hover:bg-slate-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              aria-label="Next Page"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
