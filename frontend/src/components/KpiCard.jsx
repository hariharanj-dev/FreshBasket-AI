import React from 'react';

const COLOR_MAP = {
  emerald: {
    iconBg: 'bg-emerald-50 text-emerald-700 border-emerald-100',
    border: 'hover:border-emerald-300/80',
    badge: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  },
  blue: {
    iconBg: 'bg-blue-50 text-blue-700 border-blue-100',
    border: 'hover:border-blue-300/80',
    badge: 'bg-blue-50 text-blue-700 border-blue-200',
  },
  amber: {
    iconBg: 'bg-amber-50 text-amber-700 border-amber-100',
    border: 'hover:border-amber-300/80',
    badge: 'bg-amber-50 text-amber-700 border-amber-200',
  },
  red: {
    iconBg: 'bg-red-50 text-red-700 border-red-100',
    border: 'hover:border-red-300/80',
    badge: 'bg-red-50 text-red-700 border-red-200',
  },
  purple: {
    iconBg: 'bg-purple-50 text-purple-700 border-purple-100',
    border: 'hover:border-purple-300/80',
    badge: 'bg-purple-50 text-purple-700 border-purple-200',
  },
  slate: {
    iconBg: 'bg-slate-100 text-slate-700 border-slate-200',
    border: 'hover:border-slate-300',
    badge: 'bg-slate-100 text-slate-700 border-slate-200',
  },
};

export const KpiCard = ({
  title,
  value,
  subtitle,
  icon: Icon,
  color = 'emerald',
  badge = null,
  badgeColor = null,
  isLoading = false,
}) => {
  const scheme = COLOR_MAP[color] || COLOR_MAP.emerald;
  const badgeScheme = badgeColor ? (COLOR_MAP[badgeColor] || scheme).badge : scheme.badge;

  const displayValue = value !== undefined && value !== null && value !== '' ? value : '—';

  return (
    <div
      className={`bg-white rounded-xl border border-slate-200 p-4 sm:p-4.5 shadow-2xs hover:shadow-xs transition-all duration-200 flex flex-col justify-between min-h-[110px] ${scheme.border}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="space-y-1 min-w-0 flex-1">
          <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider truncate">
            {title}
          </p>

          <div className="flex flex-wrap items-baseline gap-2 pt-0.5">
            {isLoading ? (
              <div className="h-7 w-24 bg-slate-200 animate-pulse rounded" />
            ) : (
              <h3 className="text-2xl font-bold text-slate-900 tracking-tight tabular-nums font-mono">
                {displayValue}
              </h3>
            )}

            {badge && !isLoading && (
              <span
                className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold border ${badgeScheme}`}
              >
                {badge}
              </span>
            )}
          </div>
        </div>

        {Icon && (
          <div
            className={`w-9 h-9 rounded-lg border flex items-center justify-center shrink-0 transition-transform ${scheme.iconBg}`}
          >
            <Icon className="w-4 h-4" />
          </div>
        )}
      </div>

      {subtitle && (
        <p className="text-xs text-slate-500 pt-2 border-t border-slate-100/80 mt-2 truncate">
          {subtitle}
        </p>
      )}
    </div>
  );
};
