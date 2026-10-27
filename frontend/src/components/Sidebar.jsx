import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  TrendingUp,
  Sparkles,
  Package,
  Store,
  BarChart3,
  SlidersHorizontal,
  Boxes,
  AlertTriangle,
  Lightbulb,
  Bot,
  ChevronLeft,
  ChevronRight,
  X,
  Database,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { useFilters } from '../context/FilterContext';

const NAV_GROUPS = [
  {
    group: 'OVERVIEW',
    items: [
      { path: '/', label: 'Executive Overview', icon: LayoutDashboard },
    ],
  },
  {
    group: 'DEMAND ANALYTICS',
    items: [
      { path: '/sales', label: 'Sales Analysis', icon: TrendingUp },
      { path: '/forecast', label: '13-Week Forecast', icon: Sparkles },
      { path: '/products', label: 'Product Insights', icon: Package },
      { path: '/stores', label: 'Store Performance', icon: Store },
      { path: '/evaluation', label: 'Model Evaluation', icon: BarChart3 },
    ],
  },
  {
    group: 'PLANNING',
    items: [
      { path: '/simulator', label: 'What-If Simulator', icon: SlidersHorizontal },
      { path: '/inventory', label: 'Inventory Planning', icon: Boxes },
      { path: '/alerts', label: 'Inventory Alerts', icon: AlertTriangle },
    ],
  },
  {
    group: 'INTELLIGENCE',
    items: [
      { path: '/insights', label: 'Business Insights', icon: Lightbulb },
      { path: '/assistant', label: 'Ask FreshBasket AI', icon: Bot },
    ],
  },
];

export const Sidebar = ({ isOpen, onClose, isCollapsed = false, onToggleCollapse }) => {
  const { dataMode, apiOnline } = useFilters();

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-40 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 bg-white border-r border-slate-200 flex flex-col transition-all duration-300 ease-in-out ${
          isCollapsed ? 'w-20' : 'w-64'
        } ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}`}
        aria-label="Sidebar Navigation"
      >
        {/* Brand Header */}
        <div className="h-16 px-4 border-b border-slate-100 flex items-center justify-between shrink-0">
          <div className="flex items-center space-x-3 overflow-hidden">
            <div className="w-9 h-9 rounded-xl bg-brand-600 flex items-center justify-center text-white shrink-0 shadow-xs">
              <Boxes className="w-5 h-5" />
            </div>
            {!isCollapsed && (
              <div className="truncate">
                <span className="font-bold text-slate-900 text-sm tracking-tight flex items-center gap-1">
                  FreshBasket <span className="text-brand-600 font-extrabold">AI</span>
                </span>
                <p className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
                  Demand & Inventory
                </p>
              </div>
            )}
          </div>

          {/* Close button on mobile */}
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 lg:hidden"
            aria-label="Close navigation"
          >
            <X className="w-5 h-5" />
          </button>

          {/* Desktop collapse toggle */}
          {onToggleCollapse && (
            <button
              onClick={onToggleCollapse}
              className="hidden lg:flex p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
              title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
            </button>
          )}
        </div>

        {/* Navigation Groups */}
        <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
          {NAV_GROUPS.map((grp) => (
            <div key={grp.group} className="space-y-1">
              {!isCollapsed ? (
                <div className="px-2.5 pt-1 pb-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  {grp.group}
                </div>
              ) : (
                <div className="w-full border-t border-slate-100 my-2" />
              )}
              {grp.items.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    onClick={onClose}
                    title={isCollapsed ? item.label : undefined}
                    className={({ isActive }) =>
                      `group relative flex items-center ${
                        isCollapsed ? 'justify-center px-2 py-2.5' : 'space-x-3 px-3 py-2'
                      } rounded-lg text-xs font-medium transition-all ${
                        isActive
                          ? 'bg-brand-50 text-brand-700 font-semibold shadow-2xs border border-brand-200/60'
                          : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/70'
                      }`
                    }
                  >
                    <Icon className="w-4 h-4 shrink-0 transition-transform group-hover:scale-105" />
                    {!isCollapsed && <span className="truncate">{item.label}</span>}

                    {/* Tooltip for collapsed state */}
                    {isCollapsed && (
                      <span className="fixed left-20 ml-2 px-2.5 py-1 bg-slate-900 text-white text-xs font-medium rounded-md shadow-lg opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity z-50 whitespace-nowrap">
                        {item.label}
                      </span>
                    )}
                  </NavLink>
                );
              })}
            </div>
          ))}
        </div>

        {/* Footer / System Status */}
        <div className="p-3 border-t border-slate-100 bg-slate-50/60 shrink-0 space-y-2">
          {/* Data Mode Indicator — only displayed when custom CSV is uploaded */}
          {dataMode === 'CUSTOM' && (
            !isCollapsed ? (
              <div className="flex items-center justify-between px-2.5 py-1.5 bg-white rounded-lg border border-slate-200 text-[11px]">
                <span className="text-slate-500 font-medium">Data Source</span>
                <span className="px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-amber-50 text-amber-800 border border-amber-200">
                  Uploaded CSV
                </span>
              </div>
            ) : (
              <div
                className="flex justify-center py-1 text-slate-400 group relative"
                title="Data: Uploaded CSV"
              >
                <Database className="w-4 h-4 text-slate-500" />
              </div>
            )
          )}

          {/* API Health Pill — Only displays Online when verified */}
          {!isCollapsed ? (
            <div className="flex items-center justify-between px-2.5 py-1.5 bg-white rounded-lg border border-slate-200 text-[11px]">
              <span className="text-slate-500 font-medium">FastAPI Engine</span>
              <div className="flex items-center space-x-1.5">
                {apiOnline ? (
                  <>
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                    <span className="text-emerald-700 font-semibold text-[11px]">Connected</span>
                  </>
                ) : (
                  <>
                    <span className="w-2 h-2 rounded-full bg-red-500" />
                    <span className="text-red-700 font-semibold text-[11px]">Disconnected</span>
                  </>
                )}
              </div>
            </div>
          ) : (
            <div
              className="flex justify-center py-1"
              title={`FastAPI Engine: ${apiOnline ? 'Connected' : 'Disconnected'}`}
            >
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  apiOnline ? 'bg-emerald-500' : 'bg-red-500'
                }`}
              />
            </div>
          )}
        </div>
      </aside>
    </>
  );
};
