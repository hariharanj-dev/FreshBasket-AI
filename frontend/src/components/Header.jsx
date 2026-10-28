import React from 'react';
import { Menu, RotateCcw, Sparkles, ShieldCheck, Video, ExternalLink } from 'lucide-react';
import { useFilters } from '../context/FilterContext';

export const Header = ({ title, subtitle, onOpenSidebar, onRefresh, isRefreshing }) => {
  const { selectedModel, filterOptions } = useFilters();

  const currentModelObj = filterOptions.models.find((m) => m.id === selectedModel);

  return (
    <header className="sticky top-0 z-30 bg-white/90 backdrop-blur-md border-b border-slate-200/80 px-4 sm:px-6 lg:px-8 py-3 transition-colors">
      <div className="flex items-center justify-between gap-4">
        {/* Left: Mobile hamburger + Page Title & Subtitle */}
        <div className="flex items-center space-x-3 min-w-0">
          <button
            onClick={onOpenSidebar}
            className="p-1.5 -ml-1.5 rounded-lg text-slate-500 hover:text-slate-900 hover:bg-slate-100 lg:hidden transition-colors"
            aria-label="Open sidebar navigation"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div className="min-w-0 truncate">
            <h1 className="text-lg sm:text-xl font-bold text-slate-900 tracking-tight leading-tight truncate">
              {title}
            </h1>
            {subtitle && (
              <p className="text-xs text-slate-500 hidden sm:block truncate mt-0.5">
                {subtitle}
              </p>
            )}
          </div>
        </div>

        {/* Right: Active model pill + Refresh action */}
        <div className="flex items-center space-x-2 shrink-0">
          {/* Active Model Indicator */}
          <div className="hidden md:inline-flex items-center space-x-1.5 px-2.5 py-1.5 bg-slate-50/80 hover:bg-slate-100 rounded-lg border border-slate-200 text-xs transition-colors">
            <Sparkles className="w-3.5 h-3.5 text-brand-600" />
            <span className="text-slate-500 font-medium">Model:</span>
            <span className="font-semibold text-slate-800">
              {currentModelObj ? currentModelObj.label : 'Naive Baseline'}
            </span>
            {currentModelObj?.is_best && (
              <span className="inline-flex items-center gap-0.5 ml-1 px-1.5 py-0.5 bg-emerald-100 text-emerald-800 rounded font-bold text-[10px] tracking-wide">
                <ShieldCheck className="w-3 h-3 text-emerald-700" />
                11.61% WAPE
              </span>
            )}
          </div>

          {/* Video Presentation Link */}
          <a
            href="https://drive.google.com/file/d/1VA31He0BFewhpDQUotFOQDWqv4wuwfW2/view?usp=sharing"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center space-x-1.5 px-2.5 sm:px-3 py-1.5 bg-rose-50 hover:bg-rose-100 text-rose-700 text-xs font-semibold rounded-lg border border-rose-200 transition-colors shadow-2xs hover:border-rose-300"
            title="Watch 10-15 minute executive presentation video on Google Drive"
          >
            <Video className="w-3.5 h-3.5 text-rose-600" />
            <span className="hidden sm:inline">Video Defense</span>
            <ExternalLink className="w-3 h-3 text-rose-500 opacity-80" />
          </a>

          {/* Refresh Action */}
          {onRefresh && (
            <button
              onClick={onRefresh}
              disabled={isRefreshing}
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold rounded-lg border border-slate-200 transition-colors shadow-2xs hover:border-slate-300 disabled:opacity-60 focus:outline-none focus:ring-2 focus:ring-brand-500/20"
              title="Refresh page data"
              aria-label="Refresh page data"
            >
              <RotateCcw
                className={`w-3.5 h-3.5 text-slate-500 transition-transform ${
                  isRefreshing ? 'animate-spin text-brand-600' : ''
                }`}
              />
              <span className="hidden sm:inline">
                {isRefreshing ? 'Refreshing...' : 'Refresh'}
              </span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
