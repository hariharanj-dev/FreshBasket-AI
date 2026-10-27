import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { FilterProvider } from './context/FilterContext';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { FloatingAssistant } from './components/FloatingAssistant';

// 8 Application Pages
import { Overview } from './pages/Overview';
import { SalesAnalysis } from './pages/SalesAnalysis';
import { Forecast } from './pages/Forecast';
import { BusinessInsights } from './pages/BusinessInsights';
import { ProductInsights } from './pages/ProductInsights';
import { StorePerformance } from './pages/StorePerformance';
import { ModelEvaluation } from './pages/ModelEvaluation';
import { InventoryPlanning } from './pages/InventoryPlanning';
import { InventoryAlerts } from './pages/InventoryAlerts';
import { Assistant } from './pages/Assistant';
import { ScenarioSimulator } from './pages/ScenarioSimulator';

const PAGE_METADATA = {
  '/': {
    title: 'Executive Demand & Forecast Overview',
    subtitle: 'Strategic intelligence, 13-week forecast horizon, and global accuracy benchmarks',
  },
  '/assistant': {
    title: 'Ask FreshBasket AI — Business Query Assistant',
    subtitle: 'Conversational demand analytics, model comparisons, risk auditing, and verified recommendations',
  },
  '/simulator': {
    title: 'What-If Scenario Simulator',
    subtitle: 'Simulate demand shocks, inventory deficits, and lead-time delays side-by-side with original plans',
  },
  '/sales': {
    title: 'Historical Sales Analysis',
    subtitle: 'Weekly retail sales trends, moving averages, and promotional volume dynamics',
  },
  '/forecast': {
    title: '13-Week Demand Forecasting',
    subtitle: 'Multi-step forward predictions across Naive, LightGBM, and Seasonal Naive models',
  },
  '/insights': {
    title: 'AI Business Insights & Demand Intelligence',
    subtitle: 'Statistical demand trends, forecast opportunities, model divergence, and inventory recommendations',
  },
  '/products': {
    title: 'Product Family Performance',
    subtitle: 'Category-level sales volume, demand distributions, and model accuracy comparisons',
  },
  '/stores': {
    title: 'Store Network Performance',
    subtitle: '54 retail branches across Ecuador: sales rankings, store formats (A–E), and accuracy',
  },
  '/evaluation': {
    title: 'Model Evaluation & Auditing',
    subtitle: 'WAPE, MAE, RMSE, RMSLE benchmarks, horizon error decay, and technical variance analysis',
  },
  '/inventory': {
    title: 'Inventory Planning & Stockout Risk',
    subtitle: 'Automated reorder point heuristics, lead-time demand, and replenishment recommendations',
  },
  '/alerts': {
    title: 'Inventory Alert Center & Reports',
    subtitle: 'High & medium stockout risks, store exposures, downloadable briefings, and audit snapshots',
  },
};

const AppLayout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const location = useLocation();

  const meta = PAGE_METADATA[location.pathname] || {
    title: 'FreshBasket AI',
    subtitle: 'FMCG Demand Forecasting & Inventory Intelligence',
  };

  const handleRefresh = () => {
    setIsRefreshing(true);
    setRefreshKey((k) => k + 1);
    setTimeout(() => setIsRefreshing(false), 600);
  };

  return (
    <div className="min-h-screen bg-slate-50 flex">
      {/* Sidebar navigation */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        isCollapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main content wrapper */}
      <div
        className={`flex-1 ${
          sidebarCollapsed ? 'lg:pl-20' : 'lg:pl-64'
        } flex flex-col min-w-0 transition-all duration-300 ease-in-out`}
      >
        <Header
          title={meta.title}
          subtitle={meta.subtitle}
          onOpenSidebar={() => setSidebarOpen(true)}
          onRefresh={handleRefresh}
          isRefreshing={isRefreshing}
        />

        <main className="flex-1 px-4 sm:px-6 lg:px-8 py-6 max-w-7xl w-full mx-auto">
          <Routes key={refreshKey}>
            <Route path="/" element={<Overview />} />
            <Route path="/assistant" element={<Assistant />} />
            <Route path="/simulator" element={<ScenarioSimulator />} />
            <Route path="/sales" element={<SalesAnalysis />} />
            <Route path="/forecast" element={<Forecast />} />
            <Route path="/insights" element={<BusinessInsights />} />
            <Route path="/products" element={<ProductInsights />} />
            <Route path="/stores" element={<StorePerformance />} />
            <Route path="/evaluation" element={<ModelEvaluation />} />
            <Route path="/inventory" element={<InventoryPlanning />} />
            <Route path="/alerts" element={<InventoryAlerts />} />
            <Route path="*" element={<Overview />} />
          </Routes>
        </main>
      </div>

      {/* Persistent Floating AI Assistant Widget */}
      <FloatingAssistant />
    </div>
  );
};

export default function App() {
  return (
    <FilterProvider>
      <BrowserRouter>
        <AppLayout />
      </BrowserRouter>
    </FilterProvider>
  );
}
