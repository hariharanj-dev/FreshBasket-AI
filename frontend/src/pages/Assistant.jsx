import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  Bot,
  User,
  Send,
  RotateCcw,
  Sparkles,
  HelpCircle,
  TrendingUp,
  AlertTriangle,
  BarChart3,
  Boxes,
  Database,
  Filter,
  Info,
  ChevronDown,
  ChevronUp,
  Table,
  CheckCircle2,
  Clock,
  ArrowRight,
  ShieldCheck,
  Zap,
} from 'lucide-react';
import { useFilters } from '../context/FilterContext';
import { queryAssistant } from '../services/api';
import { DataTable } from '../components/DataTable';

const SUGGESTED_QUERIES = [
  {
    category: 'Demand Leaders',
    icon: TrendingUp,
    badge: 'Demand',
    question: 'What are the top 5 product families by forecast demand?',
  },
  {
    category: 'Lowest Demand',
    icon: Boxes,
    badge: 'Demand',
    question: 'What are the lowest demand product families?',
  },
  {
    category: 'Model Benchmarking',
    icon: BarChart3,
    badge: 'Accuracy',
    question: 'Which forecasting model performs best by WAPE?',
  },
  {
    category: 'Detailed Metrics',
    icon: BarChart3,
    badge: 'Evaluation',
    question: 'Compare models using MAE, RMSE, WAPE, and RMSLE',
  },
  {
    category: 'Inventory Risks',
    icon: AlertTriangle,
    badge: 'Stockout',
    question: 'Which product families and stores have high-risk inventory items?',
  },
  {
    category: 'Replenishment',
    icon: Boxes,
    badge: 'Supply Chain',
    question: 'What are the total suggested replenishment quantities?',
  },
  {
    category: 'Demand Trajectories',
    icon: Zap,
    badge: 'Trends',
    question: 'Which product families show rising and declining demand trends?',
  },
  {
    category: 'Total Horizon',
    icon: Clock,
    badge: 'Forecast',
    question: 'What is the total forecast demand over the 13-week period?',
  },
  {
    category: 'Store Comparison',
    icon: Database,
    badge: 'Network',
    question: 'Compare store-level forecast demand across high-volume stores',
  },
  {
    category: 'Historical Baseline',
    icon: TrendingUp,
    badge: 'Sales',
    question: 'Compare historical sales trends for Grocery I and Beverages',
  },
];

export const Assistant = () => {
  const {
    selectedStore,
    setSelectedStore,
    selectedFamily,
    setSelectedFamily,
    selectedModel,
    setSelectedModel,
    filterOptions,
    dataMode,
  } = useFilters();

  const [inputQuestion, setInputQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [expandedDataIdx, setExpandedDataIdx] = useState(null);
  const [useActiveFilters, setUseActiveFilters] = useState(true);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async (questionText) => {
    const q = (questionText || inputQuestion).trim();
    if (!q || isLoading) return;

    // Build user message
    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputQuestion('');
    setIsLoading(true);

    try {
      const payload = {
        question: q,
        store_nbr: useActiveFilters && selectedStore ? Number(selectedStore) : null,
        family: useActiveFilters && selectedFamily ? selectedFamily : null,
        model: selectedModel || null,
      };

      const res = await queryAssistant(payload);

      const assistantMsg = {
        id: Date.now() + 1,
        sender: 'assistant',
        text: res.answer,
        intent: res.intent,
        supporting_data: res.supporting_data,
        data_source: res.data_source,
        applied_filters: res.applied_filters,
        limitations: res.limitations,
        success: res.success,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg = {
        id: Date.now() + 1,
        sender: 'assistant',
        text: `**Query Processing Error**\n\n${err.message || 'Unable to communicate with the FreshBasket AI backend.'}`,
        isError: true,
        success: false,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = () => {
    setMessages([]);
    setExpandedDataIdx(null);
    inputRef.current?.focus();
  };

  // Helper to format table columns from supporting_data
  const formatColumns = (dataArray) => {
    if (!dataArray || !dataArray.length) return [];
    return Object.keys(dataArray[0]).map((key) => ({
      key,
      label: key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
      render: (val) => {
        if (val === null || val === undefined) return '—';
        if (typeof val === 'number') {
          if (key.includes('pct') || key.includes('rate') || key.includes('wape')) {
            return `${val.toFixed(2)}%`;
          }
          if (Number.isInteger(val)) return val.toLocaleString();
          return val.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 2 });
        }
        return String(val);
      },
    }));
  };

  return (
    <div className="space-y-6">
      {/* Top Header Card */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-600 text-white flex items-center justify-center shadow-sm shadow-brand-500/20">
              <Bot className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
                Ask FreshBasket AI
                <span className="inline-flex items-center px-2 py-0.5 text-xs font-semibold rounded-full bg-brand-50 text-brand-700 border border-brand-200/60">
                  Ground-Truth AI
                </span>
              </h1>
              <p className="text-xs text-slate-500 mt-0.5">
                Deterministic business intelligence answering from verified sales, 13-week forecasts, model metrics, and inventory data.
              </p>
            </div>
          </div>
        </div>

        {/* Controls & Mode Indicator */}
        <div className="flex flex-wrap items-center gap-2.5">
          <label className="inline-flex items-center gap-2 text-xs font-medium text-slate-700 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg cursor-pointer hover:bg-slate-100 transition-colors">
            <input
              type="checkbox"
              checked={useActiveFilters}
              onChange={(e) => setUseActiveFilters(e.target.checked)}
              className="w-3.5 h-3.5 text-brand-600 rounded border-slate-300 focus:ring-brand-500"
            />
            <Filter className="w-3.5 h-3.5 text-slate-500" />
            <span>Apply Active Context ({selectedStore ? `Store ${selectedStore}` : 'All Stores'}, {selectedFamily || 'All Families'})</span>
          </label>

          {messages.length > 0 && (
            <button
              onClick={handleClear}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-600 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 hover:text-slate-800 transition-colors shadow-2xs"
              title="Clear conversation history"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Clear Chat</span>
            </button>
          )}
        </div>
      </div>

      {/* Suggested Questions Pill Carousel */}
      {messages.length === 0 && (
        <div className="bg-gradient-to-br from-white to-slate-50 rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-brand-600" />
              Supported Business Questions
            </h2>
            <span className="text-[11px] text-slate-400">
              Click any question to query the live engine
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
            {SUGGESTED_QUERIES.map((sq, i) => {
              const Icon = sq.icon;
              return (
                <button
                  key={i}
                  onClick={() => handleSend(sq.question)}
                  className="flex items-start gap-2.5 p-3 text-left rounded-lg border border-slate-200 bg-white hover:border-brand-300 hover:shadow-xs transition-all group"
                >
                  <div className="w-7 h-7 rounded-md bg-slate-50 text-slate-500 flex items-center justify-center shrink-0 group-hover:bg-brand-50 group-hover:text-brand-600 transition-colors mt-0.5">
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5 mb-0.5">
                      <span className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 group-hover:bg-brand-100 group-hover:text-brand-700 transition-colors">
                        {sq.badge}
                      </span>
                    </div>
                    <p className="text-xs font-medium text-slate-800 line-clamp-2 group-hover:text-brand-900 transition-colors">
                      {sq.question}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Verification Callout */}
          <div className="mt-3 pt-3 border-t border-slate-200/60 flex items-center justify-between text-[11px] text-slate-500">
            <div className="flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
              <span>Grounded in verified historical sales (2013–2017) & 13-week forecast horizon ({filterOptions?.date_range?.forecast_start} to {filterOptions?.date_range?.forecast_end}).</span>
            </div>
            <span className="font-semibold text-slate-600">Mode: {dataMode}</span>
          </div>
        </div>
      )}

      {/* Chat Messages Container */}
      <div className="space-y-4 min-h-[380px] max-h-[620px] overflow-y-auto px-1 py-2">
        {messages.map((msg, idx) => {
          const isUser = msg.sender === 'user';
          const hasSupportingData = Array.isArray(msg.supporting_data) && msg.supporting_data.length > 0;
          const isExpanded = expandedDataIdx === idx;

          return (
            <div
              key={msg.id || idx}
              className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}
            >
              {/* Avatar for assistant */}
              {!isUser && (
                <div className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 shadow-2xs ${
                  msg.isError
                    ? 'bg-rose-100 text-rose-600'
                    : 'bg-gradient-to-tr from-brand-600 to-indigo-600 text-white'
                }`}>
                  <Bot className="w-4 h-4" />
                </div>
              )}

              {/* Message Bubble Card */}
              <div
                className={`max-w-3xl rounded-2xl p-4 transition-all shadow-sm ${
                  isUser
                    ? 'bg-brand-600 text-white rounded-tr-xs'
                    : msg.isError
                    ? 'bg-rose-50 border border-rose-200 text-rose-900 rounded-tl-xs'
                    : 'bg-white border border-slate-200 text-slate-800 rounded-tl-xs'
                }`}
              >
                {/* Meta Header for Assistant */}
                {!isUser && (
                  <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 mb-2.5 border-b border-slate-100 text-xs">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-slate-900">FreshBasket AI</span>
                      {msg.intent && (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-slate-100 text-slate-600">
                          intent: {msg.intent}
                        </span>
                      )}
                      {msg.data_source && (
                        <span className="hidden sm:inline-flex items-center gap-1 text-[11px] text-slate-500">
                          <Database className="w-3 h-3 text-slate-400" />
                          {msg.data_source}
                        </span>
                      )}
                    </div>
                    <span className="text-[10px] text-slate-400">{msg.timestamp}</span>
                  </div>
                )}

                {/* Main Content */}
                {isUser ? (
                  <div className="text-sm font-normal leading-relaxed whitespace-pre-wrap">
                    {msg.text}
                  </div>
                ) : (
                  <div className="prose prose-sm max-w-none text-slate-800 space-y-2 text-sm leading-relaxed">
                    <ReactMarkdown
                      components={{
                        h3: ({ node, ...props }) => (
                          <h3 className="font-bold text-slate-900 text-base mt-3 mb-1.5 border-b border-slate-100 pb-1" {...props} />
                        ),
                        p: ({ node, ...props }) => <p className="mb-2 leading-relaxed text-slate-700" {...props} />,
                        ul: ({ node, ...props }) => <ul className="list-disc list-inside space-y-1 mb-2 text-slate-700" {...props} />,
                        ol: ({ node, ...props }) => <ol className="list-decimal list-inside space-y-1 mb-2 text-slate-700" {...props} />,
                        li: ({ node, ...props }) => <li className="text-slate-700 text-xs sm:text-sm" {...props} />,
                        strong: ({ node, ...props }) => <strong className="font-semibold text-slate-900" {...props} />,
                        code: ({ node, ...props }) => (
                          <code className="bg-slate-100 text-brand-700 px-1.5 py-0.5 rounded text-xs font-mono" {...props} />
                        ),
                        table: ({ node, ...props }) => (
                          <div className="overflow-x-auto my-3 border border-slate-200 rounded-lg">
                            <table className="min-w-full text-xs text-left border-collapse" {...props} />
                          </div>
                        ),
                        th: ({ node, ...props }) => (
                          <th className="bg-slate-50 px-3 py-2 font-semibold text-slate-700 border-b border-slate-200" {...props} />
                        ),
                        td: ({ node, ...props }) => (
                          <td className="px-3 py-2 border-b border-slate-100 text-slate-600" {...props} />
                        ),
                      }}
                    >
                      {msg.text}
                    </ReactMarkdown>
                  </div>
                )}

                {/* Supporting Data Drawer */}
                {hasSupportingData && (
                  <div className="mt-3 pt-3 border-t border-slate-100">
                    <button
                      onClick={() => setExpandedDataIdx(isExpanded ? null : idx)}
                      className="w-full flex items-center justify-between text-xs font-semibold text-brand-700 hover:text-brand-800 bg-brand-50/60 hover:bg-brand-50 p-2.5 rounded-lg border border-brand-200/60 transition-colors"
                    >
                      <span className="flex items-center gap-1.5">
                        <Table className="w-3.5 h-3.5" />
                        Supporting Dataset ({msg.supporting_data.length} records)
                      </span>
                      {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>

                    {isExpanded && (
                      <div className="mt-2.5 bg-slate-50 p-2 rounded-lg border border-slate-200 overflow-x-auto">
                        <DataTable
                          columns={formatColumns(msg.supporting_data)}
                          data={msg.supporting_data}
                          pageSize={5}
                          searchPlaceholder="Search supporting records..."
                        />
                      </div>
                    )}
                  </div>
                )}

                {/* Applied Filters & Limitations Footer */}
                {!isUser && !msg.isError && (
                  <div className="mt-3 pt-2.5 border-t border-slate-100 space-y-1.5">
                    {/* Applied Filters Tags */}
                    {msg.applied_filters && (
                      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-slate-500">
                        <span className="font-semibold text-slate-600">Filters:</span>
                        <span className="bg-slate-100 px-2 py-0.5 rounded text-slate-600 font-mono text-[10px]">
                          Store: {msg.applied_filters.store_nbr ?? 'All (54)'}
                        </span>
                        <span className="bg-slate-100 px-2 py-0.5 rounded text-slate-600 font-mono text-[10px]">
                          Family: {msg.applied_filters.family ?? 'All (33)'}
                        </span>
                        <span className="bg-slate-100 px-2 py-0.5 rounded text-slate-600 font-mono text-[10px]">
                          Model: {msg.applied_filters.model ?? 'naive_forecast'}
                        </span>
                      </div>
                    )}

                    {/* Limitations Notice */}
                    {msg.limitations && (
                      <div className="text-[11px] text-amber-800 bg-amber-50/70 border border-amber-200/80 rounded-md p-2 flex items-start gap-1.5">
                        <Info className="w-3.5 h-3.5 text-amber-600 shrink-0 mt-0.5" />
                        <span className="leading-tight">{msg.limitations}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* User Avatar */}
              {isUser && (
                <div className="w-8 h-8 rounded-xl bg-slate-200 text-slate-600 flex items-center justify-center shrink-0">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          );
        })}

        {/* Loading Indicator */}
        {isLoading && (
          <div className="flex gap-3 justify-start">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-600 text-white flex items-center justify-center shrink-0 shadow-2xs">
              <Bot className="w-4 h-4" />
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-xs p-4 shadow-sm flex items-center gap-3">
              <div className="flex space-x-1.5">
                <div className="w-2 h-2 bg-brand-600 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                <div className="w-2 h-2 bg-brand-600 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                <div className="w-2 h-2 bg-brand-600 rounded-full animate-bounce"></div>
              </div>
              <span className="text-xs font-medium text-slate-500">
                Auditing verified forecasting & inventory data...
              </span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Query Pill Strip when conversation is active */}
      {messages.length > 0 && (
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs no-scrollbar">
          <span className="text-[11px] font-semibold text-slate-400 shrink-0 uppercase tracking-wider pl-1">
            Suggested:
          </span>
          {SUGGESTED_QUERIES.slice(0, 5).map((sq, i) => (
            <button
              key={i}
              onClick={() => handleSend(sq.question)}
              disabled={isLoading}
              className="px-2.5 py-1 rounded-full bg-white border border-slate-200 text-slate-600 hover:text-brand-600 hover:border-brand-300 text-xs whitespace-nowrap shadow-2xs transition-colors shrink-0 disabled:opacity-50"
            >
              {sq.question}
            </button>
          ))}
        </div>
      )}

      {/* Chat Input Bar */}
      <div className="bg-white rounded-xl border border-slate-200 p-3 shadow-sm">
        <div className="flex items-end gap-2">
          <textarea
            ref={inputRef}
            rows={2}
            value={inputQuestion}
            onChange={(e) => setInputQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            placeholder="Ask a question (e.g. 'What are the top 5 product families?', 'Best model by WAPE', 'Which items are at risk?')..."
            className="flex-1 resize-none bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:bg-white focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-colors disabled:opacity-50"
          />

          <button
            onClick={() => handleSend()}
            disabled={isLoading || !inputQuestion.trim()}
            className="h-10 px-4 rounded-lg bg-brand-600 hover:bg-brand-700 text-white font-medium text-sm flex items-center justify-center gap-1.5 shadow-sm shadow-brand-500/20 transition-colors disabled:opacity-50 disabled:pointer-events-none"
          >
            <span>Ask</span>
            <Send className="w-4 h-4" />
          </button>
        </div>

        <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-100 text-[11px] text-slate-400">
          <span>Press <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded text-[10px] text-slate-600 font-mono">Enter</kbd> to send, <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded text-[10px] text-slate-600 font-mono">Shift+Enter</kbd> for newline</span>
          <span className="flex items-center gap-1">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
            Deterministic logic — zero hallucinations
          </span>
        </div>
      </div>
    </div>
  );
};
