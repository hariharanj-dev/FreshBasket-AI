import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  Bot,
  Send,
  X,
  Minus,
  Sparkles,
  TrendingUp,
  AlertTriangle,
  Boxes,
  Lightbulb,
  RotateCcw,
  Clock,
  ShieldCheck,
  ChevronDown
} from 'lucide-react';
import { queryAssistant } from '../services/api';
import { useFilters } from '../context/FilterContext';
import './FloatingAssistant.css';

const INITIAL_WELCOME_TEXT = `Hi! 👋 Welcome to FreshBasket AI.

I'm your smart retail assistant. I can help you understand sales forecasts, inventory risks, replenishment recommendations, and business insights.

What would you like to know today?`;

const SUGGESTED_QUESTIONS = [
  {
    label: 'Analyze Sales Forecast',
    icon: TrendingUp,
    emoji: '📈',
    query: 'Analyze Sales Forecast',
    color: 'hover:border-emerald-300 hover:bg-emerald-50/60 text-emerald-950',
  },
  {
    label: 'Check Inventory Risks',
    icon: AlertTriangle,
    emoji: '📦',
    query: 'Check Inventory Risks',
    color: 'hover:border-amber-300 hover:bg-amber-50/60 text-amber-950',
  },
  {
    label: 'View Replenishment Needs',
    icon: Boxes,
    emoji: '🛒',
    query: 'View Replenishment Needs',
    color: 'hover:border-blue-300 hover:bg-blue-50/60 text-blue-950',
  },
  {
    label: 'Get Business Insights',
    icon: Lightbulb,
    emoji: '💡',
    query: 'Get Business Insights',
    color: 'hover:border-purple-300 hover:bg-purple-50/60 text-purple-950',
  },
];

const getFormattedTime = () =>
  new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

export const FloatingAssistant = () => {
  const { selectedStore, selectedFamily, selectedModel } = useFilters();

  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [hasInteracted, setHasInteracted] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);

  const [inputQuestion, setInputQuestion] = useState('');
  const [messages, setMessages] = useState(() => [
    {
      id: 'welcome-initial',
      sender: 'assistant',
      text: INITIAL_WELCOME_TEXT,
      isWelcome: true,
      timestamp: getFormattedTime(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const chatWindowRef = useRef(null);

  // Auto-scroll to bottom when new messages or loading state changes
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen && !isMinimized) {
      scrollToBottom();
    }
  }, [messages, isLoading, isOpen, isMinimized]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen && !isMinimized) {
      setTimeout(() => {
        inputRef.current?.focus();
      }, 150);
    }
  }, [isOpen, isMinimized]);

  const toggleOpen = () => {
    if (!isOpen) {
      setIsOpen(true);
      setIsMinimized(false);
      setHasInteracted(true);
    } else if (isMinimized) {
      setIsMinimized(false);
    } else {
      setIsOpen(false);
    }
  };

  const handleMinimize = (e) => {
    e.stopPropagation();
    setIsMinimized(true);
  };

  const handleClose = (e) => {
    e.stopPropagation();
    setIsOpen(false);
    setIsMinimized(false);
  };

  const handleResetChat = (e) => {
    e.stopPropagation();
    setMessages([
      {
        id: `welcome-${Date.now()}`,
        sender: 'assistant',
        text: INITIAL_WELCOME_TEXT,
        isWelcome: true,
        timestamp: getFormattedTime(),
      },
    ]);
    setInputQuestion('');
  };

  const handleSend = async (questionOverride) => {
    const q = (typeof questionOverride === 'string' ? questionOverride : inputQuestion).trim();
    if (!q || isLoading) return;

    const userMessage = {
      id: `usr-${Date.now()}`,
      sender: 'user',
      text: q,
      timestamp: getFormattedTime(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputQuestion('');
    setIsLoading(true);

    try {
      const payload = {
        question: q,
        store_nbr: selectedStore ? Number(selectedStore) : null,
        family: selectedFamily || null,
        model: selectedModel || 'naive_forecast',
      };

      const res = await queryAssistant(payload);

      const assistantMessage = {
        id: `ast-${Date.now()}`,
        sender: 'assistant',
        text: res.answer || 'No answer provided by backend.',
        intent: res.intent,
        supporting_data: res.supporting_data,
        data_source: res.data_source,
        limitations: res.limitations,
        success: res.success,
        timestamp: getFormattedTime(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage = {
        id: `err-${Date.now()}`,
        sender: 'assistant',
        text: `**Query Unavailable**\n\n${
          err.message || 'Unable to connect to the FreshBasket AI backend service. Please check your network or server connection.'
        }`,
        isError: true,
        timestamp: getFormattedTime(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setTimeout(() => inputRef.current?.focus(), 80);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <aside
      aria-label="FreshBasket AI Floating Assistant"
      className="fixed z-50 pointer-events-none bottom-5 right-5 sm:bottom-6 sm:right-6 font-sans antialiased"
    >
      {/* 1. EXPANDABLE CHAT WINDOW */}
      {isOpen && !isMinimized && (
        <div
          ref={chatWindowRef}
          role="dialog"
          aria-labelledby="fb-assistant-title"
          className="pointer-events-auto fb-window-anim mb-4 w-[calc(100vw-2.5rem)] sm:w-[390px] h-[560px] max-h-[82vh] flex flex-col bg-white rounded-2xl shadow-2xl shadow-slate-900/25 border border-slate-200/90 overflow-hidden"
        >
          {/* Header */}
          <div className="bg-gradient-to-r from-emerald-600 via-teal-600 to-emerald-700 px-4 py-3.5 text-white flex items-center justify-between shrink-0 shadow-sm">
            <div className="flex items-center space-x-3 min-w-0">
              <div className="relative">
                <div className="w-9 h-9 rounded-xl bg-white/15 backdrop-blur-md border border-white/25 flex items-center justify-center text-white shadow-inner">
                  <Bot className="w-5 h-5 text-white" />
                </div>
                <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-emerald-400 border-2 border-emerald-700 rounded-full" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-1.5">
                  <h2 id="fb-assistant-title" className="font-bold text-sm text-white tracking-tight leading-none truncate">
                    FreshBasket AI
                  </h2>
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/40 text-emerald-100 border border-emerald-300/20">
                    Pro
                  </span>
                </div>
                <div className="flex items-center gap-1.5 mt-1 text-[11px] text-emerald-100/90 leading-none">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-300 animate-pulse" />
                  <span className="truncate">Your Smart Retail Assistant</span>
                </div>
              </div>
            </div>

            {/* Header Action Buttons */}
            <div className="flex items-center space-x-1 shrink-0 text-white/90">
              <button
                type="button"
                onClick={handleResetChat}
                title="Restart Conversation"
                aria-label="Restart Conversation"
                className="p-1.5 rounded-lg hover:bg-white/15 hover:text-white transition-all text-emerald-100"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
              <button
                type="button"
                onClick={handleMinimize}
                title="Minimize Chat"
                aria-label="Minimize Chat"
                className="p-1.5 rounded-lg hover:bg-white/15 hover:text-white transition-all text-emerald-100"
              >
                <Minus className="w-4 h-4" />
              </button>
              <button
                type="button"
                onClick={handleClose}
                title="Close Chat"
                aria-label="Close Chat"
                className="p-1.5 rounded-lg hover:bg-white/15 hover:text-white transition-all text-emerald-100"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Active Filter Scope Pill (if user has filtered dashboard) */}
          {(selectedStore || selectedFamily) && (
            <div className="bg-emerald-50/80 border-b border-emerald-100/80 px-4 py-1.5 flex items-center justify-between text-[11px] text-emerald-800">
              <span className="flex items-center gap-1.5 truncate">
                <span className="font-semibold text-emerald-900">Active Scope:</span>
                {selectedStore ? `Store #${selectedStore}` : 'All Stores'}
                {selectedFamily ? ` • ${selectedFamily}` : ''}
              </span>
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600 shrink-0 ml-1" />
            </div>
          )}

          {/* Messages Scroll Area */}
          <div className="flex-1 overflow-y-auto px-4 py-3.5 space-y-3.5 bg-slate-50/50 fb-chat-scroll">
            {messages.map((msg) => {
              const isUser = msg.sender === 'user';
              return (
                <div
                  key={msg.id}
                  className={`flex flex-col fb-message-anim ${isUser ? 'items-end' : 'items-start'}`}
                >
                  <div
                    className={`max-w-[88%] rounded-2xl px-3.5 py-2.5 shadow-xs text-xs sm:text-[13px] leading-relaxed ${
                      isUser
                        ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white rounded-br-xs'
                        : msg.isError
                        ? 'bg-rose-50 border border-rose-200 text-rose-900 rounded-bl-xs'
                        : 'bg-white border border-slate-200/80 text-slate-800 rounded-bl-xs'
                    }`}
                  >
                    {isUser ? (
                      <p className="whitespace-pre-wrap">{msg.text}</p>
                    ) : (
                      <div className="fb-markdown-body">
                        <ReactMarkdown>{msg.text}</ReactMarkdown>
                      </div>
                    )}
                  </div>

                  {/* Timestamp & Status Metadata */}
                  <div
                    className={`flex items-center gap-1.5 mt-1 px-1 text-[10px] text-slate-400 ${
                      isUser ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    <Clock className="w-2.5 h-2.5" />
                    <span>{msg.timestamp}</span>
                    {msg.intent && (
                      <span className="font-medium text-slate-500 bg-slate-100 px-1 rounded text-[9px]">
                        {msg.intent.replace(/_/g, ' ')}
                      </span>
                    )}
                  </div>

                  {/* Suggested Question Pills (Shown with welcome message) */}
                  {msg.isWelcome && messages.length === 1 && (
                    <div className="mt-3 w-full space-y-1.5 fb-message-anim">
                      <div className="text-[11px] font-semibold text-slate-500 px-1 flex items-center gap-1">
                        <Sparkles className="w-3 h-3 text-emerald-600" />
                        <span>Suggested Queries:</span>
                      </div>
                      <div className="grid grid-cols-1 gap-1.5">
                        {SUGGESTED_QUESTIONS.map((sq, i) => {
                          const Icon = sq.icon;
                          return (
                            <button
                              key={i}
                              type="button"
                              onClick={() => handleSend(sq.query)}
                              disabled={isLoading}
                              className={`flex items-center justify-between text-left px-3 py-2 rounded-xl text-xs font-medium bg-white border border-slate-200/90 shadow-2xs hover:shadow-xs transition-all duration-150 group cursor-pointer ${sq.color} disabled:opacity-50 disabled:cursor-not-allowed`}
                            >
                              <span className="flex items-center gap-2 truncate">
                                <span className="text-sm">{sq.emoji}</span>
                                <span className="truncate">{sq.label}</span>
                              </span>
                              <Icon className="w-3.5 h-3.5 text-slate-400 group-hover:text-emerald-600 transition-colors shrink-0 ml-1.5" />
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}

            {/* Typing Indicator */}
            {isLoading && (
              <div className="flex items-start fb-message-anim">
                <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-xs px-3.5 py-2.5 shadow-xs flex items-center space-x-1.5 text-slate-500">
                  <Bot className="w-3.5 h-3.5 text-emerald-600 mr-1 shrink-0 animate-pulse" />
                  <span className="text-xs text-slate-500 font-medium mr-1">FreshBasket AI is thinking</span>
                  <span className="w-1.5 h-1.5 bg-emerald-600 rounded-full fb-dot-1" />
                  <span className="w-1.5 h-1.5 bg-emerald-600 rounded-full fb-dot-2" />
                  <span className="w-1.5 h-1.5 bg-emerald-600 rounded-full fb-dot-3" />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Chat Input Area Fixed at Bottom */}
          <div className="p-3 bg-white border-t border-slate-200/80 shrink-0">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSend();
              }}
              className="flex items-center gap-2"
            >
              <div className="relative flex-1">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputQuestion}
                  onChange={(e) => setInputQuestion(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask anything about your business..."
                  disabled={isLoading}
                  className="w-full text-xs sm:text-[13px] text-slate-800 placeholder-slate-400 bg-slate-50/90 border border-slate-200 rounded-xl px-3.5 py-2.5 focus:outline-none focus:ring-2 focus:ring-emerald-500/30 focus:border-emerald-500 focus:bg-white transition-all disabled:opacity-50"
                  aria-label="Assistant question input"
                />
              </div>
              <button
                type="submit"
                disabled={!inputQuestion.trim() || isLoading}
                aria-label="Send message"
                className="w-9 h-9 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-600 text-white flex items-center justify-center hover:from-emerald-500 hover:to-teal-500 disabled:from-slate-200 disabled:to-slate-300 disabled:text-slate-400 disabled:cursor-not-allowed shadow-sm hover:shadow transition-all duration-150 shrink-0 cursor-pointer"
              >
                <Send className="w-4 h-4 ml-0.5" />
              </button>
            </form>
            <div className="mt-1.5 flex items-center justify-between text-[10px] text-slate-400 px-1">
              <span>Press <kbd className="font-mono bg-slate-100 px-1 py-0.5 rounded text-[9px] text-slate-500">Enter</kbd> to send</span>
              <span className="flex items-center gap-1 text-emerald-700 font-medium">
                <ShieldCheck className="w-2.5 h-2.5 text-emerald-600" />
                Ground-Truth AI
              </span>
            </div>
          </div>
        </div>
      )}

      {/* 2. FLOATING ASSISTANT BUTTON CONTAINER */}
      <div className="relative flex items-center justify-end">
        {/* Tooltip on hover when closed */}
        {showTooltip && !isOpen && (
          <div className="pointer-events-none absolute right-16 top-1/2 -translate-y-1/2 whitespace-nowrap bg-slate-900 text-white text-xs font-semibold px-3 py-1.5 rounded-xl shadow-lg fb-tooltip-anim border border-slate-700 flex items-center gap-1.5 mr-2">
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            <span>Ask FreshBasket AI</span>
            <span className="w-2 h-2 bg-slate-900 border-t border-r border-slate-700 rotate-45 absolute -right-1 top-1/2 -translate-y-1/2" />
          </div>
        )}

        {/* Pulse attention ring */}
        {!isOpen && !hasInteracted && (
          <div className="absolute -inset-1 rounded-full bg-emerald-500/30 fb-pulse-ring pointer-events-none" />
        )}

        {/* Circular Floating Button */}
        <button
          type="button"
          onClick={toggleOpen}
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
          aria-expanded={isOpen && !isMinimized}
          aria-label={isOpen && !isMinimized ? 'Close FreshBasket AI Assistant' : 'Ask FreshBasket AI'}
          className={`pointer-events-auto relative w-14 h-14 rounded-full text-white shadow-xl flex items-center justify-center cursor-pointer transition-all duration-300 fb-float-btn ${
            isOpen && !isMinimized
              ? 'bg-slate-800 hover:bg-slate-900 shadow-slate-900/30 rotate-0'
              : 'bg-gradient-to-tr from-emerald-600 via-teal-600 to-emerald-500 hover:from-emerald-500 hover:to-teal-500 shadow-emerald-700/35 hover:shadow-emerald-700/50'
          }`}
        >
          {isOpen && !isMinimized ? (
            <ChevronDown className="w-6 h-6 text-white" />
          ) : (
            <>
              <Bot className="w-6 h-6 text-white" />
              {/* Online glowing dot badge on button */}
              <span className="absolute top-1 right-1 flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-300 opacity-75" />
                <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-400 border-2 border-white" />
              </span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
};
