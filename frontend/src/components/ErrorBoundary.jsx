import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-red-900 my-6 shadow-2xs">
          <div className="flex items-start space-x-3.5">
            <div className="p-2 bg-red-100 rounded-lg text-red-600 shrink-0">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="text-sm font-bold text-red-900">
                {this.props.title || 'An unexpected rendering error occurred'}
              </h3>
              <p className="text-xs text-red-700 mt-1 leading-relaxed break-words font-mono bg-red-100/50 p-2 rounded">
                {this.state.error?.message || 'Component failed to render.'}
              </p>
              <button
                onClick={this.handleReset}
                className="mt-3.5 inline-flex items-center space-x-1.5 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-xs font-semibold rounded-lg transition-colors shadow-2xs"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Retry Component</span>
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
