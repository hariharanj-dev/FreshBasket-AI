import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response error interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    let message = 'An unexpected error occurred.';
    if (error.response?.data?.detail) {
      if (typeof error.response.data.detail === 'string') {
        message = error.response.data.detail;
      } else if (Array.isArray(error.response.data.detail?.errors)) {
        message = error.response.data.detail.errors.join('; ');
      } else {
        message = JSON.stringify(error.response.data.detail);
      }
    } else if (error.message) {
      message = error.message;
    }
    return Promise.reject(new Error(message));
  }
);

// 1. Health & Filters
export const getHealth = async () => {
  const { data } = await apiClient.get('/api/health');
  return data;
};

export const getFilters = async () => {
  const { data } = await apiClient.get('/api/filters');
  return data;
};

// 2. Overview
export const getOverview = async (params = {}) => {
  const { data } = await apiClient.get('/api/overview', { params });
  return data;
};

// 3. Sales Analysis
export const getSales = async (params = {}) => {
  const { data } = await apiClient.get('/api/sales', { params });
  return data;
};

// 4. Forecasts
export const getForecasts = async (params = {}) => {
  const { data } = await apiClient.get('/api/forecasts', { params });
  return data;
};

// 5. Model Evaluation
export const getEvaluation = async () => {
  const { data } = await apiClient.get('/api/evaluation');
  return data;
};

// 6. Product Family Insights
export const getProducts = async (params = {}) => {
  const { data } = await apiClient.get('/api/products', { params });
  return data;
};

// 7. Store Performance
export const getStores = async (params = {}) => {
  const { data } = await apiClient.get('/api/stores', { params });
  return data;
};

// 8. Inventory Planning
export const getInventorySummary = async (params = {}) => {
  const { data } = await apiClient.get('/api/inventory/summary', { params });
  return data;
};

export const getInventoryRecommendations = async (params = {}) => {
  const { data } = await apiClient.get('/api/inventory/recommendations', { params });
  return data;
};

export const uploadInventoryCSV = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await apiClient.post('/api/inventory/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return data;
};

export const resetInventory = async () => {
  const { data } = await apiClient.post('/api/inventory/reset');
  return data;
};

// 9. Alerts
export const getInventoryAlerts = async (params = {}) => {
  const { data } = await apiClient.get('/api/inventory/alerts', { params });
  return data;
};

export const captureAlertSnapshot = async (model = 'naive_forecast') => {
  const { data } = await apiClient.post(`/api/inventory/alerts/snapshot?model=${model}`);
  return data;
};

export const getAlertHistory = async () => {
  const { data } = await apiClient.get('/api/inventory/alerts/history');
  return data;
};

// 10. Reports & Downloads
export const downloadReplenishmentCSV = async (params = {}) => {
  const response = await apiClient.get('/api/reports/replenishment', {
    params,
    responseType: 'blob',
  });
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = url;
  const contentDispo = response.headers['content-disposition'];
  let filename = 'freshbasket_replenishment_report.csv';
  if (contentDispo) {
    const match = contentDispo.match(/filename=(.+)/);
    if (match && match[1]) filename = match[1].replace(/"/g, '');
  }
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
};

export const getExecutiveReport = async (params = {}) => {
  const { data } = await apiClient.get('/api/reports/executive', {
    params,
    responseType: 'text',
  });
  return data;
};

export const getEmailDraft = async (params = {}) => {
  const { data } = await apiClient.get('/api/reports/email', {
    params,
    responseType: 'text',
  });
  return data;
};

// 11. AI Business Insights
export const getInsightsOverview = async (params = {}) => {
  const { data } = await apiClient.get('/api/insights/overview', { params });
  return data;
};

export const getInsightsDemandTrends = async (params = {}) => {
  const { data } = await apiClient.get('/api/insights/demand-trends', { params });
  return data;
};

export const getInsightsProductOpportunities = async (params = {}) => {
  const { data } = await apiClient.get('/api/insights/product-opportunities', { params });
  return data;
};

export const getInsightsInventoryRisks = async (params = {}) => {
  const { data } = await apiClient.get('/api/insights/inventory-risks', { params });
  return data;
};

// 12. AI Business Query Assistant ("Ask FreshBasket AI")
export const queryAssistant = async (payload) => {
  const { data } = await apiClient.post('/api/assistant/query', payload);
  return data;
};

// 13. What-If Scenario Simulator
export const simulateScenario = async (payload) => {
  const { data } = await apiClient.post('/api/scenarios/simulate', payload);
  return data;
};
