import React, { createContext, useContext, useState, useEffect } from 'react';
import { getFilters, getHealth } from '../services/api';

const FilterContext = createContext(null);

export const FilterProvider = ({ children }) => {
  const [filterOptions, setFilterOptions] = useState({
    stores: [],
    families: [],
    models: [
      { id: 'naive_forecast', label: 'Naive Baseline', wape: 11.61, is_best: true },
      { id: 'lightgbm_forecast', label: 'LightGBM (Recursive)', wape: 12.27, is_best: false },
      { id: 'seasonal_naive_forecast', label: 'Seasonal Naive (Lag 52)', wape: 16.17, is_best: false },
    ],
    date_range: {
      min_date: '2013-01-07',
      max_date: '2017-08-15',
      last_complete_week: '2017-08-13',
      forecast_origin: '2017-08-13',
      forecast_start: '2017-08-14',
      forecast_end: '2017-11-12',
    },
  });

  const [selectedStore, setSelectedStore] = useState(''); // '' means All Stores
  const [selectedFamily, setSelectedFamily] = useState(''); // '' means All Families
  const [selectedModel, setSelectedModel] = useState('naive_forecast');
  const [dataMode, setDataMode] = useState('DEMO'); // DEMO or CUSTOM
  const [loadingFilters, setLoadingFilters] = useState(true);
  const [apiOnline, setApiOnline] = useState(true);

  const fetchMetadata = async () => {
    try {
      const [filterData, healthData] = await Promise.all([getFilters(), getHealth()]);
      setFilterOptions(filterData);
      if (healthData.data_mode) {
        setDataMode(healthData.data_mode);
      }
      setApiOnline(true);
    } catch (err) {
      console.warn('Failed to fetch initial filters or health:', err.message);
      setApiOnline(false);
    } finally {
      setLoadingFilters(false);
    }
  };

  useEffect(() => {
    fetchMetadata();
  }, []);

  const resetFilters = () => {
    setSelectedStore('');
    setSelectedFamily('');
    setSelectedModel('naive_forecast');
  };

  return (
    <FilterContext.Provider
      value={{
        filterOptions,
        selectedStore,
        setSelectedStore,
        selectedFamily,
        setSelectedFamily,
        selectedModel,
        setSelectedModel,
        dataMode,
        setDataMode,
        loadingFilters,
        apiOnline,
        resetFilters,
        refreshMetadata: fetchMetadata,
      }}
    >
      {children}
    </FilterContext.Provider>
  );
};

export const useFilters = () => {
  const context = useContext(FilterContext);
  if (!context) {
    throw new Error('useFilters must be used within a FilterProvider');
  }
  return context;
};
