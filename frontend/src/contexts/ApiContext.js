import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import axios from 'axios';

// Create context
const ApiContext = createContext();

// Custom hook to use the API context
export const useApi = () => useContext(ApiContext);

// Provider component
export const ApiProvider = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isResourcesLoaded, setIsResourcesLoaded] = useState(false);
  const [currentQuery, setCurrentQuery] = useState(null);
  const [numDocs, setNumDocs] = useState(5); // Default to 5 documents

  // Function to check if resources are loaded
  const refreshStatus = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await axios.get('/api/status');
      setIsResourcesLoaded(response.data.resources_loaded);
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message;
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Check if resources are loaded on mount
  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  // Submit a query
  const submitQuery = useCallback(async (question) => {
    if (!isResourcesLoaded) {
      setError('Resources not loaded. Please wait for the system to initialize.');
      return null;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/api/query', {
        question,
        num_docs: numDocs,
      });
      
      // Store the current query result
      setCurrentQuery(response.data);
      
      return response.data;
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message;
      setError(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [isResourcesLoaded, numDocs]);

  // Load resources with optional API key
  const loadResources = useCallback(async (apiKey = null) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const payload = apiKey ? { api_key: apiKey } : {};
      const response = await axios.post('/api/load-resources', payload);
      
      setIsResourcesLoaded(response.data.success);
      
      if (!response.data.success) {
        setError(response.data.message);
      }
      
      return response.data;
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message;
      setError(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Update number of documents to retrieve - keep this for internal use
  const updateNumDocs = useCallback((num) => {
    setNumDocs(num);
  }, []);

  // Reset current query
  const resetQuery = useCallback(() => {
    setCurrentQuery(null);
  }, []);

  // Value object
  const value = {
    isLoading,
    error,
    isResourcesLoaded,
    currentQuery,
    numDocs,
    submitQuery,
    loadResources,
    updateNumDocs,
    resetQuery,
    refreshStatus,
  };

  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
};

export default ApiContext; 