import React, { createContext, useContext } from 'react';

// Create context
const DocumentsContext = createContext();

// Custom hook to use the documents context
export const useDocuments = () => {
  const context = useContext(DocumentsContext);
  if (!context) {
    throw new Error('useDocuments must be used within a DocumentsProvider');
  }
  return context;
};

// Provider component
export const DocumentsProvider = ({ children, value }) => {
  return (
    <DocumentsContext.Provider value={value}>
      {children}
    </DocumentsContext.Provider>
  );
}; 