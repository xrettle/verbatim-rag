import React, { createContext, useContext, useState, useEffect } from 'react';

// Create context
const SidebarContext = createContext();

// Custom hook to use the sidebar context
export const useSidebar = () => useContext(SidebarContext);

// Provider component
export const SidebarProvider = ({ children }) => {
  // Default to open on first load for desktop, closed for mobile
  const [isOpen, setIsOpen] = useState(false);
  
  // Check if we should open the sidebar on first load
  useEffect(() => {
    // Open sidebar automatically on first load for desktop
    const isMobile = window.innerWidth < 768;
    if (!isMobile) {
      setIsOpen(true);
    }
  }, []);
  
  const onOpen = () => setIsOpen(true);
  const onClose = () => setIsOpen(false);
  const onToggle = () => setIsOpen(!isOpen);
  
  // Value object
  const value = {
    isOpen,
    onOpen,
    onClose,
    onToggle
  };
  
  return <SidebarContext.Provider value={value}>{children}</SidebarContext.Provider>;
};

export default SidebarContext; 