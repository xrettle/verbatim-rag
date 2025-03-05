import React, { useState, useEffect } from 'react';
import {
  Box,
  ChakraProvider,
  Container,
  Grid,
  GridItem,
  extendTheme,
} from '@chakra-ui/react';
import { AnimatePresence } from 'framer-motion';

import ChatPanel from './components/ChatPanel';
import DocumentPanel from './components/DocumentPanel';
import Header from './components/Header';
import { ApiProvider, useApi } from './contexts/ApiContext';
import { DocumentsProvider } from './contexts/DocumentsContext';

// Create a custom theme
const theme = extendTheme({
  config: {
    initialColorMode: 'light',
    useSystemColorMode: false,
  },
});

// Main content component that manages state
const AppContent = () => {
  const { loadResources } = useApi();
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const [showDocPanel, setShowDocPanel] = useState(!isMobile);

  // Load resources on mount
  useEffect(() => {
    loadResources();
  }, [loadResources]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      
      // On desktop, always show doc panel
      // On mobile, only show if a document is selected
      if (!mobile) {
        setShowDocPanel(true);
      } else if (!selectedDocId) {
        setShowDocPanel(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [selectedDocId]);

  // Handle document selection
  const handleDocumentSelect = (docInfo) => {
    setSelectedDocId(docInfo);
    
    // On mobile, show the document panel when a document is selected
    if (isMobile) {
      setShowDocPanel(true);
    }
  };

  // Handle back navigation on mobile
  const handleBack = () => {
    if (isMobile) {
      setShowDocPanel(false);
    }
  };

  return (
    <DocumentsProvider value={{ selectedDocId, setSelectedDocId: handleDocumentSelect }}>
      <Box minH="100vh" bg="gray.50">
        <Header />
        
        <Container maxW="container.xl" py={4}>
          <Grid
            templateColumns={isMobile ? "1fr" : "1fr 1fr"}
            gap={4}
            h="calc(100vh - 80px)"
          >
            {/* Show chat panel if on desktop or if doc panel is hidden on mobile */}
            {(!isMobile || !showDocPanel) && (
              <GridItem>
                <ChatPanel />
              </GridItem>
            )}
            
            {/* Show document panel if it should be visible */}
            {showDocPanel && (
              <GridItem>
                <DocumentPanel onBack={handleBack} />
              </GridItem>
            )}
          </Grid>
        </Container>
      </Box>
    </DocumentsProvider>
  );
};

function App() {
  return (
    <ChakraProvider theme={theme}>
      <ApiProvider>
        <AppContent />
      </ApiProvider>
    </ChakraProvider>
  );
}

export default App; 