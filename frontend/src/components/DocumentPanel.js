import React, { useState } from 'react';
import {
  Box,
  Flex,
  Heading,
  IconButton,
  Spinner,
  Text,
  useColorModeValue,
  Input,
  InputGroup,
  InputLeftElement,
  Badge,
} from '@chakra-ui/react';
import { FaChevronLeft, FaSearch } from 'react-icons/fa';
import { motion, AnimatePresence } from 'framer-motion';
import { useApi } from '../contexts/ApiContext';
import { useDocuments } from '../contexts/DocumentsContext';
import DocumentViewer from './DocumentViewer';

const MotionBox = motion(Box);

const DocumentPanel = ({ onBack }) => {
  const { isLoading, currentQuery } = useApi();
  const { selectedDocId } = useDocuments();
  const [searchText, setSearchText] = useState('');
  
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  // Determine if we're on mobile based on whether onBack is provided
  const isMobile = !!onBack;
  
  // Get the current document based on selectedDocId
  const document = currentQuery?.documents && selectedDocId?.docIndex !== undefined
    ? currentQuery.documents[selectedDocId.docIndex]
    : null;
  
  // Filter highlights if search text is provided
  const filteredDocument = document && searchText
    ? {
        ...document,
        highlights: document.highlights.filter(h => 
          h.text.toLowerCase().includes(searchText.toLowerCase())
        )
      }
    : document;
  
  return (
    <MotionBox
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ duration: 0.3 }}
      bg={bgColor}
      borderRadius="lg"
      borderWidth="1px"
      borderColor={borderColor}
      h="100%"
      display="flex"
      flexDirection="column"
    >
      {/* Header */}
      <Flex
        p={4}
        borderBottomWidth="1px"
        borderColor={borderColor}
        align="center"
        justify="space-between"
      >
        <Flex align="center">
          {isMobile && (
            <IconButton
              icon={<FaChevronLeft />}
              aria-label="Back"
              variant="ghost"
              mr={2}
              onClick={onBack}
            />
          )}
          <Heading size="md">Source Document</Heading>
        </Flex>
        
        {document && currentQuery && (
          <Text fontSize="sm">
            Document {selectedDocId.docIndex + 1} of {currentQuery.documents.length}
          </Text>
        )}
      </Flex>
      
      {/* Search bar - only show if we have a document */}
      {document && (
        <Box px={4} py={3} borderBottomWidth="1px" borderColor={borderColor}>
          <InputGroup size="sm">
            <InputLeftElement pointerEvents="none">
              <FaSearch color="gray.300" />
            </InputLeftElement>
            <Input
              placeholder="Search in highlights..."
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              borderRadius="md"
            />
          </InputGroup>
          {searchText && (
            <Flex mt={2} align="center">
              <Text fontSize="xs" color="gray.500" mr={2}>
                Found:
              </Text>
              <Badge colorScheme={filteredDocument.highlights.length > 0 ? "green" : "red"}>
                {filteredDocument.highlights.length} highlight{filteredDocument.highlights.length !== 1 ? 's' : ''}
              </Badge>
            </Flex>
          )}
        </Box>
      )}
      
      {/* Content */}
      <Box flex="1" overflowY="auto" p={4}>
        <AnimatePresence mode="wait">
          {isLoading ? (
            <Flex justify="center" align="center" h="100%">
              <Spinner size="xl" color="blue.500" thickness="4px" />
            </Flex>
          ) : !currentQuery ? (
            <Flex
              justify="center"
              align="center"
              h="100%"
              textAlign="center"
              px={4}
            >
              <Text color="gray.500">
                Ask a question to see source documents
              </Text>
            </Flex>
          ) : !selectedDocId ? (
            <Flex
              justify="center"
              align="center"
              h="100%"
              textAlign="center"
              px={4}
            >
              <Text color="gray.500">
                Click on a document button to view its content
              </Text>
            </Flex>
          ) : document ? (
            <DocumentViewer 
              document={filteredDocument} 
              selectedHighlightIndex={selectedDocId.highlightIndex}
            />
          ) : (
            <Flex
              justify="center"
              align="center"
              h="100%"
              textAlign="center"
              px={4}
            >
              <Text color="gray.500">
                Document not found
              </Text>
            </Flex>
          )}
        </AnimatePresence>
      </Box>
    </MotionBox>
  );
};

export default DocumentPanel; 