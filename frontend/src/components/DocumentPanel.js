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
  Button,
} from '@chakra-ui/react';
import { FaChevronLeft, FaSearch, FaChevronRight } from 'react-icons/fa';
import { motion, AnimatePresence } from 'framer-motion';
import { useApi } from '../contexts/ApiContext';
import { useDocuments } from '../contexts/DocumentsContext';
import DocumentViewer from './DocumentViewer';

const MotionBox = motion(Box);

const DocumentPanel = ({ onBack }) => {
  const { isLoading, currentQuery } = useApi();
  const { selectedDocId, setSelectedDocId } = useDocuments();
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
        
        {currentQuery && currentQuery.documents && currentQuery.documents.length > 0 && (
          <Flex align="center">
            <Text fontSize="sm" mr={2}>
              Document {selectedDocId ? selectedDocId.docIndex + 1 : 0} of {currentQuery.documents.length}
            </Text>
            
            {/* Navigation buttons */}
            <Flex>
              <IconButton
                icon={<FaChevronLeft />}
                aria-label="Previous document"
                size="xs"
                variant="ghost"
                isDisabled={!selectedDocId || selectedDocId.docIndex === 0}
                onClick={() => {
                  if (selectedDocId && selectedDocId.docIndex > 0) {
                    setSelectedDocId({ docIndex: selectedDocId.docIndex - 1 });
                  }
                }}
                mr={1}
              />
              <IconButton
                icon={<FaChevronRight />}
                aria-label="Next document"
                size="xs"
                variant="ghost"
                isDisabled={!selectedDocId || !currentQuery || !currentQuery.documents || selectedDocId.docIndex === currentQuery.documents.length - 1}
                onClick={() => {
                  if (selectedDocId && currentQuery && currentQuery.documents && 
                      selectedDocId.docIndex < currentQuery.documents.length - 1) {
                    setSelectedDocId({ docIndex: selectedDocId.docIndex + 1 });
                  }
                }}
              />
            </Flex>
          </Flex>
        )}
      </Flex>
      
      {/* Document selector - show if we have documents */}
      {currentQuery && currentQuery.documents && currentQuery.documents.length > 0 && (
        <Box px={4} py={3} borderBottomWidth="1px" borderColor={borderColor}>
          <Flex justify="space-between" align="center" mb={2}>
            <Text fontSize="sm" fontWeight="medium">
              Source Documents ({currentQuery.documents.length})
            </Text>
            <Flex gap={2} align="center">
              <Flex align="center">
                <Box w="10px" h="10px" borderRadius="full" bg="green.500" mr={1}></Box>
                <Text fontSize="xs">With highlights</Text>
              </Flex>
              <Flex align="center">
                <Box w="10px" h="10px" borderRadius="full" bg="gray.400" mr={1}></Box>
                <Text fontSize="xs">No highlights</Text>
              </Flex>
            </Flex>
          </Flex>
          
          <Text fontSize="xs" color="gray.500" mb={2}>
            Click on a document to view its content:
          </Text>
          
          <Flex wrap="wrap" gap={2}>
            {currentQuery.documents.map((doc, idx) => {
              const hasHighlights = doc.highlights && doc.highlights.length > 0;
              const isSelected = selectedDocId && selectedDocId.docIndex === idx;
              
              return (
                <Button
                  key={idx}
                  size="xs"
                  variant={isSelected ? "solid" : "outline"}
                  colorScheme={hasHighlights ? "green" : "gray"}
                  onClick={() => setSelectedDocId({ docIndex: idx })}
                  leftIcon={
                    <Box 
                      w="8px" 
                      h="8px" 
                      borderRadius="full" 
                      bg={hasHighlights ? "green.500" : "gray.400"}
                    />
                  }
                  boxShadow={isSelected ? "md" : "none"}
                  _hover={{ transform: "translateY(-2px)", boxShadow: "sm" }}
                  transition="all 0.2s"
                >
                  {idx + 1}
                  {hasHighlights && (
                    <Badge ml={1} colorScheme="green" variant="solid">
                      {doc.highlights.length}
                    </Badge>
                  )}
                </Button>
              );
            })}
          </Flex>
        </Box>
      )}
      
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
          {isLoading && !currentQuery ? (
            <Flex justify="center" align="center" h="100%">
              <Spinner size="xl" color="blue.500" thickness="4px" />
            </Flex>
          ) : isLoading && currentQuery && !selectedDocId ? (
            <Flex
              justify="center"
              align="center"
              h="100%"
              textAlign="center"
              px={4}
              direction="column"
            >
              <Spinner size="lg" color="blue.500" thickness="3px" mb={4} />
              <Text color="gray.500">
                Loading documents...
              </Text>
              {currentQuery.documents && currentQuery.documents.length > 0 && (
                <Text color="blue.500" mt={2} fontWeight="medium">
                  Found {currentQuery.documents.length} document{currentQuery.documents.length !== 1 ? 's' : ''}
                </Text>
              )}
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
              direction="column"
            >
              <Box
                mb={6}
                p={4}
                borderRadius="md"
                bg="blue.50"
                borderWidth="1px"
                borderColor="blue.200"
                maxW="400px"
              >
                <Text fontSize="md" fontWeight="medium" color="blue.700" mb={3}>
                  Source Documents Available
                </Text>
                <Text color="gray.600" fontSize="sm" mb={4}>
                  {currentQuery.documents && currentQuery.documents.length > 0 
                    ? `${currentQuery.documents.length} documents were retrieved for your query. Click on any document button above to view its content.`
                    : "No documents were found for this query."}
                </Text>
                
                {currentQuery.documents && currentQuery.documents.length > 0 && (
                  <Flex wrap="wrap" gap={2} justify="center">
                    {currentQuery.documents.slice(0, 5).map((doc, idx) => (
                      <Button
                        key={idx}
                        size="sm"
                        colorScheme={doc.highlights && doc.highlights.length > 0 ? "green" : "gray"}
                        onClick={() => setSelectedDocId({ docIndex: idx })}
                        leftIcon={
                          <Box 
                            w="8px" 
                            h="8px" 
                            borderRadius="full" 
                            bg={doc.highlights && doc.highlights.length > 0 ? "green.500" : "gray.400"}
                          />
                        }
                      >
                        Document {idx + 1}
                      </Button>
                    ))}
                    
                    {currentQuery.documents.length > 5 && (
                      <Text fontSize="sm" color="gray.500" mt={2} width="100%" textAlign="center">
                        + {currentQuery.documents.length - 5} more documents
                      </Text>
                    )}
                  </Flex>
                )}
              </Box>
            </Flex>
          ) : document ? (
            <DocumentViewer 
              document={filteredDocument} 
              selectedHighlightIndex={selectedDocId.highlightIndex}
              isLoading={isLoading}
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