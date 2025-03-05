import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Button,
  Flex,
  FormControl,
  Input,
  Text,
  VStack,
  useColorModeValue,
  Spinner,
  Heading,
  Divider,
  List,
  ListItem,
  ListIcon,
  Tooltip,
  Badge,
  IconButton,
  Tag,
  TagLabel,
  TagLeftIcon,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Avatar,
  InputGroup,
  InputRightElement,
  Alert,
  AlertIcon,
  AlertDescription,
} from '@chakra-ui/react';
import { FaPaperPlane, FaQuoteRight, FaSearch, FaLink, FaComments, FaFileAlt, FaFile } from 'react-icons/fa';
import { motion, AnimatePresence } from 'framer-motion';
import { useApi } from '../contexts/ApiContext';
import { useDocuments } from '../contexts/DocumentsContext';

const MotionBox = motion(Box);

const ChatPanel = () => {
  const { isLoading, isResourcesLoaded, currentQuery, submitQuery } = useApi();
  const { setSelectedDocId } = useDocuments();
  const [question, setQuestion] = useState('');
  const [searchText, setSearchText] = useState('');
  const [selectedHighlight, setSelectedHighlight] = useState(null);
  const messagesEndRef = useRef(null);
  const answerRef = useRef(null);
  
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const questionBgColor = useColorModeValue('gray.100', 'gray.700');
  const answerBgColor = useColorModeValue('brand.50', 'brand.900');
  const answerTextColor = useColorModeValue('gray.800', 'white');
  const highlightColor = useColorModeValue('yellow.100', 'yellow.800');
  const selectedHighlightColor = useColorModeValue('orange.100', 'orange.800');
  const searchHighlightColor = useColorModeValue('green.100', 'green.800');
  
  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [currentQuery]);
  
  // Format the answer with citations when available
  useEffect(() => {
    if (!answerRef.current || !currentQuery) return;
    
    const answer = currentQuery.answer;
    
    // If we have structured answer with citations, use it
    if (currentQuery.structured_answer && currentQuery.structured_answer.citations) {
      const citations = currentQuery.structured_answer.citations;
      
      // If we have a selected highlight, highlight it in the answer
      if (selectedHighlight) {
        const highlightText = selectedHighlight.text;
        
        // Create HTML with the highlight
        let html = '';
        let lastIndex = 0;
        let startIndex = answer.indexOf(highlightText);
        
        while (startIndex !== -1) {
          // Add text before the match
          html += answer.substring(lastIndex, startIndex);
          
          // Add the highlighted match
          html += `<mark style="background-color: ${searchHighlightColor}; padding: 0 2px; border-radius: 2px;">${highlightText}</mark>`;
          
          // Update indices
          lastIndex = startIndex + highlightText.length;
          startIndex = answer.indexOf(highlightText, lastIndex);
        }
        
        // Add remaining text
        html += answer.substring(lastIndex);
        
        // Set the HTML
        answerRef.current.innerHTML = html;
      } else {
        // No selected highlight, just show the answer with citation markers
        let html = answer;
        
        // Add citation markers
        citations.forEach((citation, index) => {
          const citationText = citation.text;
          const citationMarker = `<sup><a href="#" data-citation="${index}" style="color: #3182CE; text-decoration: none; font-weight: bold;">[${index + 1}]</a></sup>`;
          
          // Replace the citation text with the citation text + marker
          // Only replace the first occurrence to avoid duplicate markers
          const citationIndex = html.indexOf(citationText);
          if (citationIndex !== -1) {
            html = html.substring(0, citationIndex + citationText.length) + 
                   citationMarker + 
                   html.substring(citationIndex + citationText.length);
          }
        });
        
        // Set the HTML
        answerRef.current.innerHTML = html;
        
        // Add click event listeners to citation markers
        const citationLinks = answerRef.current.querySelectorAll('a[data-citation]');
        citationLinks.forEach(link => {
          link.addEventListener('click', (e) => {
            e.preventDefault();
            const citationIndex = parseInt(link.getAttribute('data-citation'));
            const citation = citations[citationIndex];
            
            // Find the highlight in the document
            if (citation && currentQuery.documents[citation.doc_index]) {
              const doc = currentQuery.documents[citation.doc_index];
              if (doc.highlights && doc.highlights[citation.highlight_index]) {
                const highlight = {
                  ...doc.highlights[citation.highlight_index],
                  docIndex: citation.doc_index
                };
                setSelectedHighlight(highlight);
                setSelectedDocId({ docIndex: citation.doc_index });
              }
            }
          });
        });
      }
    } else {
      // No structured answer, just show the plain text
      answerRef.current.textContent = answer;
    }
    
    return () => {
      if (answerRef.current) {
        answerRef.current.textContent = currentQuery.answer;
      }
    };
  }, [currentQuery, selectedHighlight, searchHighlightColor, setSelectedDocId]);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim() || !isResourcesLoaded) return;
    
    setSelectedHighlight(null);
    setSearchText('');
    await submitQuery(question);
    setQuestion('');
  };

  // Extract all unique highlights from all documents
  const getAllHighlights = () => {
    if (!currentQuery || !currentQuery.documents) return [];
    
    const allHighlights = [];
    currentQuery.documents.forEach((doc, docIndex) => {
      if (doc.highlights && doc.highlights.length > 0) {
        doc.highlights.forEach(highlight => {
          // Add document index to each highlight for reference
          allHighlights.push({
            ...highlight,
            docIndex
          });
        });
      }
    });
    
    // Sort by document index and then by start position
    return allHighlights.sort((a, b) => {
      if (a.docIndex !== b.docIndex) return a.docIndex - b.docIndex;
      return a.start - b.start;
    });
  };
  
  // Filter highlights by search text
  const getFilteredHighlights = () => {
    const highlights = getAllHighlights();
    if (!searchText) return highlights;
    
    return highlights.filter(highlight => 
      highlight.text.toLowerCase().includes(searchText.toLowerCase())
    );
  };
  
  // Handle highlight selection
  const handleHighlightClick = (highlight) => {
    setSelectedHighlight(highlight === selectedHighlight ? null : highlight);
    setSelectedDocId({ docIndex: highlight.docIndex });
  };
  
  // Get citation for a highlight
  const getCitationIndex = (highlight) => {
    if (!currentQuery || !currentQuery.structured_answer || !currentQuery.structured_answer.citations) return -1;
    
    return currentQuery.structured_answer.citations.findIndex(
      citation => citation.doc_index === highlight.docIndex && 
                 citation.highlight_index === currentQuery.documents[highlight.docIndex].highlights.findIndex(
                   h => h.text === highlight.text
                 )
    );
  };
  
  return (
    <MotionBox
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
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
      <Box p={4} borderBottomWidth="1px" borderColor={borderColor}>
        <Heading size="md">Conversation</Heading>
      </Box>
      
      {/* Messages */}
      <Box flex="1" overflowY="auto" p={4}>
        <AnimatePresence>
          {!currentQuery ? (
            <MotionBox
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              textAlign="center"
              py={10}
            >
              <Text color="gray.500">
                {isResourcesLoaded
                  ? "Ask a question to get started"
                  : "Load resources to begin"}
              </Text>
            </MotionBox>
          ) : (
            <VStack spacing={6} align="stretch">
              {/* Question */}
              <MotionBox
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <Flex direction="column">
                  <Text fontWeight="bold" mb={1}>
                    You
                  </Text>
                  <Box
                    bg={questionBgColor}
                    p={3}
                    borderRadius="md"
                  >
                    <Text>{currentQuery.question}</Text>
                  </Box>
                </Flex>
              </MotionBox>
              
              {/* Answer */}
              <MotionBox
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.1 }}
              >
                <Flex direction="column">
                  <Text fontWeight="bold" mb={1}>
                    Assistant
                  </Text>
                  <Box
                    bg={answerBgColor}
                    p={3}
                    borderRadius="md"
                    color={answerTextColor}
                  >
                    <Box ref={answerRef} whiteSpace="pre-wrap">{currentQuery.answer}</Box>
                    
                    {/* Relevant Sentences */}
                    <Box mt={4}>
                      <Divider my={2} />
                      
                      <Flex justify="space-between" align="center" mb={2}>
                        <Text fontWeight="medium" fontSize="sm">
                          Relevant Sentences:
                        </Text>
                        <Flex align="center">
                          <Badge mr={2} colorScheme="gray" variant="subtle">
                            {getAllHighlights().length} total highlights
                          </Badge>
                          <Input 
                            size="xs" 
                            placeholder="Search in sentences..." 
                            value={searchText}
                            onChange={(e) => setSearchText(e.target.value)}
                            width="150px"
                            mr={1}
                          />
                          <IconButton
                            icon={<FaSearch />}
                            size="xs"
                            aria-label="Search"
                            isDisabled={!searchText}
                            onClick={() => setSearchText('')}
                          />
                        </Flex>
                      </Flex>
                    </Box>
                    
                    <List spacing={2} maxH="300px" overflowY="auto">
                      {getFilteredHighlights().map((highlight, idx) => {
                        const citationIndex = getCitationIndex(highlight);
                        return (
                          <ListItem 
                            key={idx}
                            p={2}
                            borderRadius="md"
                            bg={highlight === selectedHighlight ? selectedHighlightColor : highlightColor}
                            cursor="pointer"
                            _hover={{ opacity: 0.8 }}
                            onClick={() => handleHighlightClick(highlight)}
                            position="relative"
                          >
                            <Flex>
                              <ListIcon as={FaQuoteRight} color="brand.500" mt={1} />
                              <Text fontSize="sm">{highlight.text}</Text>
                            </Flex>
                            <Flex justify="space-between" mt={1}>
                              <Text fontSize="xs" color="gray.500">
                                Document {highlight.docIndex + 1}
                              </Text>
                              <Flex>
                                {citationIndex !== -1 && (
                                  <Tag size="sm" colorScheme="blue" mr={1}>
                                    <TagLeftIcon as={FaLink} boxSize="10px" />
                                    <TagLabel>[{citationIndex + 1}]</TagLabel>
                                  </Tag>
                                )}
                                <Tooltip label="Click to see where this appears in the answer">
                                  <Badge colorScheme="brand">Trace</Badge>
                                </Tooltip>
                              </Flex>
                            </Flex>
                          </ListItem>
                        );
                      })}
                      
                      {getFilteredHighlights().length === 0 && (
                        <Box textAlign="center" py={4}>
                          <Text fontSize="sm" color="gray.500">
                            {searchText ? "No matching sentences found" : "No relevant sentences found"}
                          </Text>
                        </Box>
                      )}
                    </List>
                  </Box>
                </Flex>
              </MotionBox>
            </VStack>
          )}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </Box>
      
      {/* Input */}
      <Box p={4} borderTopWidth="1px" borderColor={borderColor}>
        <form onSubmit={handleSubmit}>
          <Flex>
            <FormControl>
              <Input
                placeholder={
                  isResourcesLoaded
                    ? "Ask a question..."
                    : "Load resources first"
                }
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                isDisabled={!isResourcesLoaded || isLoading}
              />
            </FormControl>
            <Button
              ml={2}
              colorScheme="brand"
              type="submit"
              isLoading={isLoading}
              isDisabled={!isResourcesLoaded || !question.trim()}
              leftIcon={<FaPaperPlane />}
            >
              Send
            </Button>
          </Flex>
        </form>
      </Box>
    </MotionBox>
  );
};

export default ChatPanel; 