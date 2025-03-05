import React, { useEffect, useRef, useState } from 'react';
import {
  Box,
  Text,
  useColorModeValue,
  Spinner,
  Flex,
  Badge,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Button,
  Collapse,
  IconButton,
} from '@chakra-ui/react';
import { FaChevronDown, FaChevronUp, FaExpand, FaCompress } from 'react-icons/fa';

const DocumentViewer = ({ document: docData, selectedHighlightIndex, isLoading }) => {
  const contentRef = useRef(null);
  const highlightRefs = useRef([]);
  const [showFullContent, setShowFullContent] = useState(false);
  
  const highlightColor = useColorModeValue('yellow.200', 'yellow.700');
  const selectedHighlightColor = useColorModeValue('orange.300', 'orange.600');
  const textColor = useColorModeValue('gray.800', 'gray.100');
  
  // Apply highlights to the document content
  useEffect(() => {
    if (!contentRef.current || !docData) return;
    
    const content = docData.content;
    
    // If there are no highlights, just show the plain content
    if (!docData.highlights || docData.highlights.length === 0) {
      contentRef.current.textContent = content;
      return;
    }
    
    const highlights = [...docData.highlights].sort((a, b) => a.start - b.start);
    
    // Create a new HTML content with highlights
    let html = '';
    let lastIndex = 0;
    
    // Reset highlight refs array
    highlightRefs.current = new Array(highlights.length);
    
    highlights.forEach((highlight, index) => {
      // Add text before the highlight
      html += content.substring(lastIndex, highlight.start);
      
      // Add the highlighted text with a data attribute for the index
      const isSelected = index === selectedHighlightIndex;
      const bgColor = isSelected ? selectedHighlightColor : highlightColor;
      const borderStyle = isSelected ? 'solid 2px orange' : 'none';
      
      html += `<mark 
        id="highlight-${index}" 
        class="highlight ${isSelected ? 'selected' : ''}" 
        style="background-color: ${bgColor}; padding: 0 2px; border-radius: 2px; border: ${borderStyle};"
        data-index="${index}"
      >${highlight.text}</mark>`;
      
      // Update the last index
      lastIndex = highlight.end;
    });
    
    // Add any remaining text
    html += content.substring(lastIndex);
    
    // Set the HTML content
    contentRef.current.innerHTML = html;
    
    // Scroll to selected highlight if specified
    if (selectedHighlightIndex !== undefined && selectedHighlightIndex !== null) {
      setTimeout(() => {
        // Use window.document instead of document to avoid the naming conflict
        const selectedElement = window.document.getElementById(`highlight-${selectedHighlightIndex}`);
        if (selectedElement) {
          selectedElement.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
          });
          setShowFullContent(true); // Auto-expand when a highlight is selected
        }
      }, 100);
    }
  }, [docData, highlightColor, selectedHighlightColor, selectedHighlightIndex]);
  
  if (!docData) {
    return <Text>No document selected</Text>;
  }
  
  const hasHighlights = docData.highlights && docData.highlights.length > 0;
  
  return (
    <Box>
      {isLoading && (
        <Flex justify="flex-end" mb={2}>
          <Badge colorScheme="blue" display="flex" alignItems="center" p={1}>
            <Spinner size="xs" mr={1} />
            <Text fontSize="xs">Loading highlights...</Text>
          </Badge>
        </Flex>
      )}
      
      {!hasHighlights && !isLoading && (
        <Alert status="info" mb={3} borderRadius="md">
          <AlertIcon />
          <Box>
            <AlertTitle fontSize="sm">No highlights found</AlertTitle>
            <AlertDescription fontSize="xs">
              This document doesn't contain any passages relevant to the query.
            </AlertDescription>
          </Box>
        </Alert>
      )}
      
      {/* Document summary */}
      <Box 
        mb={3} 
        p={3} 
        borderWidth="1px" 
        borderRadius="md" 
        borderColor={hasHighlights ? "green.200" : "gray.200"} 
        bg={hasHighlights ? "green.50" : "gray.50"}
        position="relative"
      >
        <Flex justify="space-between" align="center" mb={2}>
          <Text fontSize="sm" fontWeight="bold" color={hasHighlights ? "green.700" : "gray.700"}>
            Document Summary
          </Text>
          <IconButton
            icon={showFullContent ? <FaCompress /> : <FaExpand />}
            size="xs"
            aria-label={showFullContent ? "Collapse" : "Expand"}
            onClick={() => setShowFullContent(!showFullContent)}
            variant="ghost"
          />
        </Flex>
        
        <Text fontSize="sm" color="gray.700" mb={hasHighlights ? 2 : 0}>
          {docData.content.length > 300 
            ? docData.content.substring(0, 300) + '...' 
            : docData.content}
        </Text>
        
        {hasHighlights && (
          <Box>
            <Text fontSize="xs" fontWeight="bold" mt={3} mb={1} color="green.700">
              Relevant Passages ({docData.highlights.length}):
            </Text>
            <Flex mt={1} wrap="wrap" gap={1}>
              {docData.highlights.map((highlight, idx) => (
                <Badge 
                  key={idx} 
                  colorScheme="yellow" 
                  fontSize="xs"
                  p={1}
                  cursor="pointer"
                  onClick={() => {
                    setShowFullContent(true);
                    setTimeout(() => {
                      // Use window.document instead of document to avoid the naming conflict
                      const selectedElement = window.document.getElementById(`highlight-${idx}`);
                      if (selectedElement) {
                        selectedElement.scrollIntoView({ 
                          behavior: 'smooth', 
                          block: 'center' 
                        });
                      }
                    }, 100);
                  }}
                >
                  {highlight.text.length > 40 
                    ? highlight.text.substring(0, 40) + '...' 
                    : highlight.text}
                </Badge>
              ))}
            </Flex>
          </Box>
        )}
        
        <Button 
          size="xs" 
          width="100%" 
          mt={3}
          onClick={() => setShowFullContent(!showFullContent)}
          rightIcon={showFullContent ? <FaChevronUp /> : <FaChevronDown />}
          variant="outline"
          colorScheme={hasHighlights ? "green" : "gray"}
        >
          {showFullContent ? "Hide Full Content" : "Show Full Content"}
        </Button>
      </Box>
      
      {/* Full document content */}
      <Collapse in={showFullContent} animateOpacity>
        <Box
          ref={contentRef}
          whiteSpace="pre-wrap"
          color={textColor}
          fontSize="sm"
          fontFamily="mono"
          p={3}
          borderWidth="1px"
          borderColor={isLoading ? "blue.200" : hasHighlights ? "green.200" : "gray.200"}
          borderRadius="md"
          position="relative"
          maxHeight="calc(100vh - 350px)"
          overflowY="auto"
          css={{
            '.highlight': {
              transition: 'background-color 0.3s, border 0.3s',
            },
            '.highlight.selected': {
              boxShadow: '0 0 8px rgba(255, 165, 0, 0.5)',
            }
          }}
        />
      </Collapse>
      
      {docData.highlights && docData.highlights.length === 0 && isLoading && (
        <Flex justify="center" mt={4}>
          <Badge colorScheme="yellow">
            Searching for relevant passages...
          </Badge>
        </Flex>
      )}
    </Box>
  );
};

export default DocumentViewer; 