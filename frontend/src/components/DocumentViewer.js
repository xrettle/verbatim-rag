import React, { useEffect, useRef } from 'react';
import {
  Box,
  Text,
  useColorModeValue,
} from '@chakra-ui/react';

const DocumentViewer = ({ document, selectedHighlightIndex }) => {
  const contentRef = useRef(null);
  const highlightRefs = useRef([]);
  
  const highlightColor = useColorModeValue('yellow.200', 'yellow.700');
  const selectedHighlightColor = useColorModeValue('orange.300', 'orange.600');
  const textColor = useColorModeValue('gray.800', 'gray.100');
  
  // Apply highlights to the document content
  useEffect(() => {
    if (!contentRef.current || !document || !document.highlights) return;
    
    const content = document.content;
    const highlights = [...document.highlights].sort((a, b) => a.start - b.start);
    
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
        const selectedElement = document.getElementById(`highlight-${selectedHighlightIndex}`);
        if (selectedElement) {
          selectedElement.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
          });
        }
      }, 100);
    }
  }, [document, highlightColor, selectedHighlightColor, selectedHighlightIndex]);
  
  if (!document) {
    return <Text>No document selected</Text>;
  }
  
  return (
    <Box>
      <Box
        ref={contentRef}
        whiteSpace="pre-wrap"
        color={textColor}
        fontSize="sm"
        fontFamily="mono"
        p={2}
        css={{
          '.highlight': {
            transition: 'background-color 0.3s, border 0.3s',
          },
          '.highlight.selected': {
            boxShadow: '0 0 8px rgba(255, 165, 0, 0.5)',
          }
        }}
      />
    </Box>
  );
};

export default DocumentViewer; 