import React from 'react';
import {
  Box,
  Flex,
  Text,
  Badge,
  useColorModeValue,
  Heading,
} from '@chakra-ui/react';
import { motion } from 'framer-motion';
import HighlightedText from './HighlightedText';
import { useDocuments } from '../contexts/DocumentsContext';

const MotionBox = motion(Box);

const MessageItem = ({ message, index }) => {
  const questionBg = useColorModeValue('gray.50', 'gray.700');
  const answerBg = useColorModeValue('blue.50', 'blue.900');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const { setSelectedDocId } = useDocuments();
  
  const handleFactClick = (factData) => {
    setSelectedDocId({ docIndex: factData.doc_id });
  };
  
  return (
    <MotionBox
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
    >
      {/* Question */}
      <Box
        p={3}
        bg={questionBg}
        borderRadius="md"
        mb={2}
        borderWidth="1px"
        borderColor={borderColor}
      >
        <Flex align="center" mb={1}>
          <Badge colorScheme="purple" mr={2}>
            Question
          </Badge>
          <Text fontSize="xs" color="gray.500">
            {new Date(message.timestamp).toLocaleTimeString()}
          </Text>
        </Flex>
        <Text fontWeight="medium">{message.question}</Text>
      </Box>
      
      {/* Answer */}
      <Box
        p={4}
        bg={answerBg}
        borderRadius="md"
        borderWidth="1px"
        borderColor={borderColor}
        position="relative"
        boxShadow="sm"
      >
        <Flex align="center" mb={2}>
          <Badge colorScheme="blue" mr={2}>
            Answer
          </Badge>
          <Text fontSize="xs" color="gray.500">
            {message.facts.length} facts found
          </Text>
        </Flex>
        
        <HighlightedText
          text={message.answer}
          facts={message.facts}
          onFactClick={handleFactClick}
        />
      </Box>
    </MotionBox>
  );
};

export default MessageItem; 