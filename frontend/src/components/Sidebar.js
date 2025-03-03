import React, { useState } from 'react';
import {
  Box,
  Button,
  Divider,
  Flex,
  FormControl,
  FormLabel,
  Heading,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Switch,
  Text,
  VStack,
  useColorModeValue,
  useDisclosure,
} from '@chakra-ui/react';
import { FaCog, FaEye, FaEyeSlash, FaSync } from 'react-icons/fa';
import { motion } from 'framer-motion';
import { useApi } from '../contexts/ApiContext';
import { useSidebar } from '../contexts/SidebarContext';

const MotionBox = motion(Box);

const Sidebar = () => {
  const { isOpen, onToggle } = useSidebar();
  const { 
    isLoading, 
    isResourcesLoaded, 
    numDocs, 
    updateNumDocs, 
    loadResources,
    resetQuery,
    refreshStatus 
  } = useApi();
  
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  const handleLoadResources = async () => {
    await loadResources(apiKey || null);
  };
  
  const handleNewConversation = () => {
    resetQuery();
  };
  
  return (
    <MotionBox
      position="fixed"
      top="80px"
      left={0}
      bottom={0}
      width="300px"
      bg={bgColor}
      borderRightWidth="1px"
      borderColor={borderColor}
      zIndex={10}
      initial={false}
      animate={{
        x: isOpen ? 0 : -300,
        boxShadow: isOpen ? '6px 0px 20px rgba(0, 0, 0, 0.1)' : 'none',
      }}
      transition={{ duration: 0.3 }}
      p={4}
      overflowY="auto"
    >
      <VStack spacing={6} align="stretch">
        {/* System Status */}
        <Box>
          <Heading size="sm" mb={2}>
            System Status
          </Heading>
          <Flex align="center" justify="space-between">
            <Text fontSize="sm">
              Resources Loaded:
            </Text>
            <Flex align="center">
              <Box
                w={3}
                h={3}
                borderRadius="full"
                bg={isResourcesLoaded ? 'green.400' : 'red.400'}
                mr={2}
              />
              <Text fontSize="sm">
                {isResourcesLoaded ? 'Yes' : 'No'}
              </Text>
            </Flex>
          </Flex>
        </Box>
        
        <Divider />
        
        {/* API Key */}
        <FormControl>
          <FormLabel fontSize="sm">OpenAI API Key</FormLabel>
          <InputGroup size="sm">
            <Input
              type={showApiKey ? 'text' : 'password'}
              placeholder="Enter your API key"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
            <InputRightElement>
              <IconButton
                icon={showApiKey ? <FaEyeSlash /> : <FaEye />}
                variant="ghost"
                size="sm"
                onClick={() => setShowApiKey(!showApiKey)}
                aria-label={showApiKey ? 'Hide API key' : 'Show API key'}
              />
            </InputRightElement>
          </InputGroup>
          <Text fontSize="xs" color="gray.500" mt={1}>
            If not provided, will use environment variable
          </Text>
        </FormControl>
        
        {/* Load Resources */}
        <Button
          leftIcon={<FaSync />}
          colorScheme="brand"
          isLoading={isLoading}
          onClick={handleLoadResources}
          size="sm"
        >
          Load Resources
        </Button>
        
        <Divider />
        
        {/* Settings */}
        <Box>
          <Heading size="sm" mb={3}>
            Settings
          </Heading>
          
          <FormControl mb={3}>
            <FormLabel fontSize="sm">Number of Documents</FormLabel>
            <NumberInput
              size="sm"
              min={1}
              max={10}
              value={numDocs}
              onChange={(_, value) => updateNumDocs(value)}
            >
              <NumberInputField />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
          </FormControl>
        </Box>
        
        <Divider />
        
        {/* Actions */}
        <Box>
          <Heading size="sm" mb={3}>
            Actions
          </Heading>
          
          <Button
            colorScheme="gray"
            size="sm"
            width="full"
            onClick={handleNewConversation}
            mb={2}
          >
            New Conversation
          </Button>
        </Box>
      </VStack>
    </MotionBox>
  );
};

export default Sidebar; 