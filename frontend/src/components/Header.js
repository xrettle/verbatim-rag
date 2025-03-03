import React from 'react';
import {
  Box,
  Flex,
  Heading,
  IconButton,
  useColorMode,
  useColorModeValue,
} from '@chakra-ui/react';
import { FaMoon, FaSun } from 'react-icons/fa';
import { motion } from 'framer-motion';

const MotionBox = motion(Box);

const Header = () => {
  const { colorMode, toggleColorMode } = useColorMode();
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <Box
      as="header"
      bg={bgColor}
      borderBottom="1px"
      borderColor={borderColor}
      py={3}
      px={4}
      position="sticky"
      top={0}
      zIndex={10}
      boxShadow="sm"
    >
      <Flex maxW="container.xl" mx="auto" align="center" justify="space-between">
        <Flex align="center">
          <MotionBox
            initial={{ rotate: -10 }}
            animate={{ rotate: 0 }}
            transition={{ duration: 0.5 }}
            mr={3}
          >
            <Box
              as="span"
              fontSize="2xl"
              fontWeight="bold"
              role="img"
              aria-label="Logo"
            >
              📚
            </Box>
          </MotionBox>
          <Heading
            as="h1"
            size="md"
            bgGradient="linear(to-r, brand.500, purple.500)"
            bgClip="text"
            fontWeight="bold"
          >
            Verbatim RAG
          </Heading>
        </Flex>

        <IconButton
          aria-label={`Switch to ${colorMode === 'light' ? 'dark' : 'light'} mode`}
          variant="ghost"
          color={colorMode === 'light' ? 'gray.600' : 'gray.400'}
          onClick={toggleColorMode}
          icon={colorMode === 'light' ? <FaMoon /> : <FaSun />}
          _hover={{
            bg: colorMode === 'light' ? 'gray.100' : 'gray.700',
          }}
        />
      </Flex>
    </Box>
  );
};

export default Header; 