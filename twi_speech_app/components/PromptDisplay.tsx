import React from 'react';
import { View, Text, ScrollView } from 'react-native';
import { ScriptPrompt } from '@/types';

interface PromptDisplayProps {
  prompt: ScriptPrompt | null;
  currentNumber: number;
  totalNumber: number;
  containerClassName?: string;
  promptTextClassName?: string;
  infoTextClassName?: string;
  typeTextClassName?: string;
  meaningTextClassName?: string; // Added prop for styling the meaning text
}

export const PromptDisplay: React.FC<PromptDisplayProps> = ({
  prompt,
  currentNumber,
  totalNumber,
  containerClassName = "p-5 rounded-xl my-4 w-full items-center",
  promptTextClassName = "text-3xl text-neutral-800 leading-tight my-3 text-center font-semibold",
  infoTextClassName = "text-center text-sm text-neutral-500 mb-1",
  typeTextClassName = "text-center text-xs font-semibold uppercase tracking-wider text-primary mb-3",
  meaningTextClassName = "text-base text-neutral-500 italic mt-2 text-center px-2", // Default styling for meaning
}) => {
  if (!prompt) {
    // Optionally render a placeholder or loading state if needed
    return (
      <View className={containerClassName}>
        <Text className={infoTextClassName}>Loading prompt...</Text>
      </View>
    );
  }
  const displayType = prompt.type.replace(/_/g, ' ');

  return (
    <View className={containerClassName}>
      <View className='-mt-24 mb-36'>

        <Text className={`${typeTextClassName}`}>
          {displayType}
        </Text>
        <Text className={infoTextClassName}>
          Prompt {currentNumber} / {totalNumber}
        </Text>
      </View>
      {/* ScrollView ensures long prompts don't overflow */}
      <ScrollView
        style={{ flexShrink: 1, width: '100%' }} // Ensure ScrollView takes width
        contentContainerStyle={{ flexGrow: 1, justifyContent: 'center', alignItems: 'center' }} // Center content
        showsVerticalScrollIndicator={false} // Hide scrollbar unless needed
      >
        {/* Main Prompt Text */}
        <Text className={promptTextClassName}>
          {prompt.text}
        </Text>

        {/* Meaning/Translation Text (Conditionally Rendered) */}
        {prompt.meaning && ( // Only render if meaning exists
          <Text className={meaningTextClassName}>
            ({prompt.meaning}) {/* Wrap in parentheses */}
          </Text>
        )}
      </ScrollView>
    </View>
  );
};
