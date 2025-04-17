import React from 'react';
import { View, Text, ScrollView } from 'react-native';
import { ScriptPrompt } from '@/types';
import { useThemeColor } from '@/hooks/useThemeColor'; // Import useThemeColor
import { MaterialCommunityIcons } from '@expo/vector-icons'; // Import icons

interface PromptDisplayProps {
  prompt: ScriptPrompt | null;
  currentNumber: number;
  totalNumber: number;
  containerClassName?: string;
  promptTextClassName?: string;
  infoTextClassName?: string;
  typeTextClassName?: string;
  meaningTextClassName?: string;
}

export const PromptDisplay: React.FC<PromptDisplayProps> = ({
  prompt,
  currentNumber,
  totalNumber,
  containerClassName = "p-5 rounded-xl my-4 w-full items-center",
  // Adjust default prompt text style slightly if needed, maybe slightly smaller for spontaneous
  promptTextClassName = "text-3xl text-neutral-900 dark:text-neutral-100 leading-tight my-3 text-center font-medium",
  infoTextClassName = "text-center text-sm text-neutral-500 dark:text-neutral-400 mb-1",
  typeTextClassName = "text-center text-xs font-semibold uppercase tracking-wider text-primary dark:text-primary-light mb-3",
  meaningTextClassName = "text-base text-neutral-500 dark:text-neutral-400 italic mt-2 text-center px-2",
}) => {
  // --- Get themed colors ---
  const themedTextColor = useThemeColor({}, 'text');
  const themedSecondaryTextColor = useThemeColor({}, 'secondaryText'); // Assuming you have this or use neutral-500/400
  const instructionBgColor = useThemeColor({ light: '#E0E7FF', dark: '#312E81' }, 'background'); // Example: primary-light/dark variant
  const instructionBorderColor = useThemeColor({ light: '#A5B4FC', dark: '#6366F1' }, 'border'); // Example: primary variant
  const instructionTextColor = useThemeColor({ light: '#3730A3', dark: '#E0E7FF' }, 'text'); // Example: primary-dark/light variant

  if (!prompt) {
    return (
      <View className={containerClassName}>
        <Text className={infoTextClassName} style={{ color: themedTextColor }}>Loading prompt...</Text>
      </View>
    );
  }

  const isSpontaneous = prompt.type === 'spontaneous';
  const displayType = prompt.type.replace(/_/g, ' ');

  // --- Conditional container styling ---
  const spontaneousContainerStyle = isSpontaneous
    ? "bg-neutral-50 dark:bg-neutral-800/50 border-l-4 border-primary dark:border-primary-light shadow-md"
    : "";
  const finalContainerClass = `${containerClassName} ${spontaneousContainerStyle}`;

  // --- Split spontaneous prompt text if needed (Optional, basic split shown) ---
  // This assumes English instructions come first, then Twi in parentheses. Adjust if needed.
  let spontaneousTopic = prompt.text;
  let spontaneousGuidance = '';
  if (isSpontaneous) {
    const match = prompt.text.match(/^(.*?)\s*\((.*)\)$/s); // Try to split "Topic (Guidance)"
    if (match) {
      spontaneousTopic = match[1].trim();
      spontaneousGuidance = `(${match[2].trim()})`; // Keep parentheses for Twi part
    }
  }


  return (
    <View className={finalContainerClass}>
      {/* --- Prompt Type and Number --- */}
      <View className='-mt-4 mb-4 self-stretch items-center'>
        <Text className={typeTextClassName}>
          {displayType}
        </Text>
        <Text className={infoTextClassName} style={{ color: themedSecondaryTextColor || '#6B7280' }}>
          Prompt {currentNumber} / {totalNumber}
        </Text>
      </View>

      {/* --- NEW: Spontaneous Instructions Box --- */}
      {isSpontaneous && (
        <View
          className="mb-5 p-3 rounded-lg border w-full"
          style={{ backgroundColor: instructionBgColor, borderColor: instructionBorderColor }}
        >
          <View className='flex-row items-center justify-center mb-1'>
            <MaterialCommunityIcons name="information-outline" size={18} color={instructionTextColor} />
            <Text
              className="ml-2 font-bold text-center"
              style={{ color: instructionTextColor }}
            >
              INSTRUCTION
            </Text>
          </View>
          <Text
            className="text-sm text-center"
            style={{ color: instructionTextColor }}
          >
            Read the topic below, then tap Record and speak naturally about it in Twi.
          </Text>
          <Text
            className="text-sm font-bold text-center mt-1"
            style={{ color: instructionTextColor }}
          >
            Do not read the text below aloud.
          </Text>
          {/* Twi Instructions */}
          <Text
            className="text-xs text-center italic mt-2"
            style={{ color: instructionTextColor, opacity: 0.8 }}
          >
            (Kenkan asɛmti a ɛwɔ aseɛ yi, afei mia Record na kasa fa ho wɔ Twi kasa mu. Nkenkan nsɛm a ɛwɔ aseɛ yi.)
          </Text>
        </View>
      )}

      {/* --- Main Prompt Text Area --- */}
      <ScrollView
        style={{ flexShrink: 1, width: '100%' }}
        contentContainerStyle={{ flexGrow: 1, justifyContent: 'center', alignItems: 'center' }}
        showsVerticalScrollIndicator={false}
      >
        {/* Conditionally style spontaneous prompt differently if desired */}
        <Text
          className={`${promptTextClassName} ${isSpontaneous ? 'text-2xl' : ''}`} // Slightly smaller for spontaneous
          style={{ color: themedTextColor }}
        >
          {/* Display only the parsed topic for spontaneous, or full text for scripted */}
          {isSpontaneous ? spontaneousTopic : prompt.text}
        </Text>

        {/* Display Twi guidance part for spontaneous (optional) */}
        {isSpontaneous && spontaneousGuidance && (
          <Text
            className="text-lg italic mt-2 text-center px-2"
            style={{ color: themedSecondaryTextColor || '#6B7280' }}
          >
            {spontaneousGuidance}
          </Text>
        )}

        {/* Meaning/Translation Text (Only for scripted prompts) */}
        {!isSpontaneous && prompt.meaning && (
          <Text
            className={meaningTextClassName}
            style={{ color: themedSecondaryTextColor || '#6B7280' }}
          >
            ({prompt.meaning})
          </Text>
        )}
      </ScrollView>
    </View>
  );
}
