import React from 'react';
import { View, Text } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Button } from './Button';
import { useThemeColor } from '@/hooks/useThemeColor';

interface SectionIntroProps {
  sectionTitle: string;
  sectionDescription: string;
  sectionNumber: number;
  totalSections: number;
  onStartSection: () => void;
  iconName?: keyof typeof MaterialCommunityIcons.glyphMap; // Optional icon
}

export const SectionIntro: React.FC<SectionIntroProps> = ({
  sectionTitle,
  sectionDescription,
  sectionNumber,
  totalSections,
  onStartSection,
  iconName = "play-box-outline" // Default icon
}) => {
  const primaryColor = useThemeColor({}, 'tint');
  const textColor = useThemeColor({}, 'text');
  const secondaryTextColor = useThemeColor({ light: '#4B5563', dark: '#9CA3AF' }, 'text'); // Slightly darker secondary
  const cardBg = useThemeColor({ light: 'white', dark: '#1F2937' }, 'card');

  return (
    <View className="flex-1 justify-center items-center p-6 bg-neutral-100 dark:bg-neutral-900">
      <View
        className="bg-white dark:bg-neutral-800 p-8 rounded-2xl shadow-xl items-center w-full max-w-md"
        style={{ backgroundColor: cardBg }}
      >
        {/* ... Icon, Title, Description ... */}
        <MaterialCommunityIcons name={iconName} size={64} color={primaryColor} />
        <Text className="text-sm font-semibold text-neutral-500 dark:text-neutral-400 mt-4 mb-1 uppercase tracking-wider">
          Section {sectionNumber} / {totalSections}
        </Text>
        <Text className="text-3xl font-bold text-center mb-3" style={{ color: textColor }}>
          {sectionTitle}
        </Text>
        <Text className="text-base text-center mb-8" style={{ color: secondaryTextColor }}>
          {sectionDescription}
        </Text>

        {/* --- CORRECTED BUTTON --- */}
        <Button
          title="Start Section"
          icon="arrow-right-circle"
          iconPosition='right'
          onPress={onStartSection}
          className="bg-primary dark:bg-primary-dark w-full flex-row items-center py-3.5 rounded-xl shadow-md px-12"// Keep justify-center here
          // REMOVED the negative margin hack below
          textClassName="text-white text-lg font-semibold dark:text-gray-100"
          iconColor="white"
        />
      </View>
    </View>
  );
};
