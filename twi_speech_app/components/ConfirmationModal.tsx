import React, { memo } from 'react';
import { View, Text, StyleSheet, Platform } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Button } from './Button';
import { useThemeColor } from '@/hooks/useThemeColor';
import { ThemedView } from './ThemedView';

export interface ConfirmationModalContentProps {
  title: string;
  message: string;
  options: Array<{
    text: string;
    onPress: () => void;
    style?: 'primary' | 'secondary' | 'destructive';
    icon?: keyof typeof MaterialCommunityIcons.glyphMap | null;
  }>;
  iconName?: keyof typeof MaterialCommunityIcons.glyphMap;
  iconsPosition?: ""
}

export const ConfirmationModalContent = memo(({
  title,
  message,
  options,
  iconName = "check-circle-outline",
  iconsPosition
}: ConfirmationModalContentProps) => {
  const cardBg = useThemeColor({ light: 'white', dark: '#1F2937' }, 'card');
  const textColor = useThemeColor({}, 'text');
  const secondaryTextColor = useThemeColor({ light: '#4B5563', dark: '#9CA3AF' }, 'text');
  const primaryColor = useThemeColor({}, 'tint');

  return (
    <ThemedView style={[
      styles.container,
      { backgroundColor: cardBg }
    ]}>
      <View style={styles.content}>
        {/* Icon */}
        <View style={styles.iconContainer}>
          <MaterialCommunityIcons
            name={iconName}
            size={56}
            color={primaryColor}
          />
        </View>

        {/* Title */}
        <Text style={[styles.title, { color: textColor }]}>
          {title}
        </Text>

        {/* Message */}
        <Text style={[styles.message, { color: secondaryTextColor }]}>
          {message}
        </Text>

        {/* Buttons */}
        <View style={styles.buttonContainer}>
          {options.map((option, index) => (
            <Button
              key={index}
              title={option.text}
              onPress={option.onPress}
              icon={option.icon}
              iconsPosition={option.iconsPosition}
              className={`
                w-full flex-row items-center justify-center py-3.5 rounded-lg mb-3
                ${option.style === 'primary' ? 'bg-primary dark:bg-primary-dark' : ''}
                ${option.style === 'destructive' ? 'bg-danger dark:bg-danger-dark' : ''}
                ${option.style === 'secondary' ? 'bg-neutral-200 dark:bg-neutral-700' : ''}
              `}
              textClassName={`
                font-semibold text-lg ml-2
                ${option.style === 'primary' || option.style === 'destructive' ? 'text-white' : ''}
                ${option.style === 'secondary' ? 'text-neutral-800 dark:text-neutral-100' : ''}
              `}
              iconColor={option.style === 'primary' || option.style === 'destructive' ? 'white' : textColor}
              iconSize={24}
            />
          ))}
        </View>
      </View>
    </ThemedView>
  );
});

const styles = StyleSheet.create({
  container: {
    borderRadius: 16,
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.25,
        shadowRadius: 4,
      },
      android: {
        elevation: 5,
      },
    }),
  },
  content: {
    padding: 24,
  },
  iconContainer: {
    alignItems: 'center',
    marginBottom: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 8,
  },
  message: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 24,
  },
  buttonContainer: {
    width: '100%',
  },
});
