import React from 'react';
import { SafeAreaView, type SafeAreaViewProps } from 'react-native-safe-area-context';

import { useThemeColor } from '@/hooks/useThemeColor'; // Use your existing hook

// Export the Props type if you need it elsewhere
export type ThemedSafeAreaViewProps = SafeAreaViewProps & {
  lightColor?: string;
  darkColor?: string;
};

export function ThemedSafeAreaView({
  style,
  lightColor,
  darkColor,
  ...rest // Use 'rest' to capture all other SafeAreaViewProps
}: ThemedSafeAreaViewProps) {
  // Use the 'background' color name from your Colors constant by default
  const backgroundColor = useThemeColor({ light: lightColor, dark: darkColor }, 'background');

  return (
    <SafeAreaView
      style={[{ backgroundColor }, style]} // Apply the themed background color
      {...rest} // Pass all other props down
    />
  );
}
