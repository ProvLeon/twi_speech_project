import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';

interface RecordingIndicatorProps {
  isRecording: boolean;
}

// Simple pulsing effect can be added with Reanimated later
export const RecordingIndicator: React.FC<RecordingIndicatorProps> = ({ isRecording }) => {
  if (!isRecording) {
    return null; // Don't render anything if not recording
  }

  return (
    <View className="flex-row items-center justify-center my-2 p-2 bg-red-100 rounded-full">
      <MaterialCommunityIcons name="record-circle" size={20} color="#EF4444" />
      <Text className="ml-2 text-red-600 font-semibold">Recording...</Text>
    </View>
  );
};
