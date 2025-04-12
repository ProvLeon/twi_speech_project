import React from 'react';
import { View, Text, TouchableOpacity, ActivityIndicator, StyleSheet } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { RecordingMetadata } from '@/types';
import { useThemeColor } from '@/hooks/useThemeColor';

interface RecordingListItemProps {
  recording: RecordingMetadata;
  onPlay: (uri: string) => void;
  onDelete: (id: string, promptId: string) => void;
  isPlaying: boolean;
  isDeleting?: boolean;
  showParticipantCode?: boolean;
}

const formatDuration = (ms?: number): string => {
  if (ms === undefined || ms === null || ms <= 0) return '--:--';
  const totalSeconds = Math.round(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
};

export const RecordingListItem: React.FC<RecordingListItemProps> = ({
  recording,
  onPlay,
  onDelete,
  isPlaying,
  isDeleting = false,
  showParticipantCode = false,
}) => {
  const playIcon = isPlaying ? "pause-circle" : "play-circle";
  const statusIcon = recording.uploaded ? "cloud-check-outline" : "clock-outline";
  const statusColor = recording.uploaded ? "#10B981" : "#F59E0B"; // Green / Amber
  const statusText = recording.uploaded ? 'Uploaded' : 'Pending';
  const statusColorClass = recording.uploaded ? "text-success" : "text-warning-dark dark:text-warning";

  // Use themed colors for consistency
  const textColor = useThemeColor({}, 'text');
  const secondaryTextColor = useThemeColor({ light: '#6B7280', dark: '#9CA3AF' }, 'text');
  const iconColor = useThemeColor({}, 'icon');
  const playButtonBg = useThemeColor({ light: 'rgba(79, 70, 229, 0.1)', dark: 'rgba(99, 102, 241, 0.2)' }, 'background');
  const playIconColor = recording.uploaded
    ? useThemeColor({ light: '#9CA3AF', dark: '#6B7280' }, 'icon')
    : useThemeColor({ light: '#4F46E5', dark: '#C7D2FE' }, 'tint');
  const backgroundColor = useThemeColor({ light: '#F3F4F6', dark: '#374151' }, 'background');

  return (
    <View
      className={`
        bg-white rounded-xl border border-neutral-200
        dark:bg-neutral-800 dark:border-neutral-700
        overflow-hidden my-1.5 mx-1 transition-opacity duration-300
        ${isDeleting ? 'opacity-40' : 'opacity-100'}
      `}
      style={styles.itemContainer}
    >
      <View className="flex-row items-center p-3 space-x-3">
        {/* Play/Pause Button */}
        <TouchableOpacity
          onPress={() => onPlay(recording.localUri)}
          disabled={recording.uploaded ? true : isDeleting}
          style={{
            backgroundColor: playButtonBg,
            opacity: recording.uploaded ? 0.5 : 1,
            width: 40,
            height: 40,
            alignItems: 'center',
            justifyContent: 'center',
            borderRadius: 20,
            padding: 4
          }}
          activeOpacity={0.7}
        >
          <MaterialCommunityIcons
            name={playIcon}
            size={28}
            color={playIconColor}
          />
        </TouchableOpacity>

        {/* Info Section */}
        <View className="flex-1">
          {/* Prompt Text */}
          <Text
            className="text-base font-semibold mb-1"
            numberOfLines={2}
            style={{ color: textColor }}
          >
            {recording.promptText || recording.promptId.replace(/_/g, ' ')}
          </Text>

          {/* Sub-Info Row */}
          <View className="flex-row items-center flex-wrap gap-x-3 gap-y-1">
            {/* Status */}
            <View className="flex-row items-center">
              <MaterialCommunityIcons
                name={statusIcon}
                size={16}
                color={statusColor}
                style={{ marginRight: 4 }}
              />
              <Text style={{
                color: statusColor,
                fontSize: 12,
                fontWeight: '500'
              }}>
                {statusText}
              </Text>
            </View>

            {/* Duration */}
            <View className="flex-row items-center">
              <MaterialCommunityIcons
                name="timer-outline"
                size={16}
                color={iconColor}
                style={{ marginRight: 4 }}
              />
              <Text style={{
                color: secondaryTextColor,
                fontSize: 12
              }}>
                {formatDuration(recording.recordingDuration)}
              </Text>
            </View>

            {/* Participant Code (Conditional) */}
            {showParticipantCode && (
              <View style={{
                flexDirection: 'row',
                alignItems: 'center',
                backgroundColor: backgroundColor,
                paddingHorizontal: 6,
                paddingVertical: 2,
                borderRadius: 4
              }}>
                <MaterialCommunityIcons
                  name="account-outline"
                  size={14}
                  color={iconColor}
                  style={{ marginRight: 4 }}
                />
                <Text style={{
                  color: secondaryTextColor,
                  fontSize: 12
                }}>
                  {recording.participantCode.replace('TWI_Speaker_', '')}
                </Text>
              </View>
            )}
          </View>
        </View>

        {/* Delete Button */}
        <TouchableOpacity
          onPress={() => onDelete(recording.id, recording.promptId)}
          disabled={isDeleting}
          style={{
            padding: 8,
            alignItems: 'center',
            justifyContent: 'center'
          }}
          activeOpacity={0.6}
        >
          {isDeleting ? (
            <ActivityIndicator size="small" color="#EF4444" />
          ) : (
            <MaterialCommunityIcons
              name="delete-outline"
              size={24}
              color="#EF4444"
            />
          )}
        </TouchableOpacity>
      </View>
    </View >
  );
};

// Define styles with proper shadow properties
const styles = StyleSheet.create({
  itemContainer: {
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 1,
    },
    shadowOpacity: 0.15,
    shadowRadius: 1.5,
    elevation: 2, // for Android
  },
});
