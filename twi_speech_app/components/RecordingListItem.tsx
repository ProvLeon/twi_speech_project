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
  isUploadingNow?: boolean; // Is this specific item currently uploading?
  showParticipantCode?: boolean;
}

const formatDuration = (ms?: number): string => {
  // ... (keep existing formatDuration function)
  if (ms === undefined || ms === null || ms < 0) return '--:--'; // Handle negative just in case
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
  isUploadingNow = false, // Default to false
  showParticipantCode = false,
}) => {
  // --- Themed Colors ---
  const textColor = useThemeColor({}, 'text');
  const secondaryTextColor = useThemeColor({ light: '#6B7280', dark: '#9CA3AF' }, 'text');
  const iconColor = useThemeColor({}, 'icon'); // General icon color
  const playButtonBg = useThemeColor({ light: 'rgba(79, 70, 229, 0.1)', dark: 'rgba(99, 102, 241, 0.2)' }, 'background');
  const dangerColor = useThemeColor({}, 'danger');
  const primaryColor = useThemeColor({}, 'tint');
  const successColor = useThemeColor({}, 'success');
  const warningColor = useThemeColor({}, 'warning');
  const disabledColor = useThemeColor({ light: '#9CA3AF', dark: '#6B7280' }, 'icon'); // Color for disabled elements/text
  const backgroundColor = useThemeColor({ light: '#FFFFFF', dark: '#1F2937' }, 'card'); // Use card color


  // --- Determine Status ---
  let statusIcon: keyof typeof MaterialCommunityIcons.glyphMap | null = null; // Use null for ActivityIndicator case
  let statusColor = warningColor; // Default: Pending
  let statusText = 'Pending Upload';
  let statusTextColor = warningColor; // Default text color

  if (isUploadingNow) {
    statusIcon = null; // Will render ActivityIndicator instead
    statusColor = primaryColor;
    statusText = 'Uploading...';
    statusTextColor = primaryColor;
  } else if (recording.uploadStatus === 'failed') {
    statusIcon = "alert-circle-outline";
    statusColor = dangerColor;
    statusText = 'Upload Failed';
    statusTextColor = dangerColor;
  } else if (recording.uploaded) {
    statusIcon = "cloud-check-outline";
    statusColor = successColor;
    statusText = 'Uploaded';
    statusTextColor = successColor;
  }
  // else: Keep default pending values

  // --- Determine Playability & Button State ---
  const canPlay = !recording.uploaded && recording.uploadStatus !== 'failed' && !isUploadingNow && !isDeleting;
  const playIcon = isPlaying ? "pause-circle" : "play-circle";
  const playIconColor = canPlay ? primaryColor : disabledColor;
  const playButtonOpacity = canPlay ? 1 : 0.5;

  return (
    <View
      className={`
        rounded-xl border
        overflow-hidden my-1.5 mx-1 transition-opacity duration-300
        ${isDeleting ? 'opacity-40' : 'opacity-100'}
        ${recording.uploadStatus === 'failed' ? 'border-danger/50 dark:border-danger/40' : 'border-neutral-200 dark:border-neutral-700'}
      `}
      style={[styles.itemContainer, { backgroundColor }]} // Apply themed background
    >
      <View className="flex-row items-center p-3 space-x-3">
        {/* Play/Pause Button */}
        <TouchableOpacity
          onPress={() => canPlay && onPlay(recording.localUri)}
          disabled={!canPlay} // Disable based on combined state
          style={{
            backgroundColor: playButtonBg,
            opacity: playButtonOpacity, // Dim if cannot play
            width: 40,
            height: 40,
            alignItems: 'center',
            justifyContent: 'center',
            borderRadius: 20,
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
            {/* Use promptText if available, otherwise format promptId */}
            {recording.promptText || recording.promptId.replace(/_/g, ' ')}
          </Text>

          {/* Sub-Info Row */}
          <View className="flex-row items-center flex-wrap gap-x-3 gap-y-1">
            {/* Status */}
            <View className="flex-row items-center">
              {isUploadingNow ? (
                <ActivityIndicator size="small" color={statusColor} style={{ marginRight: 4 }} />
              ) : (
                statusIcon && ( // Only render icon if not uploading
                  <MaterialCommunityIcons
                    name={statusIcon}
                    size={16}
                    color={statusColor}
                    style={{ marginRight: 4 }}
                  />
                )
              )}
              <Text style={{
                color: statusTextColor,
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
                color={secondaryTextColor}
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
                backgroundColor: useThemeColor({ light: '#E5E7EB', dark: '#374151' }, 'background'), // Use a neutral background
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
                  {/* Simplify code display if needed */}
                  {recording.participantCode.replace('TWI_Speaker_', 'P')}
                </Text>
              </View>
            )}
          </View>
        </View>

        {/* Delete Button */}
        <TouchableOpacity
          onPress={() => onDelete(recording.id, recording.promptId)}
          disabled={isDeleting || isUploadingNow} // Also disable delete while uploading THIS item
          style={{
            padding: 8,
            alignItems: 'center',
            justifyContent: 'center',
            opacity: (isDeleting || isUploadingNow) ? 0.4 : 1, // Dim if disabled
          }}
          activeOpacity={0.6}
        >
          {isDeleting ? (
            <ActivityIndicator size="small" color={dangerColor} />
          ) : (
            <MaterialCommunityIcons
              name="delete-outline"
              size={24}
              color={dangerColor}
            />
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
};

// --- Styles ---
const styles = StyleSheet.create({
  itemContainer: {
    // Removed background color here, applied dynamically with themed color
    shadowColor: '#000', // Keep shadows for depth
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.15,
    shadowRadius: 1.5,
    elevation: 2, // Android shadow
  },
});
