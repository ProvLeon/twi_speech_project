import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, FlatList, Alert, ActivityIndicator } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useThemeColor } from '@/hooks/useThemeColor';
import { ParticipantDetails } from '@/types';
import { deleteParticipantWithRecordings, getAllParticipants, getRecordingsForParticipant, saveParticipantDetails } from '@/lib/storage';
import { Button } from './Button';

interface ParticipantSelectorProps {
  currentParticipant: ParticipantDetails | null;
  onSelectParticipant: (participant: ParticipantDetails | null) => void;
  onCreateNewParticipant: () => void;
}

export const ParticipantSelector: React.FC<ParticipantSelectorProps> = ({
  currentParticipant,
  onSelectParticipant,
  onCreateNewParticipant,
}) => {
  const [participants, setParticipants] = useState<ParticipantDetails[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [deletingCode, setDeletingCode] = useState<string | null>(null);

  const textColor = useThemeColor({}, 'text');
  const backgroundColor = useThemeColor({}, 'background');
  const cardBgColor = useThemeColor({ light: '#F9FAFB', dark: '#1F2937' }, 'card');
  const primaryColor = useThemeColor({}, 'tint');
  const dangerColor = useThemeColor({}, 'danger');
  const secondaryText = useThemeColor({ light: '#6B7280', dark: '#9CA3AF' }, 'text');

  // Load participants
  const loadParticipants = async () => {
    setIsLoading(true);
    try {
      const allParticipants = await getAllParticipants();
      setParticipants(allParticipants);
    } catch (error) {
      console.error("Error loading participants:", error);
      Alert.alert("Error", "Failed to load participants");
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadParticipants();
  }, []);

  const handleSelectParticipant = async (participant: ParticipantDetails) => {
    try {
      // Save as current participant
      await saveParticipantDetails(participant);
      onSelectParticipant(participant);
    } catch (error) {
      console.error("Error selecting participant:", error);
      Alert.alert("Error", "Failed to select participant");
    }
  };

  const handleDeleteParticipant = async (participant: ParticipantDetails) => {
    setDeletingCode(participant.code);

    try {
      // Get recordings count
      const recordings = await getRecordingsForParticipant(participant.code);

      Alert.alert(
        "Delete Participant",
        `Are you sure you want to delete ${participant.code}? ${recordings.length ? `This will also delete ${recordings.length} recording(s).` : ''}`,
        [
          { text: "Cancel", style: "cancel", onPress: () => setDeletingCode(null) },
          {
            text: "Delete",
            style: "destructive",
            onPress: async () => {
              try {
                await deleteParticipantWithRecordings(participant.code);
                // If this was the current participant, tell parent
                if (currentParticipant?.code === participant.code) {
                  onSelectParticipant(null);
                }
                // Reload list
                loadParticipants();
              } catch (error) {
                console.error("Error deleting participant:", error);
                Alert.alert("Error", "Failed to delete participant");
              } finally {
                setDeletingCode(null);
              }
            }
          }
        ]
      );
    } catch (error) {
      console.error("Error in delete dialog:", error);
      setDeletingCode(null);
      Alert.alert("Error", "Failed to prepare deletion");
    }
  };

  const renderParticipantItem = ({ item }: { item: ParticipantDetails }) => {
    const isActive = currentParticipant?.code === item.code;
    const isDeleting = deletingCode === item.code;

    return (
      <TouchableOpacity
        className={`
          py-4 px-4 mb-3 rounded-lg flex-row items-center justify-between
          ${isActive ? 'border-2 border-primary dark:border-primary-light' : 'border border-neutral-200 dark:border-neutral-700'}
        `}
        style={{ backgroundColor: cardBgColor }}
        disabled={isActive || isDeleting}
        onPress={() => handleSelectParticipant(item)}
        activeOpacity={0.7}
      >
        <View className="flex-1">
          <Text className="text-lg font-semibold" style={{ color: textColor }}>
            {item.code}
          </Text>

          {/* Details */}
          <View className="mt-1 flex-row flex-wrap">
            {item.dialect && (
              <Text className="text-sm mr-3" style={{ color: secondaryText }}>
                Dialect: {item.dialect}
              </Text>
            )}

            {item.age_range && (
              <Text className="text-sm mr-3" style={{ color: secondaryText }}>
                Age: {item.age_range}
              </Text>
            )}

            {item.gender && (
              <Text className="text-sm" style={{ color: secondaryText }}>
                Gender: {item.gender}
              </Text>
            )}
          </View>

          {isActive && (
            <View className="mt-1 flex-row items-center">
              <MaterialCommunityIcons name="check-circle" size={16} color={primaryColor} />
              <Text className="ml-1 text-sm font-medium" style={{ color: primaryColor }}>
                Current Participant
              </Text>
            </View>
          )}
        </View>

        {!isActive && (
          <TouchableOpacity
            className="p-2 rounded-full"
            onPress={() => handleDeleteParticipant(item)}
            disabled={isDeleting}
          >
            {isDeleting ? (
              <ActivityIndicator size="small" color={dangerColor} />
            ) : (
              <MaterialCommunityIcons name="delete-outline" size={24} color={dangerColor} />
            )}
          </TouchableOpacity>
        )}
      </TouchableOpacity>
    );
  };

  if (isLoading && !refreshing) {
    return (
      <View className="flex-1 justify-center items-center p-4">
        <ActivityIndicator size="large" color={primaryColor} />
        <Text className="mt-4 text-lg" style={{ color: textColor }}>Loading participants...</Text>
      </View>
    );
  }

  return (
    <View className="flex-1 p-4">
      <FlatList
        data={participants}
        keyExtractor={(item) => item.code}
        renderItem={renderParticipantItem}
        refreshing={refreshing}
        onRefresh={() => {
          setRefreshing(true);
          loadParticipants();
        }}
        ListHeaderComponent={
          <View className="mb-4">
            <Button
              title="Create New Participant"
              icon="account-plus"
              onPress={onCreateNewParticipant}
              className="bg-primary dark:bg-primary-dark p-4 rounded-lg flex-row justify-center items-center"
              textClassName="text-white dark:text-white ml-2 font-semibold text-lg"
              iconColor="white"
            />
          </View>
        }
        ListEmptyComponent={
          <View className="py-10 items-center justify-center">
            <MaterialCommunityIcons name="account-question" size={56} color={secondaryText} />
            <Text className="text-lg mt-4 text-center" style={{ color: textColor }}>
              No participants found
            </Text>
            <Text className="text-sm mt-2 text-center max-w-xs" style={{ color: secondaryText }}>
              Create a new participant to start recording.
            </Text>
          </View>
        }
      />
    </View>
  );
};
