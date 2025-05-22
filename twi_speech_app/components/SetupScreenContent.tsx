import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, TextInput, Alert, ActivityIndicator, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { Button } from '@/components/Button';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { ParticipantDetails, RecordingMetadata } from '@/types';
import { isValidCode } from '@/lib/utils';
import { ThemedView } from './ThemedView';
import { useThemeColor } from '@/hooks/useThemeColor';
import { getAllParticipants, savePendingRecording } from '@/lib/storage';
import { EXPECTED_TOTAL_RECORDINGS } from '@/constants/script';
import { checkParticipantExists } from '@/lib/api';
import { Modal } from './Modal';

// Add this interface to handle syncing recordings from the server

interface SyncStatus {
  isChecking: boolean;
  isImporting: boolean;
  foundOnServer: boolean;
  recordingsFound: number;
  recordingsImported: number;
}

interface SetupScreenContentProps {
  onSetupComplete: (details: ParticipantDetails) => Promise<void>;
  initialDetails?: ParticipantDetails | null;
  isNewParticipant?: boolean;
  onCancel?: () => void;
}

export default function SetupScreenContent({
  onSetupComplete,
  initialDetails,
  isNewParticipant = false,
  onCancel
}: SetupScreenContentProps) {
  // Existing state variables
  const [code, setCode] = useState(initialDetails?.code || '');
  const [dialect, setDialect] = useState(initialDetails?.dialect || '');
  const [ageRange, setAgeRange] = useState(initialDetails?.age_range || '');
  const [gender, setGender] = useState(initialDetails?.gender || '');
  const [existingCodes, setExistingCodes] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingCodes, setIsLoadingCodes] = useState(isNewParticipant);

  // Validation error states
  const [codeError, setCodeError] = useState('');
  const [dialectError, setDialectError] = useState('');
  const [ageRangeError, setAgeRangeError] = useState('');
  const [genderError, setGenderError] = useState('');

  // New state for backend sync
  const [syncStatus, setSyncStatus] = useState<SyncStatus>({
    isChecking: false,
    isImporting: false,
    foundOnServer: false,
    recordingsFound: 0,
    recordingsImported: 0
  });

  // Theme colors
  const inputBgColor = useThemeColor({}, 'background');
  const textColor = useThemeColor({}, 'text');
  const placeholderColor = useThemeColor({ light: '#9CA3AF', dark: '#6B7280' }, 'text');
  const borderColor = useThemeColor({ light: '#D1D5DB', dark: '#4B5563' }, 'border');
  const errorBorderColor = useThemeColor({ light: '#EF4444', dark: '#DC2626' }, 'danger');
  const successColor = useThemeColor({}, 'success');
  const secondaryTextColor = useThemeColor({ light: '#6B7280', dark: '#9CA3AF' }, 'text') || '#6B7280';
  const primaryColor = useThemeColor({}, 'tint') || '#4F46E5';

  // Load existing participant codes when creating a new participant
  useEffect(() => {
    if (isNewParticipant) {
      const loadExistingCodes = async () => {
        try {
          setIsLoadingCodes(true);
          const participants = await getAllParticipants();
          setExistingCodes(participants.map(p => p.code));
        } catch (error) {
          console.error("Error loading existing codes:", error);
        } finally {
          setIsLoadingCodes(false);
        }
      };

      loadExistingCodes();
    }
  }, [isNewParticipant]);

  // Effect to update form if initialDetails change
  useEffect(() => {
    if (initialDetails) {
      setCode(initialDetails.code || '');
      setDialect(initialDetails.dialect || '');
      setAgeRange(initialDetails.age_range || '');
      setGender(initialDetails.gender || '');
    }
  }, [initialDetails]);

  // Check if participant exists on server when code is validated
  const checkServerForParticipant = useCallback(async (participantCode: string) => {
    if (!isNewParticipant || !isValidCode(participantCode)) return;

    setSyncStatus(prev => ({ ...prev, isChecking: true, foundOnServer: false }));

    try {
      const serverData = await checkParticipantExists(participantCode);

      if (serverData && serverData.participant) {
        console.log(`[SetupScreen] Participant ${participantCode} found on server with ${serverData.recordings.length} recordings`);

        // Auto-fill participant details if available
        if (serverData.participant.dialect && !dialect) {
          setDialect(serverData.participant.dialect);
        }
        if (serverData.participant.age_range && !ageRange) {
          setAgeRange(serverData.participant.age_range);
        }
        if (serverData.participant.gender && !gender) {
          setGender(serverData.participant.gender);
        }

        // Update sync status
        setSyncStatus(prev => ({
          ...prev,
          foundOnServer: true,
          recordingsFound: serverData.recordings.length
        }));
      } else {
        console.log(`[SetupScreen] Participant ${participantCode} not found on server`);
        setSyncStatus(prev => ({ ...prev, foundOnServer: false }));
      }
    } catch (error) {
      console.error(`[SetupScreen] Error checking server for participant ${participantCode}:`, error);
    } finally {
      setSyncStatus(prev => ({ ...prev, isChecking: false }));
    }
  }, [isNewParticipant, dialect, ageRange, gender]);

  const syncRecordingsFromServer = async (participantCode: string) => {
    if (!isValidCode(participantCode)) return null;

    setSyncStatus(prev => ({
      ...prev,
      isChecking: true,
      foundOnServer: false,
      recordingsFound: 0
    }));

    try {
      // Check if participant exists on server
      const serverData = await checkParticipantExists(participantCode);

      if (!serverData) {
        setSyncStatus(prev => ({ ...prev, isChecking: false }));
        return null;
      }

      const { participant, recordings } = serverData;

      // Auto-fill participant fields from server data - make sure to update state
      if (participant) {
        if (participant.dialect) {
          setDialect(participant.dialect);
        }
        if (participant.age_range) {
          setAgeRange(participant.age_range);
        }
        if (participant.gender) {
          setGender(participant.gender);
        }
      }

      setSyncStatus(prev => ({
        ...prev,
        isChecking: false,
        foundOnServer: true,
        recordingsFound: recordings.length
      }));

      return { participant, recordings };
    } catch (error) {
      console.error('Error syncing from server:', error);
      setSyncStatus(prev => ({ ...prev, isChecking: false }));
      return null;
    }
  };


  // Import recordings from server
  const importServerRecordings = async (serverData: { participant: any; recordings: any[] }) => {
    if (!serverData || !serverData.recordings.length) return 0;

    setSyncStatus(prev => ({
      ...prev,
      isImporting: true,
      recordingsImported: 0
    }));

    let importedCount = 0;

    try {
      for (const recording of serverData.recordings) {
        // Create a unique ID for each imported recording
        const uniqueId = `import_${recording.id || Math.random().toString(36).substring(2, 15)}`;

        // Create a local metadata record that uses the server URL
        const localRecording: RecordingMetadata = {
          id: uniqueId, // Use our generated unique ID
          participantCode: recording.participant_code,
          promptId: recording.prompt_id,
          promptText: recording.prompt_text || recording.prompt_id.replace(/_/g, ' '),
          timestamp: new Date(recording.uploaded_at).getTime(),
          localUri: recording.file_url, // Use the remote URL directly
          originalFilename: recording.filename_original || `${recording.prompt_id}.m4a`,
          contentType: recording.content_type || 'audio/mp4',
          uploaded: true, // Mark as already uploaded since we're using the server URL
          recordingDuration: recording.recording_duration || 0, // Default to 0 if null
          dialect: serverData.participant.dialect,
          age_range: serverData.participant.age_range,
          gender: serverData.participant.gender,
          uploadStatus: 'pending'
        };

        // Save to local storage
        await savePendingRecording(localRecording);
        importedCount++;

        // Update progress
        setSyncStatus(prev => ({
          ...prev,
          recordingsImported: importedCount
        }));
      }

      console.log(`[SetupScreen] Imported ${importedCount} recordings from server`);
      return importedCount;
    } catch (error) {
      console.error('Error importing server recordings:', error);
      return importedCount; // Return how many we managed to import
    } finally {
      setSyncStatus(prev => ({ ...prev, isImporting: false }));
    }
  };

  // Validate all inputs
  const validateInputs = (): boolean => {
    let isValid = true;
    setCodeError('');
    setDialectError('');
    setAgeRangeError('');
    setGenderError('');

    // Validate Participant Code (Required)
    if (!isValidCode(code)) {
      setCodeError('Code format: TWI_Speaker_ followed by exactly 3 digits');
      isValid = false;
    } else if (isNewParticipant && existingCodes.includes(code.trim())) {
      setCodeError('This participant code already exists in your local device');
      isValid = false;
    }

    // Optional fields: only validate if non-empty and just whitespace
    if (dialect && dialect.trim().length === 0) {
      setDialectError('Dialect cannot be only whitespace');
      isValid = false;
    }
    if (ageRange && ageRange.trim().length === 0) {
      setAgeRangeError('Age Range cannot be only whitespace');
      isValid = false;
    }
    if (gender && gender.trim().length === 0) {
      setGenderError('Gender cannot be only whitespace');
      isValid = false;
    }

    // If valid and creating a new participant, check server
    if (isValid && isNewParticipant) {
      checkServerForParticipant(code.trim());
    }

    return isValid;
  };

  // Handle form submission
  const handleSave = async () => {
    if (!validateInputs()) {
      Alert.alert('Validation Error', 'Please correct the highlighted fields.');
      return;
    }

    setIsLoading(true);
    setCodeError('');

    try {
      // For new participants, check server
      if (isNewParticipant) {
        const serverData = await syncRecordingsFromServer(code.trim());

        if (serverData && serverData.recordings.length > 0) {
          // Ask user if they want to import recordings
          Alert.alert(
            "Recordings Found",
            `Found ${serverData.recordings.length} recordings for this participant on the server. Import them?`,
            [
              {
                text: "No",
                style: "cancel",
                onPress: () => {
                  // Continue creating participant without importing
                  finalizeParticipantCreation(0);
                }
              },
              {
                text: "Yes, Import",
                onPress: async () => {
                  const importCount = await importServerRecordings(serverData);
                  finalizeParticipantCreation(importCount);
                }
              }
            ]
          );
        } else {
          // No recordings found, just create participant
          finalizeParticipantCreation(0);
        }
      } else {
        // Just update existing participant
        finalizeParticipantCreation(0);
      }
    } catch (error: any) {
      console.error("[SetupScreenContent] Error during setup:", error);
      Alert.alert('Error', `Could not complete setup: ${error.message || 'Unknown error'}`);
      setIsLoading(false);
    }
  };

  const finalizeParticipantCreation = async (importedCount: number) => {
    try {
      // Get the latest server data before creating the participant
      const serverData = isNewParticipant ? await checkParticipantExists(code.trim()) : null;
      const serverParticipant = serverData?.participant;

      // Create the participant details object, prioritizing any server data
      const participantDetails: ParticipantDetails = {
        code: code.trim(),
        dialect: dialect.trim() || serverParticipant?.dialect || undefined,
        age_range: ageRange.trim() || serverParticipant?.age_range || undefined,
        gender: gender.trim() || serverParticipant?.gender || undefined,
        progress: initialDetails?.progress ?? {
          total_recordings: importedCount,
          total_required: EXPECTED_TOTAL_RECORDINGS,
          is_complete: importedCount >= EXPECTED_TOTAL_RECORDINGS,
        },
      };

      // Call the parent's onSetupComplete callback
      await onSetupComplete(participantDetails);
    } catch (error: any) {
      console.error("[SetupScreenContent] Error finalizing participant:", error);
      Alert.alert('Error', `Could not save participant details: ${error.message || 'Unknown error'}`);
      setIsLoading(false);
    }
  };


  const getInputStyle = (hasError: boolean) => `
    border-2 rounded-lg p-3 text-base
    dark:text-white dark:placeholder-neutral-400
    focus:border-primary focus:ring-1 focus:ring-primary
    ${hasError ? 'border-danger dark:border-danger-dark' : 'border-neutral-300 dark:border-neutral-600'}
  `;

  return (
    <ThemedView className="flex-1">
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={{ flex: 1 }}
      >
        <ScrollView
          contentContainerStyle={{ flexGrow: 1, justifyContent: 'center' }}
          className="p-6"
          keyboardShouldPersistTaps="handled"
        >
          {onCancel && <View className="absolute top-0 right-0">
            <Button
              title=""
              onPress={onCancel}
              className="flex-1 mr-2 p-2 rounded-lg bg-neutral-300 dark:bg-neutral-700 items-center"
              textClassName="text-neutral-800 dark:text-neutral-100 text-lg font-medium"
              icon="close"
            />
          </View>}
          <View className="items-center mb-8">
            <MaterialCommunityIcons
              name={isNewParticipant ? "account-plus" : "account-edit-outline"}
              size={64}
              color="#4F46E5"
            />
            <Text className="text-3xl font-bold mt-4 text-neutral-800 text-center dark:text-neutral-100">
              {isNewParticipant ? 'Create Participant' : 'Update Participant'}
            </Text>
            <Text className="text-lg text-neutral-600 mt-1.5 text-center dark:text-neutral-300">
              {isNewParticipant
                ? 'Create a new participant profile.'
                : 'Update participant details.'}
            </Text>
          </View>

          {/* Participant Code */}
          <Text className="text-sm font-semibold mb-1 text-neutral-700 uppercase tracking-wider dark:text-neutral-400">
            Participant Code *
          </Text>
          <TextInput
            className={getInputStyle(!!codeError)}
            style={{
              backgroundColor: inputBgColor,
              color: textColor,
              borderColor: codeError ? errorBorderColor : borderColor
            }}
            placeholder="e.g., TWI_Speaker_001"
            value={code}
            onChangeText={(text) => setCode(text)}
            onBlur={validateInputs}
            autoCapitalize="none"
            autoCorrect={false}
            placeholderTextColor={placeholderColor}
            returnKeyType="next"
            editable={isNewParticipant} // Only editable when creating a new participant
          />
          <View className="h-5 mt-0.5 mb-2.5">
            {codeError ? <Text className="text-danger text-xs dark:text-danger-light">{codeError}</Text> : null}
            {isLoadingCodes && <Text className="text-xs italic" style={{ color: placeholderColor }}>Checking existing codes...</Text>}
            {syncStatus.isChecking && <Text className="text-xs italic" style={{ color: placeholderColor }}>Checking server...</Text>}
            {syncStatus.foundOnServer && !syncStatus.isChecking && (
              <Text className="text-xs text-success font-medium">
                Found on server with {syncStatus.recordingsFound} recordings
              </Text>
            )}
          </View>

          {/* Dialect */}
          <Text className="text-sm font-semibold mb-1 text-neutral-700 uppercase tracking-wider dark:text-neutral-400">
            Dialect (e.g., Asante, Fante)
          </Text>
          <TextInput
            className={getInputStyle(!!dialectError)}
            style={{ backgroundColor: inputBgColor, color: textColor, borderColor: dialectError ? errorBorderColor : borderColor }}
            placeholder="Optional"
            value={dialect}
            onChangeText={(text) => setDialect(text)}
            onBlur={validateInputs}
            autoCapitalize="words"
            placeholderTextColor={placeholderColor}
            returnKeyType="next"
          />
          <View className="h-5 mt-0.5 mb-2.5">
            {dialectError ? <Text className="text-danger text-xs dark:text-danger-light">{dialectError}</Text> : null}
          </View>

          {/* Age Range */}
          <Text className="text-sm font-semibold mb-1 text-neutral-700 uppercase tracking-wider dark:text-neutral-400">
            Age Range (e.g., 18-25, 26-35)
          </Text>
          <TextInput
            className={getInputStyle(!!ageRangeError)}
            style={{ backgroundColor: inputBgColor, color: textColor, borderColor: ageRangeError ? errorBorderColor : borderColor }}
            placeholder="Optional"
            value={ageRange}
            onChangeText={(text) => setAgeRange(text)}
            onBlur={validateInputs}
            placeholderTextColor={placeholderColor}
            returnKeyType="next"
          />
          <View className="h-5 mt-0.5 mb-2.5">
            {ageRangeError ? <Text className="text-danger text-xs dark:text-danger-light">{ageRangeError}</Text> : null}
          </View>

          {/* Gender */}
          <Text className="text-sm font-semibold mb-1 text-neutral-700 uppercase tracking-wider dark:text-neutral-400">
            Gender (e.g., Male, Female, Other)
          </Text>
          <TextInput
            className={getInputStyle(!!genderError)}
            style={{ backgroundColor: inputBgColor, color: textColor, borderColor: genderError ? errorBorderColor : borderColor }}
            placeholder="Optional"
            value={gender}
            onChangeText={(text) => setGender(text)}
            onBlur={validateInputs}
            autoCapitalize="words"
            placeholderTextColor={placeholderColor}
            returnKeyType="done"
            onSubmitEditing={handleSave}
          />
          <View className="h-5 mt-0.5 mb-4">
            {genderError ? <Text className="text-danger text-xs dark:text-danger-light">{genderError}</Text> : null}
          </View>

          {/* Show sync option if participant found on server */}
          {syncStatus.foundOnServer && syncStatus.recordingsFound > 0 && !syncStatus.isImporting && (
            <View className="mb-4 p-4 border border-success/30 bg-success/10 rounded-lg">
              <Text className="text-success font-medium text-center mb-2">
                {syncStatus.recordingsFound} recordings found on server
              </Text>
              <Text className="text-sm text-neutral-700 dark:text-neutral-300 text-center mb-3">
                Recordings will be imported when you save this participant.
              </Text>
            </View>
          )}

          {/* Show sync progress if importing */}
          {syncStatus.isImporting && (
            <View className="mb-4 p-4 border border-primary/30 bg-primary/10 rounded-lg">
              <Text className="text-primary font-medium text-center mb-2">
                Importing recordings...
              </Text>
              <View className="flex-row items-center justify-center mb-2">
                <ActivityIndicator size="small" color={primaryColor} />
                <Text className="ml-2 text-neutral-700 dark:text-neutral-300">
                  {syncStatus.recordingsImported} / {syncStatus.recordingsFound}
                </Text>
              </View>
            </View>
          )}

          {/* Buttons */}
          <View className="flex-row justify-between mt-2">
            <Button
              title={isNewParticipant ? "Create Participant" : "Update Details"}
              onPress={handleSave}
              disabled={isLoading || isLoadingCodes || syncStatus.isChecking || syncStatus.isImporting}
              isLoading={isLoading}
              className={`flex-1 ${onCancel ? 'ml-2' : ''} py-3.5 px-6 rounded-lg bg-primary dark:bg-primary-dark items-center shadow-md`}
              disabledClassName="bg-primary-light opacity-70 dark:bg-primary/50"
              textClassName="text-white text-lg font-bold dark:text-neutral-100"
              icon={isNewParticipant ? "account-plus" : "check-circle-outline"}
              iconColor="white"
            />
          </View>
        </ScrollView>
      </KeyboardAvoidingView>

      {/* Modal for sync success */}
      {syncStatus.recordingsImported > 0 && !syncStatus.isImporting && (
        <Modal
          visible={syncStatus.recordingsImported > 0 && !syncStatus.isImporting}
          onClose={() => setSyncStatus(prev => ({ ...prev, recordingsImported: 0 }))}
        >
          <View className="items-center p-4">
            <MaterialCommunityIcons name="cloud-download-outline" size={56} color={successColor} />
            <Text className="text-lg font-bold text-center mt-3" style={{ color: textColor }}>
              Import Complete!
            </Text>
            <Text className="text-base text-center mt-2" style={{ color: secondaryTextColor }}>
              Successfully imported {syncStatus.recordingsImported} recordings from the server.
            </Text>
            <Button
              title="OK"
              onPress={() => setSyncStatus(prev => ({ ...prev, recordingsImported: 0 }))}
              className="mt-6 bg-primary dark:bg-primary-dark px-8 py-2.5 rounded-lg"
              textClassName="text-white font-semibold"
            />
          </View>
        </Modal>
      )}
    </ThemedView>
  );
}
