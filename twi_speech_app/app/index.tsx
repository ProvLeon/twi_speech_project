import * as FileSystem from 'expo-file-system';
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { View, Text, Alert, ActivityIndicator, FlatList, RefreshControl, StyleSheet, TouchableOpacity } from 'react-native';
import { useRouter, useNavigation, useFocusEffect } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { Audio, InterruptionModeAndroid, InterruptionModeIOS } from 'expo-av';
import { Button } from '@/components/Button';
import { RecordingListItem } from '@/components/RecordingListItem';
import {
  getParticipantDetails,
  saveParticipantDetails,
  getPendingRecordings,
  deletePendingRecordingById,
  updateRecordingUploadedStatus,
  deleteAllDeviceRecordings,
  deleteAllRecordingsForParticipant,
  getAllParticipants,
  getRecordingsForParticipant,
} from '@/lib/storage';
import { uploadRecording } from '@/lib/api';
import * as Network from 'expo-network';
import { RecordingMetadata, ParticipantDetails, UploadResponse, RecordingProgress } from '@/types';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import SetupScreenContent from '@/components/SetupScreenContent';
import { ThemedSafeAreaView } from '@/components/ThemedSafeAreaView';
import { isValidCode } from '@/lib/utils';
import { useThemeColor } from '@/hooks/useThemeColor';
import { ParticipantSelector } from '@/components/ParticipantSelector';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Redirect } from "expo-router";
import { EXPECTED_TOTAL_RECORDINGS } from '@/constants/script'; // Import constant

const WELCOME_SEEN_KEY = 'welcomeScreenSeen_v1';


type PlaybackStatus = 'idle' | 'loading' | 'playing' | 'error';
type ViewMode = 'participant' | 'settings' | 'select-participant' | 'create-participant' | 'edit-participant' | 'participant-recordings';
type UploadProgressState = { current: number; total: number } | null;


export default function HomeScreen() {
  // --- Hooks at the top level ---
  const navigation = useNavigation();
  const router = useRouter();
  const iconColor = useThemeColor({}, 'icon') || '#000';
  const dangerColor = useThemeColor({}, 'danger') || '#EF4444';
  const primaryColor = useThemeColor({}, 'tint') || '#4F46E5';
  const textColor = useThemeColor({}, 'text') || '#000';
  const secondaryTextColor = useThemeColor({ light: '#6B7280', dark: '#9CA3AF' }, 'text') || '#6B7280';
  const successColor = useThemeColor({}, 'success') || '#10B981';
  const warningColor = useThemeColor({}, 'warning') || '#F59E0B';
  const cardBgColor = useThemeColor({ light: '#F9FAFB', dark: '#1F2937' }, 'card');
  const borderColor = useThemeColor({ light: '#E5E7EB', dark: '#374151' }, 'border');


  // State hooks
  const [isCheckingWelcome, setIsCheckingWelcome] = useState(true);
  const [hasSeenWelcome, setHasSeenWelcome] = useState(false);
  const [participantDetails, setParticipantDetails] = useState<ParticipantDetails | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSetupMode, setIsSetupMode] = useState(false);
  const [allRecordings, setAllRecordings] = useState<RecordingMetadata[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>('participant');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [networkAvailable, setNetworkAvailable] = useState(true);
  const [isDeletingId, setIsDeletingId] = useState<string | null>(null);
  const [isDeletingAll, setIsDeletingAll] = useState(false);
  const [participantCount, setParticipantCount] = useState<number>(0);
  const [settingsTabIndex, setSettingsTabIndex] = useState<number>(0); // 0: All Recordings, 1: Participants
  // NEW Upload State
  const [uploadingItemId, setUploadingItemId] = useState<string | null>(null); // ID of the item currently uploading
  const [uploadProgress, setUploadProgress] = useState<UploadProgressState>(null); // { current: number, total: number }


  // --- used Refs instead of state for audio playback ---
  const playbackState = useRef({
    sound: null as Audio.Sound | null,
    status: 'idle' as PlaybackStatus,
    currentUri: null as string | null,
    isActive: false,
  }).current;

  // UI updates state
  const [playbackUiVersion, setPlaybackUiVersion] = useState(0);

  // All the callbacks are defined here
  const triggerHaptic = useCallback((type: Haptics.ImpactFeedbackStyle = Haptics.ImpactFeedbackStyle.Light) => {
    Haptics.impactAsync(type).catch(console.error);
  }, []);

  const updatePlaybackUI = useCallback(() => {
    setPlaybackUiVersion(prev => prev + 1);
  }, []);

  const checkNetwork = useCallback(async () => {
    try {
      const state = await Network.getNetworkStateAsync();
      const isConnected = !!(state?.isConnected) && !!(state?.isInternetReachable);
      setNetworkAvailable(prev => {
        // if (prev !== isConnected) { // Reduce logging noise
        //   console.log(`HomeScreen: Network status changed: ${isConnected ? 'Online' : 'Offline'}`);
        // }
        return isConnected;
      });
    } catch (error) {
      console.error("HomeScreen: Error checking network:", error);
      setNetworkAvailable(false);
    }
  }, []);

  const loadAllRecordings = useCallback(async () => {
    console.log("HomeScreen: Loading ALL recordings from storage...");
    try {
      const storedRecordings = await getPendingRecordings(); // This fetches from AsyncStorage
      if (Array.isArray(storedRecordings)) {
        // Reset 'uploading' status on load, keep 'failed' or 'pending'
        const recordingsWithResetStatus = storedRecordings.map(rec => ({
          ...rec,
          uploadStatus: rec.uploadStatus === 'uploading' ? 'pending' : (rec.uploadStatus ?? 'pending') // Reset 'uploading' only
        }));
        recordingsWithResetStatus.sort((a, b) => b.timestamp - a.timestamp); // Sort newest first
        setAllRecordings(recordingsWithResetStatus);
        console.log(`HomeScreen: Loaded ${recordingsWithResetStatus.length} total recordings.`);
      } else {
        console.error("HomeScreen: getPendingRecordings returned non-array:", storedRecordings);
        setAllRecordings([]);
      }
    } catch (error) {
      console.error("HomeScreen: Error loading recordings:", error);
      setAllRecordings([]);
    }
  }, []);


  const loadParticipantCount = useCallback(async () => {
    try {
      const participants = await getAllParticipants();
      setParticipantCount(Array.isArray(participants) ? participants.length : 0);
    } catch (error) {
      console.error("HomeScreen: Error loading participant count:", error);
      setParticipantCount(0);
    }
  }, []);

  const onRefresh = useCallback(async () => {
    console.log("HomeScreen: Refreshing data...");
    setIsRefreshing(true);
    triggerHaptic();
    await loadAllRecordings();
    await loadParticipantCount();
    await checkNetwork();
    setIsRefreshing(false);
    console.log("HomeScreen: Refresh complete.");
  }, [loadAllRecordings, loadParticipantCount, checkNetwork, triggerHaptic]);

  const localRecordingCount = useMemo(() => {
    if (viewMode !== 'participant' || !participantDetails?.code) {
      return 0;
    }
    // Filter all recordings specifically for the current participant code
    return allRecordings.filter(rec => rec.participantCode === participantDetails.code).length;
  }, [allRecordings, participantDetails?.code, viewMode]);

  const displayedRecordings = useMemo(() => {
    if (viewMode === 'settings') return allRecordings;

    if (viewMode === 'participant' && participantDetails?.code) {
      return allRecordings.filter(rec => rec.participantCode === participantDetails.code);
    }
    return [];
  }, [allRecordings, viewMode, participantDetails]);

  const loadInitialData = useCallback(async (options: { forceSetup?: boolean } = {}) => {
    console.log("HomeScreen: Loading initial data...");
    setIsLoading(true);
    setIsSetupMode(false);
    setParticipantDetails(null);

    try {
      let shouldEnterSetup = false;
      let fetchedDetails: ParticipantDetails | null = null;

      if (options?.forceSetup) {
        console.log("HomeScreen: Forcing setup mode.");
        shouldEnterSetup = true;
      } else {
        fetchedDetails = await getParticipantDetails();
        if (fetchedDetails && isValidCode(fetchedDetails.code)) {
          console.log("HomeScreen: Found valid participant details:", fetchedDetails);
          setParticipantDetails(fetchedDetails);
        } else {
          console.log("HomeScreen: No valid participant found, entering setup mode.");
          shouldEnterSetup = true;
        }
      }

      // Always load participant count
      await loadParticipantCount();

      if (shouldEnterSetup) {
        setIsSetupMode(true); // Enter setup flow
        const allExistingParticipants = await getAllParticipants();
        if (Array.isArray(allExistingParticipants) && allExistingParticipants.length > 0) {
          console.log("HomeScreen: Existing participants found, showing selection screen.");
          setViewMode('select-participant'); // Show selection screen if participants exist
        } else {
          console.log("HomeScreen: No participants exist, showing creation screen.");
          setViewMode('create-participant'); // Show creation screen if none exist
        }
      } else {
        console.log("HomeScreen: Setting view mode to participant.");
        setIsSetupMode(false); // Not in setup flow
        setViewMode('participant'); // Show participant view
      }

      await loadAllRecordings();
      await checkNetwork();

    } catch (error) {
      console.error("HomeScreen: Error loading initial data:", error);
      Alert.alert("Error", "Failed to load initial data. Please restart the app.");
      setIsSetupMode(true);
      setParticipantDetails(null);
      setAllRecordings([]);
    } finally {
      setIsLoading(false);
    }
  }, [loadAllRecordings, loadParticipantCount, checkNetwork]);

  const verifyFileExists = useCallback(async (uri: string): Promise<boolean> => {
    if (!uri) return false;
    try {
      const fileInfo = await FileSystem.getInfoAsync(uri);
      return fileInfo.exists;
    } catch (error) {
      console.error("Error verifying file existence:", error);
      return false;
    }
  }, []);

  const unloadPlaybackSound = useCallback(async () => {
    if (playbackState.sound) {
      try { await playbackState.sound.unloadAsync(); }
      catch (e) { console.error("Unload error:", e); }
      playbackState.sound = null;
      playbackState.status = 'idle';
      playbackState.currentUri = null;
      playbackState.isActive = false;
      updatePlaybackUI(); // Trigger UI update
    }
  }, [playbackState, updatePlaybackUI]);

  const handlePlayRecording = useCallback(async (uri: string) => {
    triggerHaptic();

    console.log(`HomeScreen: handlePlayRecording with uri: ${uri}`);
    console.log(`HomeScreen: Current playback status: ${playbackState.status}`);
    console.log(`HomeScreen: Currently playing: ${playbackState.currentUri}`);

    // If already playing this URI, stop it
    if (playbackState.status === 'playing' && playbackState.currentUri === uri) {
      console.log("HomeScreen: Stopping current playback");
      await unloadPlaybackSound();
      return;
    }

    // Unload any existing playback
    await unloadPlaybackSound();

    // Validate URI
    if (!uri) {
      console.warn("HomeScreen: Playback attempted with invalid URI.");
      return;
    }

    // Check if file exists
    const fileExists = await verifyFileExists(uri);
    if (!fileExists) {
      console.error(`HomeScreen: File does not exist at URI: ${uri}`);
      Alert.alert("Playback Error", "Audio file not found.");
      return;
    }

    // Set refs to loading state
    console.log(`HomeScreen: Loading sound: ${uri}`);
    playbackState.status = 'loading';
    playbackState.currentUri = uri;
    playbackState.isActive = true;
    updatePlaybackUI();

    try {
      // Configure audio mode
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
        interruptionModeIOS: InterruptionModeIOS.DoNotMix,
        interruptionModeAndroid: InterruptionModeAndroid.DoNotMix,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
        staysActiveInBackground: false,
      });

      // Create and play the sound
      const { sound } = await Audio.Sound.createAsync(
        { uri },
        { shouldPlay: true },
        (status) => {
          if (!status.isLoaded) {
            if (status.error) console.error(`HomeScreen: Playback Error: ${status.error}`);
            unloadPlaybackSound();
          } else if (status.didJustFinish) {
            console.log("HomeScreen: Playback finished");
            unloadPlaybackSound();
          }
        }
      );

      playbackState.sound = sound;
      playbackState.status = 'playing';
      updatePlaybackUI();

    } catch (error: any) {
      console.error("HomeScreen: Error loading/playing sound:", error);
      Alert.alert("Playback Error", error.message || "Could not play audio.");
      unloadPlaybackSound();
    }

  }, [triggerHaptic, unloadPlaybackSound, verifyFileExists, updatePlaybackUI]);

  const handleDeleteRecording = useCallback((id: string, promptId: string) => {
    if (isDeletingAll || isUploading) return; // Prevent delete during other operations

    triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
    Alert.alert(
      "Delete Recording?",
      `Are you sure you want to delete the recording for prompt "${promptId?.replace(/_/g, ' ') || 'Unknown'}"? This cannot be undone.`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete", style: "destructive",
          onPress: async () => {
            setIsDeletingId(id); // Show spinner on the item
            try {
              const deleted = await deletePendingRecordingById(id); // Deletes from storage & file system
              if (deleted) {
                setAllRecordings(prev => prev.filter(rec => rec.id !== id));
                console.log(`HomeScreen: Deleted recording ${id}`);
              } else {
                Alert.alert("Error", "Could not delete recording metadata or file.");
                await loadAllRecordings();
              }
            } catch (error) {
              console.error("HomeScreen: Error deleting recording:", error);
              Alert.alert("Error", "An error occurred while deleting.");
              await loadAllRecordings();
            } finally {
              setIsDeletingId(null);
            }
          },
        },
      ]
    );
  }, [isDeletingAll, isUploading, triggerHaptic, loadAllRecordings]);

  const handleDeleteAll = useCallback(() => {
    if (isDeletingAll || isUploading || displayedRecordings.length === 0) return;

    triggerHaptic(Haptics.ImpactFeedbackStyle.Heavy);

    const isParticipantView = viewMode === 'participant' && participantDetails?.code;
    const count = displayedRecordings.length;
    const title = isParticipantView
      ? `Delete All for ${participantDetails?.code || 'Current Participant'}?`
      : "Delete ALL Device Recordings?";
    const message = isParticipantView
      ? `Are you sure you want to delete all ${count} recordings for participant ${participantDetails?.code || 'Current'}? This cannot be undone.`
      : `Are you sure you want to delete all ${count} recordings stored on this device? This cannot be undone.`;
    const buttonText = isParticipantView ? "Delete All for Participant" : "Delete All From Device";

    Alert.alert(title, message, [
      { text: "Cancel", style: "cancel" },
      {
        text: buttonText, style: "destructive",
        onPress: async () => {
          setIsDeletingAll(true);
          await unloadPlaybackSound();

          let result = { deletedCount: 0, failedCount: 0 };
          try {
            if (isParticipantView && participantDetails?.code) {
              result = await deleteAllRecordingsForParticipant(participantDetails.code);
              setAllRecordings(prev => prev.filter(r => r.participantCode !== participantDetails.code));
            } else {
              result = await deleteAllDeviceRecordings();
              setAllRecordings([]);
            }
          } catch (error) {
            console.error(`HomeScreen: Error during deleteAll call (Mode: ${viewMode}):`, error);
            result.failedCount = count;
            await loadAllRecordings();
          } finally {
            setIsDeletingAll(false);
          }

          let alertMessage = `${result.deletedCount} recording(s) deleted successfully.`;
          if (result.failedCount > 0) {
            alertMessage += ` ${result.failedCount} failed to delete.`;
          }
          Alert.alert("Deletion Complete", alertMessage);
        },
      },
    ]);
  }, [
    isDeletingAll, isUploading, displayedRecordings, viewMode, participantDetails,
    triggerHaptic, loadAllRecordings, unloadPlaybackSound
  ]);


  const handleUploadComplete = useCallback(async (response: UploadResponse) => {
    if (response.progress?.is_complete && participantDetails) {
      const updatedDetails: ParticipantDetails = {
        ...participantDetails,
        progress: response.progress
      };
      setParticipantDetails(updatedDetails);
      await saveParticipantDetails(updatedDetails);
      console.log(`[UploadComplete] Participant ${participantDetails.code} completion status updated.`);
    } else if (participantDetails && response.progress) {
      const updatedDetails: ParticipantDetails = {
        ...participantDetails,
        progress: response.progress
      };
      setParticipantDetails(updatedDetails);
      await saveParticipantDetails(updatedDetails);
    }
  }, [participantDetails, saveParticipantDetails]);


  const handleUploadAll = useCallback(async () => {
    if (viewMode === 'settings') {
      Alert.alert("Action Not Available", "Please switch back to the participant view to upload recordings.");
      return;
    }
    if (!participantDetails) {
      Alert.alert("Setup Needed", "Participant details are not set.");
      return;
    }
    if (isUploading) {
      console.log("[Upload] Upload already in progress.");
      return;
    }

    triggerHaptic();
    await checkNetwork();

    if (!networkAvailable) {
      Alert.alert("Offline", "No internet connection available for upload.");
      return;
    }

    const toUpload = displayedRecordings.filter(rec => !rec.uploaded && rec.uploadStatus !== 'failed');

    if (toUpload.length === 0) {
      Alert.alert("Up to Date", `No recordings waiting for upload for ${participantDetails.code}.`);
      return;
    }

    setIsUploading(true);
    setUploadProgress({ current: 0, total: toUpload.length });

    let successCount = 0;
    let failCount = 0;

    try {
      console.log(`[Upload] Starting upload for ${toUpload.length} recordings...`);

      for (let i = 0; i < toUpload.length; i++) {
        const recordingMeta = toUpload[i];
        setUploadingItemId(recordingMeta.id);
        setUploadProgress({ current: i + 1, total: toUpload.length });

        setAllRecordings(prev => prev.map(rec =>
          rec.id === recordingMeta.id ? { ...rec, uploadStatus: 'uploading' } : rec
        ));

        try {
          const fileExists = await verifyFileExists(recordingMeta.localUri);
          if (!fileExists) {
            console.error(`[Upload] File missing, skipping: ${recordingMeta.localUri}`);
            failCount++;
            setAllRecordings(prev => prev.map(rec =>
              rec.id === recordingMeta.id ? { ...rec, uploadStatus: 'failed' } : rec
            ));
            continue;
          }

          console.log(`[Upload] Uploading ${i + 1}/${toUpload.length}: ${recordingMeta.id}...`);
          const uploadResult = await uploadRecording(recordingMeta);
          console.log(`[Upload] Response for ${recordingMeta.id}:`, uploadResult);

          if (uploadResult) {
            successCount++;
            setAllRecordings(prev => prev.map(rec =>
              rec.id === recordingMeta.id ? { ...rec, uploaded: true, uploadStatus: 'pending' } : rec
            ));
            await updateRecordingUploadedStatus(recordingMeta.id, true);
            await handleUploadComplete(uploadResult);
          } else {
            failCount++;
            setAllRecordings(prev => prev.map(rec =>
              rec.id === recordingMeta.id ? { ...rec, uploadStatus: 'failed' } : rec
            ));
          }
        } catch (error) {
          console.error(`[Upload] Unexpected error processing recording ${recordingMeta.id}:`, error);
          failCount++;
          setAllRecordings(prev => prev.map(rec =>
            rec.id === recordingMeta.id ? { ...rec, uploadStatus: 'failed' } : rec
          ));
        }
      }

      let finalMessage = `${successCount} recording(s) uploaded successfully.`;
      if (failCount > 0) {
        finalMessage += ` ${failCount} failed. Check logs or try again later.`;
      }
      Alert.alert("Upload Complete", finalMessage);

    } catch (outerError) {
      console.error("[Upload] Critical error during upload process:", outerError);
      Alert.alert("Upload Error", "An unexpected error occurred during the upload process.");
    } finally {
      console.log("[Upload] Process finished. Resetting states.");
      setIsUploading(false);
      setUploadingItemId(null);
      setUploadProgress(null);
    }
  }, [
    viewMode, participantDetails, networkAvailable, displayedRecordings, isUploading,
    triggerHaptic, checkNetwork, verifyFileExists, handleUploadComplete,
    setAllRecordings, updateRecordingUploadedStatus, saveParticipantDetails
  ]);


  const handleSetupComplete = useCallback(async (details: ParticipantDetails) => {
    console.log("HomeScreen: [handleSetupComplete] Received details:", details);
    setIsLoading(true);
    try {
      if (!details || !isValidCode(details.code)) {
        throw new Error("Invalid participant code provided.");
      }

      const savedSuccessfully = await saveParticipantDetails(details);
      if (!savedSuccessfully) {
        throw new Error("Failed to save participant details to storage.");
      }

      setParticipantDetails(details);
      setIsSetupMode(false);
      setViewMode('participant');
      await loadAllRecordings();
      await loadParticipantCount();
      console.log("HomeScreen: [handleSetupComplete] Setup process complete.");

    } catch (error: any) {
      console.error("HomeScreen: [handleSetupComplete] Error:", error);
      Alert.alert("Error", `Failed to finalize setup: ${error.message || 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  }, [loadAllRecordings, loadParticipantCount]);



  const handleParticipantSelected = useCallback(async (participant: ParticipantDetails | null) => {
    if (participant) {
      const saved = await saveParticipantDetails(participant);
      if (saved) {
        setParticipantDetails(participant);
        setIsSetupMode(false);
        setViewMode('participant');
        console.log(`HomeScreen: Participant ${participant.code} selected.`);
      } else {
        Alert.alert("Error", "Could not save participant selection.");
      }
    } else {
      setParticipantDetails(null);
      setViewMode('select-participant');
    }
  }, []);

  const handleCreateNewParticipant = useCallback(() => {
    setViewMode('create-participant');
    setIsSetupMode(true);
  }, []);

  const goToSettings = useCallback(() => {
    setViewMode('settings');
    setSettingsTabIndex(0);
    triggerHaptic();
  }, [triggerHaptic]);


  // --- Effects ---
  useEffect(() => {
    let title = "Twi Speech Recorder";
    if (viewMode === 'settings') title = "Settings & Management";
    else if (isSetupMode) {
      if (viewMode === 'select-participant') title = "Select Participant";
      else if (viewMode === 'create-participant') title = "New Participant";
      else if (viewMode === 'edit-participant') title = `Edit: ${participantDetails?.code ?? 'Participant'}`;
      else title = "Participant Setup";
    } else if (participantDetails?.code) title = `Recordings (${participantDetails.code})`;
    navigation.setOptions({ title });
  }, [navigation, participantDetails, viewMode, isSetupMode]);

  // --- UPDATED: renderProgress ---
  const renderProgress = useCallback((
    recordingsForParticipant: RecordingMetadata[],
    uploadProgressState: UploadProgressState
  ) => {
    if (viewMode !== 'participant' || !participantDetails) return null;

    const localTotal = recordingsForParticipant.length;
    const localRequired = EXPECTED_TOTAL_RECORDINGS;
    const isLocallyComplete = localTotal >= localRequired;

    // Calculate uploaded count based on current state
    const uploadedCount = recordingsForParticipant.filter(rec => rec.uploaded).length;

    const localPercentage = localRequired > 0
      ? Math.min(100, Math.round((localTotal / localRequired) * 100))
      : 0;

    const uploadedPercentage = localRequired > 0
      ? Math.min(100, Math.round((uploadedCount / localRequired) * 100))
      : 0;

    // Use server progress primarily for completion flag, if available
    const serverProgress = participantDetails.progress;
    const isServerComplete = serverProgress?.is_complete ?? false;
    const serverTotal = serverProgress?.total_recordings ?? -1; // Use -1 to indicate unavailable

    const displayPercentage = localPercentage; // Always show local percentage textually

    let progressText = `${localTotal} / ${localRequired} Recorded`;
    if (uploadedCount > 0 && !isUploading) { // Show uploaded count when not actively uploading
      progressText += ` (${uploadedCount} Uploaded)`;
    } else if (isUploading && uploadProgressState) {
      // Could potentially show upload progress here too, but might be redundant with button
      // progressText = `Uploading ${uploadProgressState.current}/${uploadProgressState.total}...`;
    }

    return (
      <View>
        {/* Progress Text */}
        <View className="flex-row justify-between mb-1 px-1">
          <Text style={{ color: secondaryTextColor }} className="text-sm">
            {progressText}
          </Text>
          <Text style={{ color: isLocallyComplete ? successColor : primaryColor }} className="text-sm font-bold">
            {displayPercentage}%
          </Text>
        </View>

        {/* Progress Bar Container */}
        <View className="h-3 bg-neutral-200 dark:bg-neutral-700 rounded-full overflow-hidden relative">
          {/* Local Progress Bar (Green) - Base layer */}
          <View
            className="absolute top-0 left-0 bottom-0 bg-success dark:bg-success-dark rounded-full"
            style={{ width: `${localPercentage}%` }}
          />
          {/* Uploaded Progress Bar (Blue) - Overlay */}
          {uploadedPercentage > 0 && ( // Only show if some are uploaded
            <View
              className="absolute top-0 left-0 bottom-0 bg-primary dark:bg-primary-dark rounded-full"
              style={{ width: `${uploadedPercentage}%` }}
            />
          )}
        </View>

        {/* Completion Text */}
        {isLocallyComplete && !isUploading && ( // Hide local completion message during upload
          <View className="flex-row items-center justify-center mt-2">
            <MaterialCommunityIcons name="check-decagram" size={18} color={successColor} />
            <Text style={{ color: successColor }} className="ml-1.5 text-center font-semibold">
              All required prompts recorded!
            </Text>
          </View>
        )}

        {/* Server Completion Mismatch Warning */}
        {isServerComplete && !isLocallyComplete && !isUploading && (
          <Text style={{ color: warningColor }} className="text-xs text-center mt-1 italic">
            (Backend reports completion, upload needed)
          </Text>
        )}
        {/* Server Total Info (Optional, uncomment if useful) */}
        {/* {serverTotal !== -1 && serverTotal !== localTotal && (
           <Text style={{ color: secondaryTextColor }} className="text-xs text-center mt-1">
             (Server sync: {serverTotal} recordings)
           </Text>
         )} */}
      </View>
    );
  }, [
    viewMode, participantDetails, textColor, secondaryTextColor, primaryColor, successColor, warningColor,
    EXPECTED_TOTAL_RECORDINGS, isUploading // Added isUploading dependency
  ]);

  useEffect(() => {
    loadInitialData();
    return () => { unloadPlaybackSound(); };
  }, [loadInitialData, unloadPlaybackSound]);


  useFocusEffect(
    useCallback(() => {
      if (!isSetupMode) {
        console.log("HomeScreen: Focus detected, reloading data...");
        loadInitialData();
        unloadPlaybackSound();
      } else {
        console.log("HomeScreen: Focus detected, but in setup mode - skipping reload.");
      }
      return () => {
        unloadPlaybackSound();
      };
    }, [isSetupMode, loadInitialData, unloadPlaybackSound])
  );

  useEffect(() => {
    const intervalId = setInterval(checkNetwork, 15000);
    return () => clearInterval(intervalId);
  }, [checkNetwork]);

  useEffect(() => {
    const checkWelcomeScreen = async () => {
      try {
        const welcomeSeen = await AsyncStorage.getItem(WELCOME_SEEN_KEY);
        setHasSeenWelcome(welcomeSeen === 'true');
      } catch (e) {
        console.error('Error checking welcome screen state:', e);
        setHasSeenWelcome(false);
      } finally {
        setIsCheckingWelcome(false);
      }
    };
    checkWelcomeScreen();
  }, []);



  // --- Memos ---
  const pendingToUploadCount = useMemo(() => {
    if (viewMode !== 'participant' || !participantDetails) return 0;
    return displayedRecordings.filter(rec => !rec.uploaded && rec.uploadStatus !== 'failed').length;
  }, [displayedRecordings, viewMode, participantDetails]);

  const canUpload = useMemo(() => (
    !isUploading &&
    pendingToUploadCount > 0 &&
    networkAvailable &&
    viewMode === 'participant'
  ), [isUploading, pendingToUploadCount, networkAvailable, viewMode]);

  const canDeleteAllDisplayed = useMemo(() => (
    displayedRecordings.length > 0 && !isDeletingAll && !isUploading
  ), [displayedRecordings, isDeletingAll, isUploading]);

  if (isCheckingWelcome || isLoading) {
    return (
      <ThemedSafeAreaView className="flex-1 justify-center items-center">
        <ActivityIndicator size="large" color={primaryColor} />
      </ThemedSafeAreaView>
    );
  }

  if (!hasSeenWelcome) {
    router.replace('/welcome');
    return null;
  }


  // --- Render Methods ---
  const renderSettingsScreen = () => {
    // ... (keep existing settings screen rendering logic)
    return (
      <>
        <View className="bg-white dark:bg-neutral-800 border-b border-neutral-200 dark:border-neutral-700">
          <View className="px-4 pt-4 pb-0">
            <View className="flex-row items-center justify-between mb-4 px-1">
              <Button
                title=""
                icon="arrow-left-circle-outline"
                iconSize={26}
                onPress={() => setViewMode('participant')}
                className="p-1"
                iconColor={iconColor || undefined}
              />
              <Text
                className="text-xl font-semibold text-center flex-1 mx-2"
                numberOfLines={1}
                style={{ color: textColor }}
              >
                Settings & Management
              </Text>
              <View style={{ width: 34 }} />
            </View>
          </View>

          {/* Settings Tab Navigation */}
          <View className="flex-row border-b border-neutral-200 dark:border-neutral-700">
            <TouchableOpacity
              className={`flex-1 py-3 px-4 ${settingsTabIndex === 0 ? 'border-b-2 border-primary dark:border-primary-light' : ''}`}
              onPress={() => setSettingsTabIndex(0)}
            >
              <Text
                className={`text-center font-medium ${settingsTabIndex === 0 ? 'text-primary dark:text-primary-light' : 'text-neutral-600 dark:text-neutral-400'}`}
              >
                All Recordings
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              className={`flex-1 py-3 px-4 ${settingsTabIndex === 1 ? 'border-b-2 border-primary dark:border-primary-light' : ''}`}
              onPress={() => setSettingsTabIndex(1)}
            >
              <Text
                className={`text-center font-medium ${settingsTabIndex === 1 ? 'text-primary dark:text-primary-light' : 'text-neutral-600 dark:text-neutral-400'}`}
              >
                Participants
              </Text>
            </TouchableOpacity>
          </View>
        </View>

        {settingsTabIndex === 0 ? (
          // All Recordings Tab Content
          <FlatList
            data={allRecordings}
            keyExtractor={(item) => item.id || String(Math.random())}
            extraData={[playbackUiVersion, isDeletingId, isDeletingAll, uploadingItemId]} // Add uploadingItemId
            renderItem={({ item }) => (
              <RecordingListItem
                recording={item}
                onPlay={handlePlayRecording}
                onDelete={() => handleDeleteRecording(item.id, item.promptId)}
                isPlaying={playbackState.currentUri === item.localUri && playbackState.status === 'playing'}
                isDeleting={isDeletingId === item.id || isDeletingAll}
                isUploadingNow={uploadingItemId === item.id} // Pass uploading state
                showParticipantCode={true} // Always show code in settings view
              />
            )}
            ListHeaderComponent={
              <View className="p-4">
                <View className="flex-row items-center justify-between mb-4">
                  <Text className="text-lg font-semibold" style={{ color: textColor }}>
                    All Device Recordings ({allRecordings.length})
                  </Text>
                  <Button
                    title="Delete All"
                    icon={isDeletingAll ? undefined : "delete-sweep-outline"}
                    onPress={handleDeleteAll}
                    disabled={!canDeleteAllDisplayed}
                    isLoading={isDeletingAll}
                    className="bg-danger/10 dark:bg-danger/25 px-3 py-1.5 rounded-lg flex-row items-center justify-center"
                    textClassName="text-danger dark:text-danger-light text-sm font-medium ml-1"
                    iconColor={dangerColor || undefined}
                    iconSize={18}
                    disabledClassName="bg-neutral-200 dark:bg-neutral-700 opacity-60"
                  />
                </View>

                <View className="bg-neutral-50 dark:bg-neutral-700/50 p-4 rounded-lg mb-4">
                  <Text className="text-sm" style={{ color: secondaryTextColor }}>
                    This view shows all recordings stored on the device across all participants. You can play, delete individual recordings, or delete all recordings. Uploads must be done from the participant view.
                  </Text>
                </View>
              </View>
            }
            ListEmptyComponent={
              <View style={styles.centerContainer} className="mt-10">
                <MaterialCommunityIcons name="playlist-music-outline" size={60} color={iconColor || '#000'} />
                <Text style={[styles.emptyText, { color: textColor }]}>
                  No recordings found on this device.
                </Text>
              </View>
            }
            refreshControl={
              <RefreshControl
                refreshing={isRefreshing}
                onRefresh={onRefresh}
                tintColor={primaryColor}
                title="Pull to refresh..."
                titleColor={textColor}
              />
            }
            className="flex-1"
            contentContainerStyle={{ paddingBottom: 20 }}
          />
        ) : (
          // Participants Tab Content
          <ParticipantSelector
            currentParticipant={participantDetails}
            onSelectParticipant={handleParticipantSelected}
            onCreateNewParticipant={handleCreateNewParticipant}
          />
        )}
      </>
    );
  };

  // --- Main Render Logic ---
  return (
    <ThemedSafeAreaView className="flex-1">
      {isSetupMode ? (
        // ... (existing setup mode rendering logic)
        viewMode === 'select-participant' ? (
          <ParticipantSelector
            currentParticipant={participantDetails}
            onSelectParticipant={handleParticipantSelected}
            onCreateNewParticipant={handleCreateNewParticipant}
          />
        ) : viewMode === 'create-participant' ? (
          <SetupScreenContent
            onSetupComplete={handleSetupComplete}
            isNewParticipant={true}
            onCancel={() => {
              setIsSetupMode(false);
              setViewMode('participant');
              // If no participant exists after cancel, re-evaluate initial state
              if (!participantDetails) {
                loadInitialData(); // Re-check if setup is needed
              }
            }}
          />
        ) : viewMode === 'edit-participant' ? (
          <SetupScreenContent
            onSetupComplete={handleSetupComplete}
            initialDetails={participantDetails}
            onCancel={() => {
              setViewMode('participant');
              setIsSetupMode(false);
            }} />
        ) : null
      ) : viewMode === 'settings' ? (
        renderSettingsScreen()
      ) : (
        <>
          {(isDeletingAll) && (
            <View style={styles.overlay}>
              <ActivityIndicator size="large" color="#FFFFFF" />
              <Text style={styles.overlayText}>Deleting...</Text>
            </View>
          )}

          <FlatList
            data={displayedRecordings}
            keyExtractor={(item) => item.id || String(Math.random())}
            extraData={[
              playbackUiVersion,
              isDeletingId,
              isDeletingAll,
              participantDetails?.progress,
              uploadingItemId,
            ]}
            renderItem={({ item }) => (
              <RecordingListItem
                recording={item}
                onPlay={handlePlayRecording}
                onDelete={() => handleDeleteRecording(item.id, item.promptId)}
                isPlaying={playbackState.currentUri === item.localUri && playbackState.status === 'playing'}
                isDeleting={isDeletingId === item.id || isDeletingAll}
                isUploadingNow={uploadingItemId === item.id}
                showParticipantCode={false}
              />
            )}
            ListHeaderComponent={
              <View className="px-4 pt-4 pb-2">
                {/* Top Bar */}
                <View className="flex-row items-center justify-between mb-4 px-1">
                  <View style={{ width: 34 }} />
                  <Text
                    className="text-xl font-semibold text-neutral-800 dark:text-neutral-100 text-center flex-1 mx-2"
                    numberOfLines={1}
                    style={{ color: textColor }}
                  >
                    {participantDetails?.code ?? 'Loading...'}
                  </Text>
                  <Button
                    title="" icon="cog-outline" iconSize={26} onPress={goToSettings}
                    className="p-1" iconColor={iconColor || undefined}
                  />
                </View>

                {/* Participant Details Card */}
                {participantDetails && (
                  <View className="mb-3 p-4 bg-white dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-700 shadow-sm">
                    <View className="flex-row items-center justify-between mb-1.5">
                      <Text className="text-sm font-medium text-neutral-500 dark:text-neutral-400">
                        Participant Details:
                      </Text>
                      <Button
                        title="Edit" icon="account-edit-outline" iconColor={primaryColor || undefined} iconSize={16}
                        onPress={() => { setIsSetupMode(true); setViewMode('edit-participant'); }}
                        className="py-0.5 px-2 flex-row items-center bg-primary/10 dark:bg-primary/20 rounded-md"
                        textClassName="text-xs text-primary dark:text-primary-light ml-1 font-medium"
                      />
                    </View>
                    <Text className="text-sm" style={{ color: secondaryTextColor }}>
                      {[
                        participantDetails.dialect && `Dialect: ${participantDetails.dialect}`,
                        participantDetails.age_range && `Age: ${participantDetails.age_range}`,
                        participantDetails.gender && `Gender: ${participantDetails.gender}`
                      ].filter(Boolean).join('  •  ') || 'No optional details provided'}
                    </Text>
                    {/* --- Render Progress Bar --- */}
                    <View className="mt-4">
                      {/* Pass uploadProgress state to the render function */}
                      {renderProgress(displayedRecordings, uploadProgress)}
                    </View>
                  </View>
                )}

                {/* Recording Button */}
                <Button
                  title={displayedRecordings.length > 0 ? 'Continue Recording' : 'Start Recording'}
                  icon="microphone-plus"
                  iconColor='#F5f5f0'
                  onPress={() => router.push('/record')}
                  disabled={isDeletingAll || isUploading || participantDetails?.progress?.is_complete}
                  className={`
                    w-full flex-row items-center justify-center py-3.5 rounded-xl shadow-md my-3
                    ${participantDetails?.progress?.is_complete && !isUploading ? 'bg-neutral-400 dark:bg-neutral-600 opacity-70' : ''}
                    ${!participantDetails?.progress?.is_complete && !isUploading ? 'bg-success dark:bg-success-dark' : ''}
                    ${isUploading ? 'bg-neutral-400 dark:bg-neutral-600 opacity-70' : ''}
                  `}
                  textClassName="text-white text-lg font-semibold ml-2 dark:text-gray-100"
                  disabledClassName="bg-neutral-400 dark:bg-neutral-600 opacity-70"
                />
                {participantDetails?.progress?.is_complete && !isUploading && (
                  <Text className="text-xs text-center text-neutral-500 dark:text-neutral-400 -mt-2 mb-3">
                    All recordings completed.
                  </Text>
                )}

                {/* Recorded Items Header */}
                <View className="flex-row justify-between items-center mt-3 mb-1">
                  <Text className="text-lg font-semibold" style={{ color: textColor }}>
                    Recorded Items ({displayedRecordings.length})
                  </Text>
                  <Button
                    title="Delete All"
                    icon={isDeletingAll ? undefined : "delete-sweep-outline"}
                    onPress={handleDeleteAll}
                    disabled={!canDeleteAllDisplayed}
                    isLoading={isDeletingAll}
                    className="bg-danger/10 dark:bg-danger/25 px-3 py-1.5 rounded-lg flex-row items-center justify-center"
                    textClassName="text-danger dark:text-danger-light text-sm font-medium ml-1"
                    iconColor={dangerColor || undefined}
                    iconSize={18}
                    disabledClassName="bg-neutral-200 dark:bg-neutral-700 opacity-60"
                  />
                </View>
              </View>
            }
            ListEmptyComponent={
              <View style={styles.centerContainer} className="mt-10">
                <MaterialCommunityIcons name="playlist-music-outline" size={60} color={iconColor || '#000'} />
                <Text style={[styles.emptyText, { color: textColor }]}>
                  No recordings yet.
                </Text>
                <Text style={[styles.emptySubText, { color: secondaryTextColor }]}>
                  Press "{displayedRecordings.length > 0 ? 'Continue Recording' : 'Start Recording'}" above to begin.
                </Text>
              </View>
            }
            refreshControl={
              <RefreshControl
                refreshing={isRefreshing}
                onRefresh={onRefresh}
                tintColor={primaryColor}
                title="Pull to refresh..."
                titleColor={textColor}
              />
            }
            className="flex-1 px-2"
            contentContainerStyle={{ paddingBottom: 120 }}
          />

          {/* Upload Button Area */}
          <View className="absolute bottom-0 left-0 right-0 p-4 border-t border-neutral-200 bg-white/95 shadow-lg dark:bg-neutral-800/95 dark:border-neutral-700">
            <View className="flex-row justify-between items-center mb-2.5">
              <Text className="text-base" style={{ color: textColor }}>
                Pending Upload: <Text className="font-bold">{pendingToUploadCount}</Text>
              </Text>
              <View className="flex-row items-center">
                <View className={`w-2.5 h-2.5 rounded-full mr-1.5 ${networkAvailable ? "bg-success" : "bg-danger"}`} />
                <Text className={`text-xs font-medium ${networkAvailable ? "text-success-dark dark:text-success-light" : "text-danger-dark dark:text-danger-light"}`}>
                  {networkAvailable ? 'Online' : 'Offline'}
                </Text>
              </View>
            </View>
            <Button
              title={
                isUploading && uploadProgress
                  ? `Uploading ${uploadProgress.current} / ${uploadProgress.total}...`
                  : `Upload Pending (${pendingToUploadCount})`
              }
              icon={isUploading ? undefined : "cloud-upload-outline"}
              onPress={handleUploadAll}
              disabled={!canUpload}
              isLoading={isUploading}
              className="bg-primary dark:bg-primary-dark w-full flex-row items-center justify-center py-3 rounded-lg"
              textClassName="text-white text-lg font-semibold ml-2 dark:text-neutral-100"
              iconColor="white"
              disabledClassName="bg-primary-light dark:bg-primary/50 opacity-70"
            />
            {isUploading && uploadProgress && uploadProgress.total > 0 && (
              <View className="mt-2 h-1.5 bg-neutral-200 dark:bg-neutral-600 rounded-full overflow-hidden">
                <View
                  className="h-full bg-primary dark:bg-primary-dark rounded-full" // Use primary color for active upload progress
                  style={{ width: `${(uploadProgress.current / uploadProgress.total) * 100}%` }}
                />
              </View>
            )}
          </View>
        </>
      )}
    </ThemedSafeAreaView>
  );
}

// --- Styles ---
const styles = StyleSheet.create({
  centerContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
  },
  emptyText: {
    fontSize: 18,
    marginTop: 16,
    textAlign: 'center',
  },
  emptySubText: {
    fontSize: 14,
    marginTop: 4,
    textAlign: 'center',
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 10,
  },
  overlayText: {
    color: 'white',
    marginTop: 8,
    fontSize: 16,
  },
  tabButton: {
    paddingVertical: 12,
    paddingHorizontal: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tabIndicator: {
    position: 'absolute',
    bottom: 0,
    height: 2,
    left: 0,
    right: 0,
  },
});
