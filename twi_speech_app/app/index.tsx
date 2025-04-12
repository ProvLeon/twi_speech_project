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
  ALL_PARTICIPANTS_KEY,
  PARTICIPANT_DETAILS_KEY,
} from '@/lib/storage';
import { uploadRecording } from '@/lib/api';
import * as Network from 'expo-network';
import { RecordingMetadata, ParticipantDetails } from '@/types';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import SetupScreenContent from '@/components/SetupScreenContent';
import { ThemedSafeAreaView } from '@/components/ThemedSafeAreaView';
import { isValidCode } from '@/lib/utils';
import { useThemeColor } from '@/hooks/useThemeColor';
import { ParticipantSelector } from '@/components/ParticipantSelector';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Redirect } from "expo-router";


const WELCOME_SEEN_KEY = 'welcomeScreenSeen_v1';


type PlaybackStatus = 'idle' | 'loading' | 'playing' | 'error';
type ViewMode = 'participant' | 'settings' | 'select-participant' | 'create-participant' | 'edit-participant' | 'participant-recordings';




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

  // --- Refs instead of state for audio playback ---
  const playbackState = useRef({
    sound: null as Audio.Sound | null,
    status: 'idle' as PlaybackStatus,
    currentUri: null as string | null,
    isActive: false,
  }).current;

  // UI updates state
  const [playbackUiVersion, setPlaybackUiVersion] = useState(0);

  // All your callbacks (don't put any returns before all hooks are defined)
  const triggerHaptic = useCallback((type: Haptics.ImpactFeedbackStyle = Haptics.ImpactFeedbackStyle.Light) => {
    Haptics.impactAsync(type).catch(console.error);
  }, []);

  const updatePlaybackUI = useCallback(() => {
    setPlaybackUiVersion(prev => prev + 1);
  }, []);

  const checkNetwork = useCallback(async () => {
    try {
      const state = await Network.getNetworkStateAsync();
      // Safely check properties with optional chaining
      const isConnected = !!(state?.isConnected) && !!(state?.isInternetReachable);
      setNetworkAvailable(prev => {
        if (prev !== isConnected) {
          console.log(`HomeScreen: Network status changed: ${isConnected ? 'Online' : 'Offline'}`);
        }
        return isConnected;
      });
    } catch (error) {
      console.error("HomeScreen: Error checking network:", error);
      setNetworkAvailable(false);
    }
  }, []);

  const loadAllRecordings = useCallback(async () => {
    console.log("HomeScreen: Loading ALL recordings...");
    try {
      const storedRecordings = await getPendingRecordings();
      // Make sure we have valid recordings before sorting
      if (Array.isArray(storedRecordings)) {
        storedRecordings.sort((a, b) => b.timestamp - a.timestamp);
        setAllRecordings(storedRecordings);
        console.log(`HomeScreen: Loaded ${storedRecordings.length} total recordings.`);
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
      // Make sure we have a valid array of participants
      if (Array.isArray(participants)) {
        setParticipantCount(participants.length);
      } else {
        console.error("HomeScreen: getAllParticipants returned non-array:", participants);
        setParticipantCount(0);
      }
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

  const displayedRecordings = useMemo(() => {
    if (viewMode === 'settings') return allRecordings;
    if ((viewMode === 'participant' || viewMode === 'select-participant' || viewMode === 'create-participant') && participantDetails?.code) {
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
        // Check if we have any participants already
        const allParticipants = await getAllParticipants();
        if (Array.isArray(allParticipants) && allParticipants.length > 0) {
          // If we have participants, go to select screen instead
          setViewMode('select-participant');
        } else {
          // Otherwise go to create screen
          setViewMode('create-participant');
        }
        setIsSetupMode(true);
      } else {
        setIsSetupMode(false);
        setViewMode('participant');
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
    console.log("HomeScreen: Unloading playback sound...");
    if (playbackState.sound) {
      try {
        await playbackState.sound.unloadAsync();
      } catch (e) {
        console.error("Unload error:", e);
      }

      // Update state object properties
      playbackState.sound = null;
      playbackState.status = 'idle';
      playbackState.currentUri = null;
      playbackState.isActive = false;

      // Update UI
      updatePlaybackUI();
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

    // Rest of the function remains the same but using playbackState...
    try {
      // Configure audio mode
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
        interruptionModeIOS: InterruptionModeIOS.DoNotMix,
        interruptionModeAndroid: InterruptionModeAndroid.DoNotMix,
        shouldDuckAndroid: true,  // Added with default
        playThroughEarpieceAndroid: false,  // Added with default
        staysActiveInBackground: false,     // Added with default
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

      // Update refs
      // At the end:
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
    if (isDeletingAll) return;

    triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
    Alert.alert(
      "Delete Recording?",
      `Are you sure you want to delete the recording for prompt "${promptId?.replace(/_/g, ' ') || 'Unknown'}"? This cannot be undone.`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: async () => {
            setIsDeletingId(id);
            try {
              const deleted = await deletePendingRecordingById(id);
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
  }, [isDeletingAll, triggerHaptic, loadAllRecordings]);

  const handleDeleteAll = useCallback(() => {
    if (isDeletingAll || displayedRecordings.length === 0) return;

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
        text: buttonText,
        style: "destructive",
        onPress: async () => {
          console.log(`HomeScreen: Starting Delete All process (Mode: ${viewMode})...`);
          setIsDeletingAll(true);

          // Make sure to stop playback when deleting recordings
          await unloadPlaybackSound();

          let result = { deletedCount: 0, failedCount: 0 };

          try {
            if (isParticipantView && participantDetails?.code) {
              result = await deleteAllRecordingsForParticipant(participantDetails.code);
            } else {
              result = await deleteAllDeviceRecordings();
            }
          } catch (error) {
            console.error(`HomeScreen: Error during deleteAll call (Mode: ${viewMode}):`, error);
            result.failedCount = count;
          }

          console.log(`HomeScreen: Delete All finished. Deleted: ${result.deletedCount}, Failed: ${result.failedCount}`);
          await loadAllRecordings();
          setIsDeletingAll(false);

          let alertMessage = `${result.deletedCount} recording(s) deleted successfully.`;
          if (result.failedCount > 0) {
            alertMessage += ` ${result.failedCount} failed to delete.`;
          }
          Alert.alert("Deletion Complete", alertMessage);
        },
      },
    ]);
  }, [isDeletingAll, displayedRecordings, viewMode, participantDetails, triggerHaptic, loadAllRecordings, unloadPlaybackSound]);

  const handleUploadAll = useCallback(async () => {
    if (viewMode === 'settings') {
      Alert.alert("Action Not Available", "Please switch back to the participant view to upload recordings.");
      return;
    }
    if (!participantDetails) {
      Alert.alert("Setup Needed", "Participant details are not set.");
      return;
    }

    triggerHaptic();
    await checkNetwork();

    if (!networkAvailable) {
      return Alert.alert("Offline", "No internet connection available for upload.");
    }

    // Make sure we filter safely
    const toUpload = Array.isArray(displayedRecordings) ?
      displayedRecordings.filter(rec => !rec.uploaded) : [];

    if (toUpload.length === 0) {
      return Alert.alert("Up to Date", `No recordings waiting for upload for ${participantDetails.code}.`);
    }
    if (isUploading) return;

    setIsUploading(true);
    Alert.alert("Upload Started", `Attempting to upload ${toUpload.length} recording(s) for ${participantDetails.code}...`);

    let successCount = 0;
    let failCount = 0;
    const uploadPromises = toUpload.map(async (recordingMeta) => {
      try {
        if (recordingMeta.participantCode !== participantDetails.code) {
          console.warn(`[Upload] Skipping ${recordingMeta.id} - code mismatch.`);
          return false;
        }

        // Verify file exists before upload
        const fileExists = await verifyFileExists(recordingMeta.localUri);
        if (!fileExists) {
          console.error(`[Upload] File doesn't exist: ${recordingMeta.localUri}`);
          return false;
        }

        const success = await uploadRecording(recordingMeta);
        if (success) {
          await updateRecordingUploadedStatus(recordingMeta.id, true);
          return true;
        } else {
          return false;
        }
      } catch (error) {
        console.error(`[Upload] Error uploading recording ${recordingMeta.id}:`, error);
        return false;
      }
    });

    const results = await Promise.all(uploadPromises);
    successCount = results.filter(Boolean).length;
    failCount = results.length - successCount;

    setIsUploading(false);

    let finalMessage = `${successCount} uploaded successfully.`;
    if (failCount > 0) {
      finalMessage += ` ${failCount} failed.`;
    }
    Alert.alert("Upload Complete", finalMessage);

    await loadAllRecordings();
  }, [viewMode, participantDetails, networkAvailable, displayedRecordings, isUploading, checkNetwork, triggerHaptic, loadAllRecordings, verifyFileExists]);

  const handleSetupComplete = useCallback(async (details: ParticipantDetails) => {
    console.log("HomeScreen: [handleSetupComplete] Received details:", details);
    setIsLoading(true);
    try {
      if (!details || !isValidCode(details.code)) {
        throw new Error("Invalid participant code format.");
      }
      const savedSuccessfully = await saveParticipantDetails(details);
      if (!savedSuccessfully) {
        throw new Error("Failed to save participant details to storage.");
      }
      console.log("HomeScreen: [handleSetupComplete] Participant details saved.");

      setParticipantDetails(details);
      setIsSetupMode(false);
      setViewMode('participant');
      await loadAllRecordings();
      await loadParticipantCount();
      console.log("HomeScreen: [handleSetupComplete] Setup process complete.");
    } catch (error: any) {
      console.error("HomeScreen: [handleSetupComplete] Error:", error);
      Alert.alert("Error", `Failed to finalize setup: ${error.message || 'Unknown error'}`);
      setIsLoading(false);
    }
  }, [loadAllRecordings, loadParticipantCount]);

  const handleManageRecordings = useCallback(() => {
    setViewMode('participant-recordings');
  }, []);

  const handleParticipantSelected = useCallback((participant: ParticipantDetails | null) => {
    if (participant) {
      setParticipantDetails(participant);
      setIsSetupMode(false);
      setViewMode('participant');
    }
  }, []);

  const handleCreateNewParticipant = useCallback(() => {
    setViewMode('create-participant');
    setIsSetupMode(true);
  }, []);

  const goToSettings = useCallback(() => {
    setViewMode('settings');
    setSettingsTabIndex(0); // Default to All Recordings tab
    triggerHaptic();
  }, [triggerHaptic]);

  // useEffect(() => {
  //   const checkWelcomeScreen = async () => {
  //     try {
  //       const welcomeSeen = await AsyncStorage.getItem(WELCOME_SEEN_KEY);
  //       setHasSeenWelcome(welcomeSeen === 'true');
  //     } catch (e) {
  //       console.error('Error checking welcome screen state:', e);
  //     } finally {
  //       setIsCheckingWelcome(false);
  //     }
  //   };

  //   checkWelcomeScreen();
  // }, []);

  // Redirect if welcome hasn't been seen
  // if (!isCheckingWelcome && !hasSeenWelcome) {
  //   return <Redirect href="/welcome" />;
  // }


  // --- Effects ---
  useEffect(() => {

    let title = "Twi Speech Recorder";
    if (!isSetupMode && participantDetails?.code) {
      title = viewMode === 'settings'
        ? `Settings (${participantDetails.code})`
        : `Recordings (${participantDetails.code})`;
    } else if (isSetupMode) {
      if (viewMode === 'select-participant') {
        title = "Select Participant";
      } else if (viewMode === 'create-participant') {
        title = "New Participant";
      } else {
        title = "Participant Setup";
      }
    }
    navigation.setOptions({ title });
  }, [navigation, participantDetails, viewMode, isSetupMode]);

  useEffect(() => {
    loadInitialData();
    return () => { unloadPlaybackSound(); };
  }, [loadInitialData, unloadPlaybackSound]);

  useFocusEffect(
    useCallback(() => {
      if (!isSetupMode) {
        console.log("HomeScreen: Focus detected, reloading data...");
        loadInitialData();

        // Always unload playback on focus - don't resume playing when coming back to screen
        unloadPlaybackSound();
      } else {
        console.log("HomeScreen: Focus detected, but in setup mode - skipping reload.");
      }

      return () => {
        // Cleanup on unfocus
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
      } finally {
        setIsCheckingWelcome(false);
      }
    };

    checkWelcomeScreen();
  }, []);

  // All your other useEffects should be here

  // Now, AFTER defining ALL hooks, we can conditionally render


  // --- Memos ---
  const pendingToUploadCount = useMemo(() => {
    if (viewMode !== 'participant' || !participantDetails) return 0;
    return Array.isArray(displayedRecordings) ?
      displayedRecordings.filter(rec => !rec.uploaded).length : 0;
  }, [displayedRecordings, viewMode, participantDetails]);

  const canUpload = useMemo(() => (
    !isUploading &&
    pendingToUploadCount > 0 &&
    networkAvailable &&
    viewMode === 'participant'
  ), [isUploading, pendingToUploadCount, networkAvailable, viewMode]);

  const canDeleteAllDisplayed = useMemo(() => (
    Array.isArray(displayedRecordings) &&
    displayedRecordings.length > 0 &&
    !isDeletingAll &&
    !isUploading
  ), [displayedRecordings, isDeletingAll, isUploading]);

  if (isCheckingWelcome) {
    return (
      <ThemedSafeAreaView className="flex-1 justify-center items-center">
        <ActivityIndicator size="large" color={primaryColor} />
      </ThemedSafeAreaView>
    );
  }

  if (!hasSeenWelcome) {
    // AsyncStorage.setItem(WELCOME_SEEN_KEY, 'false');
    // AsyncStorage.setItem(ALL_PARTICIPANTS_KEY, '')
    // AsyncStorage.setItem(PARTICIPANT_DETAILS_KEY, '')
    router.replace('/welcome');
  }

  // --- Render Methods ---
  const renderSettingsScreen = () => {
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
            extraData={[playbackUiVersion, isDeletingId, isDeletingAll]}
            renderItem={({ item }) => (
              <RecordingListItem
                recording={item}
                onPlay={handlePlayRecording}
                onDelete={() => handleDeleteRecording(item.id, item.promptId)}
                isPlaying={playbackState.currentUri === item.localUri && playbackState.status === 'playing'}
                isDeleting={isDeletingId === item.id || isDeletingAll}
                showParticipantCode={viewMode === 'settings' ? true : viewMode === 'participant'}
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
                    This view shows all recordings stored on the device across all participants. You can play, delete individual recordings, or delete all recordings.
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
      {isLoading ? (
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" color={primaryColor} />
          <Text style={[styles.loadingText, { color: textColor }]}>Loading Data...</Text>
        </View>
      ) : isSetupMode ? (
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
              setIsSetupMode(false)
              setViewMode('participant')
              { !participantDetails && setHasSeenWelcome(false) }
            }}
          />
        ) : viewMode === 'edit-participant' ? (
          <SetupScreenContent onSetupComplete={handleSetupComplete} initialDetails={participantDetails} onCancel={() => {
            setViewMode('participant')
            setIsSetupMode(false)
          }} />
        ) : null
      ) : viewMode === 'settings' ? (
        renderSettingsScreen()
      ) : (
        <>
          {(isUploading || isDeletingAll) && (
            <View style={styles.overlay}>
              <ActivityIndicator size="large" color="#FFFFFF" />
              <Text style={styles.overlayText}>{isUploading ? 'Uploading...' : 'Deleting...'}</Text>
            </View>
          )}

          <FlatList
            data={displayedRecordings || []}
            keyExtractor={(item) => item.id || String(Math.random())}
            extraData={[
              playbackUiVersion, // Use the version counter for minimal re-renders
              isDeletingId,
              isDeletingAll,
              viewMode,
              pendingToUploadCount,
              allRecordings.length
            ]}
            renderItem={({ item }) => (
              <RecordingListItem
                recording={item}
                onPlay={handlePlayRecording}
                onDelete={() => handleDeleteRecording(item.id, item.promptId)}
                isPlaying={playbackState.currentUri === item.localUri && playbackState.status === 'playing'}
                isDeleting={isDeletingId === item.id || isDeletingAll}
                showParticipantCode={viewMode === 'participant' ? true : false}
              />
            )}
            ListHeaderComponent={
              <View className="px-4 pt-4 pb-2">
                <View className="flex-row items-center justify-end mb-4 px-1">
                  {viewMode !== 'participant' && (
                    <Button
                      title=""
                      icon="arrow-left-circle-outline"
                      iconSize={26}
                      onPress={() => setViewMode('participant')}
                      className="p-1"
                      iconColor={iconColor || undefined}
                    />
                  )}




                  {/* <Text
                    className="text-xl font-semibold text-neutral-800 dark:text-neutral-100 text-center flex-1 mx-2"
                    numberOfLines={1}
                    style={{ color: textColor }}
                  >
                    {viewMode === 'participant' ? (participantDetails?.code ?? 'Session') : 'Settings / All'}
                  </Text> */}


                </View>

                {viewMode === 'participant' && participantDetails && (
                  <View className="mb-3">
                    <View className="flex-row items-center justify-between mb-1">
                      <Text
                        className="text-xs font-medium text-neutral-500 dark:text-neutral-400"
                        style={{ color: secondaryTextColor }}
                      >
                        Current Participant:
                      </Text>
                      {viewMode === 'participant' ? (
                        <Button
                          title=""
                          icon="cog-outline"
                          iconSize={26}
                          onPress={goToSettings}
                          className="p-1 right-0"
                          iconColor={iconColor || undefined}
                        />
                      ) : (
                        <View style={{ width: 34 }} />
                      )}
                    </View>

                    <View className="flex-row items-center">
                      <View className="flex-1">
                        <Text className="text-base font-semibold items-center justify-center" style={{ color: textColor }}>
                          {participantDetails?.code || "Unknown"} <Button
                            title=""
                            icon="account-edit-outline"
                            iconColor={primaryColor || undefined}
                            iconSize={20}
                            onPress={() => {
                              // Switch to setup mode with current details for editing
                              setIsSetupMode(true);
                              setViewMode('edit-participant'); // Reuse the create view but with initialDetails
                            }}
                            className="py-0.5 px-2 flex-row items-center bg-transparent"
                            textClassName="text-xs text-primary dark:text-primary-light ml-1"
                          />
                        </Text>
                        <Text className="text-xs" style={{ color: secondaryTextColor }}>
                          {[
                            participantDetails?.dialect,
                            participantDetails?.age_range,
                            participantDetails?.gender
                          ].filter(Boolean).join(' • ') || 'No additional details provided'}
                        </Text>
                      </View>

                      <Text className="text-xs py-1 px-2 rounded-full bg-neutral-100 dark:bg-neutral-700" style={{ color: secondaryTextColor }}>
                        {participantCount} {participantCount === 1 ? 'Participant' : 'Participants'}
                      </Text>
                    </View>
                  </View>
                )}

                {viewMode === 'participant' && (
                  <Button
                    title={`${displayedRecordings.length > 0 ? 'Continue Recording' : 'Start Recording'}`}
                    icon="microphone-plus"
                    iconColor='#F5f5f0'
                    onPress={() => router.push('/record')}
                    className="bg-success dark:bg-success-dark w-full flex-row items-center justify-center py-3.5 rounded-xl shadow-md my-3"
                    textClassName="text-white text-lg font-semibold ml-2 dark:text-gray-100"
                    disabled={isDeletingAll || isUploading}
                    disabledClassName="bg-neutral-400 dark:bg-neutral-600 opacity-70"
                  />
                )}

                <View className="flex-row justify-between items-center mt-3 mb-1">
                  <Text
                    className="text-lg font-semibold text-neutral-700 dark:text-neutral-300"
                    style={{ color: textColor }}
                  >
                    {viewMode === 'settings' ? 'All Device Recordings' : 'Recorded Items'} ({displayedRecordings?.length || 0})
                  </Text>

                  <Button
                    title={viewMode === 'settings' ? "Delete User's" : "Delete All"}
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
                  {viewMode === 'participant' ? 'No recordings for this participant yet.' : 'No recordings found on this device.'}
                </Text>
                {viewMode === 'participant' && (
                  <Text style={[styles.emptySubText, { color: secondaryTextColor }]}>
                    Press "Start / Continue Recording" above.
                  </Text>
                )}
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
            contentContainerStyle={{ paddingBottom: viewMode === 'participant' ? 120 : 20 }}
          />

          {viewMode === 'participant' && (
            <View className="absolute bottom-0 left-0 right-0 p-4 border-t border-neutral-200 bg-white/95 shadow-lg dark:bg-neutral-800/95 dark:border-neutral-700">
              <View className="flex-row justify-between items-center mb-2.5">
                <Text className="text-base text-neutral-700 dark:text-neutral-200" style={{ color: textColor }}>
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
                title={isUploading ? "Uploading..." : `Upload All (${pendingToUploadCount})`}
                icon={isUploading ? undefined : "cloud-upload-outline"}
                onPress={handleUploadAll}
                disabled={!canUpload}
                isLoading={isUploading}
                className="bg-primary dark:bg-primary-dark w-full flex-row items-center justify-center py-3 rounded-lg"
                textClassName="text-white text-lg font-semibold ml-2 dark:text-neutral-100"
                iconColor="white"
                disabledClassName="bg-primary-light dark:bg-primary/50 opacity-70"
              />
            </View>
          )}
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
