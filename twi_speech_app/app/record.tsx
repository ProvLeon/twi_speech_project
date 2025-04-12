import * as FileSystem from 'expo-file-system';
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { View, Text, Alert, ScrollView, ActivityIndicator, TouchableOpacity } from 'react-native';
import { useFocusEffect, useRouter, useNavigation } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { Audio, AVPlaybackStatus, InterruptionModeAndroid, InterruptionModeIOS } from 'expo-av';
import { Button } from '@/components/Button';
import { PromptDisplay } from '@/components/PromptDisplay';
import { SoundWaveAnimation } from '@/components/SoundWaveAnimation';
import { useAudioRecorder } from '@/hooks/useAudioRecorder';
import { getParticipantDetails, savePendingRecording, getPendingRecordings } from '@/lib/storage';
import { RECORDING_SCRIPT } from '@/constants/script';
import { RecordingMetadata, ParticipantDetails } from '@/types';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { ThemedSafeAreaView } from '@/components/ThemedSafeAreaView';
import { useThemeColor } from '@/hooks/useThemeColor';
import { isValidCode } from '@/lib/utils';

// Helper functions defined outside component
const getNextPromptIndex = (currentIndex: number): number => {
  if (currentIndex >= RECORDING_SCRIPT.length - 1) return -1;
  return currentIndex + 1;
};

const getPreviousPromptIndex = (currentIndex: number): number => {
  if (currentIndex <= 0) return 0;
  return currentIndex - 1;
};

type PlaybackStatus = 'idle' | 'loading' | 'playing' | 'paused' | 'stopped' | 'error';

export default function RecordScreen() {
  // To better diagnose re-renders
  console.log("RecordScreen: Render Start", new Date().toISOString());

  // --- Hook definitions (all at the top level) ---
  const navigation = useNavigation();
  const router = useRouter();
  const primaryColor = useThemeColor({}, 'tint');
  const iconColor = useThemeColor({}, 'icon');
  const dangerColor = useThemeColor({}, 'danger');

  // State hooks
  const [participantDetails, setParticipantDetails] = useState<ParticipantDetails | null>(null);
  const [currentPromptIndex, setCurrentPromptIndex] = useState<number | null>(null);
  const [isSessionComplete, setIsSessionComplete] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [initializationError, setInitializationError] = useState<string | null>(null);
  const [playbackStatus, setPlaybackStatus] = useState<PlaybackStatus>('idle');
  const [playbackError, setPlaybackError] = useState<string | null>(null);
  const [currentPromptRecordingUri, setCurrentPromptRecordingUri] = useState<string | null>(null);
  const [currentPromptRecordingDuration, setCurrentPromptRecordingDuration] = useState<number | undefined>(undefined);

  // Refs - use refs for values that shouldn't trigger re-renders
  const playbackSoundRef = useRef<Audio.Sound | null>(null);
  const isMounted = useRef(true);
  const isPlayingRef = useRef(false); // Track playback without causing re-renders

  // Custom hooks
  const {
    startRecording, stopRecordingAndSave, recordingStatus,
    error: recorderError
  } = useAudioRecorder();

  // --- Derived state calculations (not hooks) ---
  const currentPrompt = (currentPromptIndex !== null && currentPromptIndex >= 0 && currentPromptIndex < RECORDING_SCRIPT.length)
    ? RECORDING_SCRIPT[currentPromptIndex]
    : null;

  const progressPercent = isSessionComplete ? 100 : Math.round(((currentPromptIndex ?? 0) / RECORDING_SCRIPT.length) * 100);

  // UI state flags
  const hasRecordedCurrentPrompt = !!currentPromptRecordingUri;
  const isRecording = recordingStatus === 'recording';
  const isPlaybackActive = playbackStatus === 'playing' || playbackStatus === 'loading';
  const canRecord = (recordingStatus === 'ready' || recordingStatus === 'stopped') && !isSessionComplete && !isPlaybackActive;
  const showReplay = hasRecordedCurrentPrompt && !isRecording;
  const canGoNext = hasRecordedCurrentPrompt && !isRecording && !isPlaybackActive && !isSessionComplete && currentPromptIndex !== null && currentPromptIndex < RECORDING_SCRIPT.length - 1;
  const canGoPrevious = !isRecording && !isPlaybackActive && currentPromptIndex !== null && currentPromptIndex > 0;

  // Status text calculation
  let statusText = 'Initializing...';
  if (!isLoading) {
    if (isRecording) statusText = 'Recording...';
    else if (recorderError) statusText = `Recorder Error: ${recorderError}`;
    else if (playbackStatus === 'loading') statusText = 'Loading Playback...';
    else if (playbackStatus === 'playing') statusText = 'Playing...';
    else if (playbackStatus === 'paused') statusText = 'Paused';
    else if (playbackStatus === 'error') statusText = `Playback Error: ${playbackError || 'Unknown'}`;
    else if (recordingStatus === 'stopped') statusText = 'Saved!';
    else if (recordingStatus === 'ready' && hasRecordedCurrentPrompt) statusText = 'Ready to Re-record';
    else if (recordingStatus === 'ready') statusText = 'Ready to Record';
    else if (recordingStatus === 'requesting') statusText = 'Requesting Permissions...';
    else if (recordingStatus === 'error') statusText = 'Microphone Permission Denied';
    else statusText = 'Ready';
  }

  // Verify file URI and check if file exists
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

  // --- All callbacks ---
  const triggerHaptic = useCallback((type: Haptics.ImpactFeedbackStyle = Haptics.ImpactFeedbackStyle.Light) => {
    Haptics.impactAsync(type).catch(console.error);
  }, []);

  const unloadPlaybackSound = useCallback(async () => {
    console.log("Attempting to unload playback sound...");
    if (playbackSoundRef.current) {
      try {
        await playbackSoundRef.current.unloadAsync();
        console.log("Playback sound unloaded successfully");
      } catch (e) {
        console.error("Error during sound unloading:", e);
      } finally {
        playbackSoundRef.current = null;
        isPlayingRef.current = false;
        // Only reset status if it wasn't already idle/stopped/error
        if (!['idle', 'stopped', 'error'].includes(playbackStatus)) {
          setPlaybackStatus('idle');
        }
      }
    } else {
      console.log("No playback sound to unload");
    }
  }, [playbackStatus]);

  const loadRecordingForPrompt = useCallback(async (promptId: string | undefined, code: string | undefined) => {
    if (!promptId || !code) {
      console.log(`[loadRecordingForPrompt] Invalid promptId (${promptId}) or code (${code})`);
      setCurrentPromptRecordingUri(null);
      setCurrentPromptRecordingDuration(undefined);
      return;
    }

    console.log(`[loadRecordingForPrompt] Loading for prompt ${promptId}, participant ${code}`);

    try {
      const allRecordings = await getPendingRecordings();
      console.log(`[loadRecordingForPrompt] Found ${allRecordings.length} total recordings`);

      const existingRecording = allRecordings.find(
        rec => rec.promptId === promptId && rec.participantCode === code
      );

      console.log(`[loadRecordingForPrompt] Found recording for this prompt:`,
        existingRecording ? "yes" : "no");

      if (existingRecording?.localUri) {
        // Always store the URI, but check if file exists
        const fileExists = await verifyFileExists(existingRecording.localUri);

        if (fileExists) {
          console.log(`[loadRecordingForPrompt] File exists: ${existingRecording.localUri}`);
          setCurrentPromptRecordingUri(existingRecording.localUri);
          setCurrentPromptRecordingDuration(existingRecording.recordingDuration);
        } else {
          console.warn(`[loadRecordingForPrompt] File doesn't exist at path: ${existingRecording.localUri}`);

          // Still set the URI in state so the UI shows the replay button,
          // but we'll do another check before actual playback
          setCurrentPromptRecordingUri(existingRecording.localUri);
          setCurrentPromptRecordingDuration(existingRecording.recordingDuration);
        }
      } else {
        console.log(`[loadRecordingForPrompt] No recording found or URI is empty`);
        setCurrentPromptRecordingUri(null);
        setCurrentPromptRecordingDuration(undefined);
      }
    } catch (error) {
      console.error("[loadRecordingForPrompt] Error loading specific recording:", error);
      setCurrentPromptRecordingUri(null);
      setCurrentPromptRecordingDuration(undefined);
    }
  }, [verifyFileExists]);

  const onPlaybackStatusUpdate = useCallback((status: AVPlaybackStatus) => {
    if (!status.isLoaded) {
      if (status.error) {
        console.error(`Playback Error: ${status.error}`);
        setPlaybackError(`Error: ${status.error}`);
        setPlaybackStatus('error');
        isPlayingRef.current = false;
        playbackSoundRef.current = null;
      } else if (playbackStatus !== 'idle') {
        setPlaybackStatus('stopped');
        isPlayingRef.current = false;
      }
    } else {
      if (status.didJustFinish) {
        console.log("Playback finished naturally.");
        setPlaybackStatus('stopped');
        isPlayingRef.current = false;
        playbackSoundRef.current?.setPositionAsync(0).catch(console.error);
      } else if (status.isPlaying) {
        isPlayingRef.current = true;
        if (playbackStatus !== 'playing') {
          setPlaybackStatus('playing');
        }
      } else if (!status.isPlaying && playbackStatus === 'playing') {
        isPlayingRef.current = false;
        setPlaybackStatus('paused');
      }
    }
  }, [playbackStatus]);

  const handlePlayPause = useCallback(async () => {
    triggerHaptic();
    setPlaybackError(null);

    // Debug logging
    console.log("handlePlayPause called:", {
      URI: currentPromptRecordingUri,
      playbackStatus,
      isPlayingRef: isPlayingRef.current,
      hasSound: !!playbackSoundRef.current
    });

    const currentSound = playbackSoundRef.current;

    // Case 1: Currently playing - pause it
    if (playbackStatus === 'playing' && currentSound) {
      console.log("Pausing playback");
      try {
        await currentSound.pauseAsync();
        isPlayingRef.current = false;
        setPlaybackStatus('paused');
      } catch (error: any) {
        console.error("Error pausing playback:", error);
        setPlaybackError(`Failed to pause: ${error.message}`);
      }
    }
    // Case 2: Paused - resume it
    else if (playbackStatus === 'paused' && currentSound) {
      console.log("Resuming playback");
      try {
        await currentSound.playAsync();
        isPlayingRef.current = true;
        setPlaybackStatus('playing');
      } catch (error: any) {
        console.error("Error resuming playback:", error);
        setPlaybackError(`Failed to resume: ${error.message}`);
      }
    }
    // Case 3: Not playing yet but we have a URI - start playback
    else if (currentPromptRecordingUri && !['playing', 'loading'].includes(playbackStatus)) {
      console.log("Starting new playback");

      // Cleanup previous sound if any
      await unloadPlaybackSound();
      setPlaybackStatus('loading');

      try {
        // Set audio mode first
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: false,
          playsInSilentModeIOS: true,
          interruptionModeIOS: InterruptionModeIOS.DoNotMix,
          interruptionModeAndroid: InterruptionModeAndroid.DoNotMix,
        });

        // Verify file exists again before trying to play
        const fileExists = await verifyFileExists(currentPromptRecordingUri);
        if (!fileExists) {
          throw new Error(`Audio file doesn't exist at: ${currentPromptRecordingUri}`);
        }

        console.log("Creating sound object with URI:", currentPromptRecordingUri);

        // Create and load the sound
        const { sound } = await Audio.Sound.createAsync(
          { uri: currentPromptRecordingUri },
          { shouldPlay: true, progressUpdateIntervalMillis: 100 },
          onPlaybackStatusUpdate
        );

        console.log("Sound created successfully");
        playbackSoundRef.current = sound;
        isPlayingRef.current = true;
        setPlaybackStatus('playing');

      } catch (error: any) {
        console.error("Error creating/starting sound:", error);
        setPlaybackError(`Playback failed: ${error.message}`);
        setPlaybackStatus('error');
        playbackSoundRef.current = null;
        isPlayingRef.current = false;
      }
    } else {
      console.log("Play/Pause ignored - Status:", playbackStatus, "URI:", currentPromptRecordingUri);
    }
  }, [currentPromptRecordingUri, playbackStatus, unloadPlaybackSound, onPlaybackStatusUpdate, triggerHaptic, verifyFileExists]);

  const handleRecordPress = useCallback(() => {
    if (playbackStatus !== 'idle' && playbackStatus !== 'stopped') {
      unloadPlaybackSound();
    }

    console.log("Starting new recording...");
    setCurrentPromptRecordingUri(null);
    setCurrentPromptRecordingDuration(undefined);
    triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
    startRecording();
  }, [startRecording, unloadPlaybackSound, playbackStatus, triggerHaptic]);

  const handleStopPress = useCallback(async () => {
    if (recordingStatus !== 'recording' || !participantDetails || !currentPrompt) {
      console.warn("Stop ignored: Not recording or missing details/prompt.");
      return;
    }

    triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
    console.log("Stopping recording...");

    try {
      console.log("Calling stopRecordingAndSave with:", {
        code: participantDetails.code,
        promptId: currentPrompt.id,
        promptText: currentPrompt.text
      });

      const savedMetadata = await stopRecordingAndSave(
        participantDetails.code,
        currentPrompt.id,
        currentPrompt.text,
        participantDetails
      );

      if (savedMetadata) {
        console.log("[RecordScreen] Recording saved with URI:", savedMetadata.localUri);

        // Verify the saved file exists
        const fileExists = await verifyFileExists(savedMetadata.localUri);
        console.log(`File exists check after saving: ${fileExists}`);

        if (!fileExists) {
          throw new Error("Recording file was not found after saving");
        }

        // Save metadata to storage
        const savedToStorage = await savePendingRecording(savedMetadata);

        if (savedToStorage) {
          setCurrentPromptRecordingUri(savedMetadata.localUri);
          setCurrentPromptRecordingDuration(savedMetadata.recordingDuration);
          console.log("[RecordScreen] State updated with new recording URI:", savedMetadata.localUri);
        } else {
          throw new Error("Failed to save recording metadata locally");
        }
      } else {
        throw new Error("Could not save the recording file or generate metadata");
      }
    } catch (error: any) {
      console.error("Error in handleStopPress:", error);
      Alert.alert("Recording Error", error.message || "An error occurred while saving the recording.");

      // Reset UI state
      setCurrentPromptRecordingUri(null);
      setCurrentPromptRecordingDuration(undefined);
    }
  }, [recordingStatus, participantDetails, currentPrompt, stopRecordingAndSave, triggerHaptic, verifyFileExists]);

  const handleRecordStopPress = useCallback(() => {
    if (recordingStatus === 'recording') {
      handleStopPress();
    } else if (canRecord) {
      handleRecordPress();
    }
  }, [recordingStatus, canRecord, handleStopPress, handleRecordPress]);

  const handleNextPrompt = useCallback(() => {
    if (!hasRecordedCurrentPrompt || isRecording || isPlaybackActive || isSessionComplete || currentPromptIndex === null) {
      if (!hasRecordedCurrentPrompt && !isSessionComplete) {
        Alert.alert("Recording Required", "Please record the current prompt before proceeding.");
        triggerHaptic(Haptics.ImpactFeedbackStyle.Heavy);
      }
      return;
    }

    triggerHaptic();
    // Ensure playback stops when changing prompts
    if (playbackSoundRef.current) {
      unloadPlaybackSound();
    }

    const nextIndex = getNextPromptIndex(currentPromptIndex);
    if (nextIndex === -1) {
      setIsSessionComplete(true);
      Alert.alert("Session Complete!", "All prompts recorded for this participant. Go back to review and upload.");
    } else {
      setCurrentPromptIndex(nextIndex);
    }
  }, [
    hasRecordedCurrentPrompt, isRecording, isPlaybackActive, isSessionComplete,
    currentPromptIndex, triggerHaptic, unloadPlaybackSound
  ]);

  const handlePreviousPrompt = useCallback(() => {
    if (isRecording || isPlaybackActive || currentPromptIndex === null || currentPromptIndex <= 0) return;

    triggerHaptic();
    // Ensure playback stops when changing prompts
    if (playbackSoundRef.current) {
      unloadPlaybackSound();
    }

    const prevIndex = getPreviousPromptIndex(currentPromptIndex);
    setCurrentPromptIndex(prevIndex);
  }, [isRecording, isPlaybackActive, currentPromptIndex, triggerHaptic, unloadPlaybackSound]);

  const initializeScreen = useCallback(async (isActive: boolean) => {
    try {
      console.log("initializeScreen: Getting participant details...");
      const fetchedDetails = await getParticipantDetails();
      if (!isActive) return;
      console.log("initializeScreen: Participant details:", fetchedDetails);

      if (!fetchedDetails || !isValidCode(fetchedDetails.code)) {
        console.log("initializeScreen: No valid participant found, navigating to setup.");
        if (isActive) {
          Alert.alert("Setup Required", "No active participant found.", [{
            text: 'Go to Participant Setup',
            onPress: () => router.replace('/')
          }]);
        }
        return;
      }

      console.log("initializeScreen: Getting recordings...");
      const recordings = await getPendingRecordings();
      if (!isActive) return;
      console.log(`initializeScreen: Got ${recordings.length} recordings.`);

      const participantRecordings = recordings.filter(r => r.participantCode === fetchedDetails.code);
      const recordedPromptIds = new Set(participantRecordings.map(r => r.promptId));

      let startingIndex = RECORDING_SCRIPT.findIndex(prompt => !recordedPromptIds.has(prompt.id));
      let sessionComplete = false;

      if (startingIndex === -1 && RECORDING_SCRIPT.length > 0) {
        startingIndex = RECORDING_SCRIPT.length - 1;
        sessionComplete = true;
        console.log("initializeScreen: Session is complete for this participant.");
      } else if (startingIndex === -1 && RECORDING_SCRIPT.length === 0) {
        startingIndex = 0;
        sessionComplete = true;
        console.log("initializeScreen: Script is empty.");
      } else {
        console.log(`initializeScreen: First unrecorded index for participant: ${startingIndex}`);
      }

      if (isActive) {
        console.log("initializeScreen: Setting component state...");
        setParticipantDetails(fetchedDetails);
        setCurrentPromptIndex(startingIndex);
        setIsSessionComplete(sessionComplete);

        // Load recording after state is set
        if (startingIndex >= 0) {
          await loadRecordingForPrompt(
            RECORDING_SCRIPT[startingIndex]?.id,
            fetchedDetails.code
          );
        }

        setIsLoading(false);
        console.log("initializeScreen: State update complete. isLoading: false");
      }
    } catch (error: any) {
      console.error("initializeScreen: CATCH BLOCK - Error during initialization!", error);
      if (isActive) {
        setInitializationError(error.message || "Failed to load session data.");
        setIsLoading(false);
        Alert.alert("Initialization Error", error.message || "Could not load session data.");
      }
    }
  }, [router, loadRecordingForPrompt]);


  // --- Effects ---
  // Cleanup on mount/unmount
  useEffect(() => {
    isMounted.current = true;

    // Initialize audio settings once
    Audio.setAudioModeAsync({
      playsInSilentModeIOS: true,
      allowsRecordingIOS: true,
      interruptionModeIOS: InterruptionModeIOS.DoNotMix,
      interruptionModeAndroid: InterruptionModeAndroid.DoNotMix,
    }).catch(console.error);

    return () => {
      isMounted.current = false;
      console.log("RecordScreen: Unmounting, unloading playback sound.");
      // Using direct call to unload rather than the callback to avoid dependency issues
      if (playbackSoundRef.current) {
        playbackSoundRef.current.unloadAsync().catch(console.error);
        playbackSoundRef.current = null;
      }
    };
  }, []);

  // Update navigation title
  useEffect(() => {
    const newTitle = isSessionComplete
      ? 'Session Complete'
      : `Prompt ${(currentPromptIndex ?? 0) + 1}/${RECORDING_SCRIPT.length}`;
    navigation.setOptions({ title: newTitle });
  }, [navigation, currentPromptIndex, isSessionComplete]);

  // Load recording when prompt changes
  useEffect(() => {
    console.log(`RecordScreen: currentPromptIndex changed to: ${currentPromptIndex}`);

    // Always unload previous playback when changing prompts
    if (playbackSoundRef.current) {
      playbackSoundRef.current.unloadAsync().catch(console.error);
      playbackSoundRef.current = null;
    }

    const currentPromptId = currentPromptIndex !== null && currentPromptIndex >= 0 ?
      RECORDING_SCRIPT[currentPromptIndex]?.id : undefined;
    const currentParticipantCode = participantDetails?.code;

    if (currentPromptIndex !== null && currentPromptId && currentParticipantCode) {
      loadRecordingForPrompt(currentPromptId, currentParticipantCode);
    } else {
      setCurrentPromptRecordingUri(null);
      setCurrentPromptRecordingDuration(undefined);
    }
  }, [currentPromptIndex, participantDetails]);

  // Alert for recorder errors
  useEffect(() => {
    if (recorderError) {
      console.error("Recorder Error State Updated:", recorderError);
      Alert.alert('Recorder Error', recorderError);
    }
  }, [recorderError]);

  // Focus effect for initialization
  useFocusEffect(
    useCallback(() => {
      let isActive = true;
      console.log("RecordScreen: useFocusEffect - Running Initialization");

      // Cleanup any existing playback to prevent memory leaks
      if (playbackSoundRef.current) {
        console.log("Cleaning up existing playback in useFocusEffect");
        playbackSoundRef.current.unloadAsync().catch(console.error);
        playbackSoundRef.current = null;
      }

      setIsLoading(true);
      setInitializationError(null);
      setParticipantDetails(null);
      setCurrentPromptIndex(null);
      setIsSessionComplete(false);
      setCurrentPromptRecordingUri(null);
      setCurrentPromptRecordingDuration(undefined);
      setPlaybackStatus('idle');

      initializeScreen(isActive);

      return () => {
        console.log("RecordScreen: useFocusEffect - Cleanup");
        isActive = false;

        // Cleanup playback on blur
        if (playbackSoundRef.current) {
          console.log("Cleaning up playback in useFocusEffect cleanup");
          playbackSoundRef.current.unloadAsync().catch(console.error);
          playbackSoundRef.current = null;
        }
      };
    }, [initializeScreen])
  );

  // --- Render Logic ---
  if (isLoading) {
    return (
      <ThemedSafeAreaView className="flex-1 justify-center items-center">
        <ActivityIndicator size="large" color={primaryColor} />
        <Text className="mt-4 text-neutral-600 dark:text-neutral-300">Loading session...</Text>
      </ThemedSafeAreaView>
    );
  }

  if (initializationError) {
    return (
      <ThemedSafeAreaView className="flex-1 justify-center items-center p-6">
        <MaterialCommunityIcons name="alert-circle-outline" size={60} color={primaryColor} />
        <Text className="mt-4 text-lg text-center text-danger-dark dark:text-danger">Error Loading Session</Text>
        <Text className="mt-2 text-center text-neutral-600 dark:text-neutral-300">{initializationError}</Text>
        <Button
          title="Go Back" icon="arrow-left"
          onPress={() => router.back()}
          className="flex flex-row bg-primary mt-6 px-6 py-3 rounded-lg shadow"
          textClassName="text-white font-semibold"
        />
      </ThemedSafeAreaView>
    );
  }

  // Main Recording Screen
  return (
    <ThemedSafeAreaView className="flex-1 bg-neutral-100 dark:bg-neutral-900">
      {/* Custom Header Elements */}
      <View className="flex-row items-center justify-between px-4 pt-2 pb-1">
        <Button icon="arrow-left" onPress={() => router.back()} title="" className="p-2" iconSize={28} iconColor={iconColor} />
        <Text className="text-center font-semibold text-neutral-600 dark:text-neutral-300">
          {isSessionComplete ? 'Session Complete' : `Prompt ${(currentPromptIndex ?? 0) + 1} / ${RECORDING_SCRIPT.length}`}
        </Text>
        <View style={{ width: 40 }} /> {/* Spacer */}
      </View>

      {/* Progress Bar */}
      <View className="h-1.5 bg-neutral-300 dark:bg-neutral-700 w-full overflow-hidden">
        <View style={{ width: `${progressPercent}%` }} className="h-full bg-primary dark:bg-primary-dark transition-all duration-300 ease-in-out" />
      </View>

      <ScrollView
        className="flex-1"
        contentContainerClassName="flex-grow justify-between p-5"
      >
        {/* Top: Prompt Display Area */}
        <View className="items-center flex-1 justify-center mb-6">
          {isSessionComplete ? (
            <View className="p-6 items-center">
              <MaterialCommunityIcons name="party-popper" size={70} color="#10B981" />
              <Text className="text-2xl font-bold text-neutral-800 mt-4 text-center dark:text-neutral-100">Session Complete!</Text>
              <Text className="text-lg text-neutral-600 mt-2 text-center dark:text-neutral-400">You can now go back to review and upload.</Text>
              <Button
                title="Back to Home" icon="home-outline"
                onPress={() => router.back()}
                className="bg-primary dark:bg-primary-dark mt-8 px-8 py-3 rounded-lg shadow flex flex-row items-center justify-center"
                textClassName="text-white text-lg font-semibold ml-2"
                iconColor="white"
              />
            </View>
          ) : currentPrompt ? (
            <PromptDisplay
              prompt={currentPrompt}
              currentNumber={(currentPromptIndex ?? 0) + 1}
              totalNumber={RECORDING_SCRIPT.length}
              containerClassName="w-full items-center"
              promptTextClassName="text-3xl md:text-4xl text-neutral-900 dark:text-neutral-100 leading-tight my-3 text-center font-medium"
              infoTextClassName="text-center text-sm text-neutral-500 dark:text-neutral-400 mb-1 font-medium"
              typeTextClassName="text-center text-xs font-bold uppercase tracking-wider text-primary dark:text-primary-light mb-3"
              meaningTextClassName="text-center text-base italic text-neutral-600 dark:text-neutral-400 mt-2"
            />
          ) : (
            <ActivityIndicator size="small" color={primaryColor} />
          )}
        </View>

        {/* Bottom: Controls Area */}
        {!isSessionComplete && (
          <View className="items-center pb-4">
            {/* Main Record/Stop Button */}
            <TouchableOpacity
              onPress={handleRecordStopPress}
              disabled={(!canRecord && !isRecording) || isPlaybackActive}
              activeOpacity={0.7}
              className={`
                rounded-full w-24 h-24 items-center justify-center shadow-lg elevation-5 mb-4
                ${isRecording ? 'bg-danger dark:bg-danger-dark' : 'bg-success dark:bg-success-dark'}
                ${(!canRecord && !isRecording) || isPlaybackActive ? 'bg-neutral-400 dark:bg-neutral-600 opacity-60' : ''}
              `}
            >
              <MaterialCommunityIcons
                name={isRecording ? "stop-circle-outline" : "microphone"}
                size={44}
                color="white"
              />
            </TouchableOpacity>

            {/* Sound Wave & Status Text */}
            <View className="h-16 justify-center items-center mb-4 w-full">
              <SoundWaveAnimation
                isAnimating={isRecording || playbackStatus === 'playing'}
                barColor={isRecording ? dangerColor : primaryColor}
              />
              <Text className={`text-center text-base mt-2 h-6 capitalize ${recorderError || playbackError ? 'text-danger-dark dark:text-danger-light' : 'text-neutral-600 dark:text-neutral-400'}`}>
                {statusText}
              </Text>
            </View>

            {/* Secondary Controls: Prev / Play / Next */}
            <View className="flex-row justify-between w-full max-w-sm mt-2 items-center px-4">
              <Button
                title="" icon="arrow-left-circle-outline" iconSize={40}
                onPress={handlePreviousPrompt}
                disabled={!canGoPrevious || isRecording || isPlaybackActive}
                className="p-2 rounded-full"
                iconColor={iconColor}
                disabledClassName="opacity-30"
              />

              {/* Play/Pause Button (conditionally shown) */}
              {showReplay ? (
                <Button
                  title=""
                  icon={playbackStatus === 'playing' ? "pause-circle-outline" : "play-circle-outline"}
                  iconSize={56}
                  onPress={handlePlayPause}
                  disabled={isRecording || playbackStatus === 'loading'}
                  isLoading={playbackStatus === 'loading'}
                  className="p-2 rounded-full"
                  iconColor={primaryColor}
                  disabledClassName="opacity-30"
                />
              ) : (
                <View style={{ width: 60, height: 60 }} />
              )}

              <Button
                title="" icon="arrow-right-circle-outline" iconSize={40}
                onPress={handleNextPrompt}
                disabled={!canGoNext || isRecording || isPlaybackActive}
                className="p-2 rounded-full"
                iconColor={iconColor}
                disabledClassName="opacity-30"
              />
            </View>
          </View>
        )}
      </ScrollView>
    </ThemedSafeAreaView>
  );
}
