import * as FileSystem from 'expo-file-system';
import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { View, Text, Alert, ScrollView, ActivityIndicator, TouchableOpacity, StyleSheet, Dimensions } from 'react-native'; // Added Dimensions
import { useFocusEffect, useRouter, useNavigation } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { Audio, AVPlaybackStatus, InterruptionModeAndroid, InterruptionModeIOS } from 'expo-av';
import ConfettiCannon from 'react-native-confetti-cannon';

import { Button } from '@/components/Button';
import { PromptDisplay } from '@/components/PromptDisplay';
import { SectionIntro } from '@/components/SectionIntro';
import { SoundWaveAnimation } from '@/components/SoundWaveAnimation';
import { useAudioRecorder } from '@/hooks/useAudioRecorder';
import { getParticipantDetails, savePendingRecording, getPendingRecordings } from '@/lib/storage';
import { RECORDING_SECTIONS, getTotalPrompts, getGlobalPromptIndex } from '@/constants/script';
import { RecordingMetadata, ParticipantDetails, ScriptPrompt } from '@/types';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { ThemedSafeAreaView } from '@/components/ThemedSafeAreaView';
import { useThemeColor } from '@/hooks/useThemeColor';
import { isValidCode } from '@/lib/utils';
import { SectionCompleteDialog } from '@/components/SectionCompleteDialog'; // Ensure this path is correct
import { useModal } from '@/context/ModalContext';

type PlaybackStatus = 'idle' | 'loading' | 'playing' | 'paused' | 'stopped' | 'error';
type DisplayMode = 'intro' | 'prompt';

interface PlaybackState {
  sound: Audio.Sound | null;
  status: PlaybackStatus;
  currentUri: string | null;
  error: string | null;
}

// --- Constants ---
const CONFETTI_DURATION = 3500; // ms
const { width: screenWidth } = Dimensions.get('window'); // Get screen width for confetti

export default function RecordScreen() {
  console.log("RecordScreen: Render Start", new Date().toISOString());

  // --- Hooks ---
  const navigation = useNavigation();
  const router = useRouter();

  const { showModal, hideModal } = useModal();

  const primaryColor = useThemeColor({}, 'tint');
  const iconColor = useThemeColor({}, 'icon');
  const dangerColor = useThemeColor({}, 'danger');
  const themedTextColor = useThemeColor({}, 'text');
  const successColor = useThemeColor({}, 'success'); // For confetti
  const timeoutRef = useRef<NodeJS.Timeout>();

  // --- State ---
  const [participantDetails, setParticipantDetails] = useState<ParticipantDetails | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [initializationError, setInitializationError] = useState<string | null>(null);
  const [currentSectionIndex, setCurrentSectionIndex] = useState<number>(0);
  const [currentPromptInSectionIndex, setCurrentPromptInSectionIndex] = useState<number>(0);
  const [displayMode, setDisplayMode] = useState<DisplayMode>('intro');
  const [isSessionComplete, setIsSessionComplete] = useState(false);
  const [currentPromptRecordingUri, setCurrentPromptRecordingUri] = useState<string | null>(null);
  const [currentPromptRecordingDuration, setCurrentPromptRecordingDuration] = useState<number | undefined>(undefined);
  const [showConfetti, setShowConfetti] = useState(false);
  const [playbackUiUpdate, setPlaybackUiUpdate] = useState(0);
  // State for custom modal visibility
  const [isSectionCompleteModalVisible, setIsSectionCompleteModalVisible] = useState(false);

  // --- Refs ---
  const isMounted = useRef(true);
  const confettiRef = useRef<ConfettiCannon>(null);
  const playbackStateRef = useRef<PlaybackState>({
    sound: null,
    status: 'idle',
    currentUri: null,
    error: null,
  }).current;

  // --- Custom Hooks ---
  const {
    startRecording, stopRecordingAndSave, recordingStatus,
    error: recorderError
  } = useAudioRecorder();

  // --- Derived State ---
  const currentSection = RECORDING_SECTIONS[currentSectionIndex];
  const currentPrompt: ScriptPrompt | null =
    displayMode === 'prompt' && currentSection && currentPromptInSectionIndex < currentSection.prompts.length
      ? currentSection.prompts[currentPromptInSectionIndex]
      : null;

  const totalPrompts = useMemo(getTotalPrompts, []);
  const globalPromptIndex = getGlobalPromptIndex(currentSectionIndex, currentPromptInSectionIndex);
  const progressPercent = isSessionComplete ? 100 : Math.round(((globalPromptIndex) / totalPrompts) * 100);

  const hasRecordedCurrentPrompt = !!currentPromptRecordingUri;
  const isRecording = recordingStatus === 'recording';
  const isPlaybackActive = playbackStateRef.status === 'playing' || playbackStateRef.status === 'loading';

  // --- Conditions updated to check modal visibility ---
  const canRecord = (recordingStatus === 'ready' || recordingStatus === 'stopped')
    ? displayMode === 'prompt' && !isSessionComplete && !isPlaybackActive
    : false;
  const showReplay = hasRecordedCurrentPrompt && !isRecording && displayMode === 'prompt';
  const canGoNext = displayMode === 'prompt' && hasRecordedCurrentPrompt && !isRecording && !isPlaybackActive && !isSessionComplete;
  const canGoPrevious = displayMode === 'prompt' && !isRecording && !isPlaybackActive && (currentSectionIndex > 0 || currentPromptInSectionIndex > 0);

  const getLastRecordedPosition = async (participantCode: string) => {
    try {
      const recordings = await getPendingRecordings();
      if (!recordings.length) return { sectionIndex: 0, promptIndex: 0 };

      // Filter recordings for this participant and sort by timestamp
      const participantRecordings = recordings
        .filter(rec => rec.participantCode === participantCode)
        .sort((a, b) => b.timestamp - a.timestamp);

      if (!participantRecordings.length) return { sectionIndex: 0, promptIndex: 0 };

      // Find the section and prompt index for the last recording
      const lastRecording = participantRecordings[0];
      let foundSection = 0;
      let foundPrompt = 0;

      for (let sectionIndex = 0; sectionIndex < RECORDING_SECTIONS.length; sectionIndex++) {
        const promptIndex = RECORDING_SECTIONS[sectionIndex].prompts
          .findIndex(prompt => prompt.id === lastRecording.promptId);

        if (promptIndex !== -1) {
          foundSection = sectionIndex;
          foundPrompt = promptIndex + 1; // Move to next prompt
          if (foundPrompt >= RECORDING_SECTIONS[sectionIndex].prompts.length) {
            foundSection++;
            foundPrompt = 0;
          }
          break;
        }
      }

      return { sectionIndex: foundSection, promptIndex: foundPrompt };
    } catch (error) {
      console.error("Error getting last recorded position:", error);
      return { sectionIndex: 0, promptIndex: 0 };
    }
  };

  // --- Status Text Calculation (remains the same) ---
  let statusText = 'Initializing...';
  if (!isLoading) {
    if (isRecording) statusText = 'Recording...';
    else if (recorderError) statusText = `Recorder Error: ${recorderError}`;
    else if (playbackStateRef.status === 'loading') statusText = 'Loading Playback...';
    else if (playbackStateRef.status === 'playing') statusText = 'Playing...';
    else if (playbackStateRef.status === 'paused') statusText = 'Paused';
    else if (playbackStateRef.status === 'error') statusText = `Playback Error: ${playbackStateRef.error || 'Unknown'}`;
    else if (recordingStatus === 'stopped') statusText = 'Saved!';
    else if (recordingStatus === 'ready' && hasRecordedCurrentPrompt) statusText = 'Ready to Re-record';
    else if (recordingStatus === 'ready') statusText = 'Ready to Record';
    else if (recordingStatus === 'requesting') statusText = 'Requesting Permissions...';
    else if (recordingStatus === 'error') statusText = 'Microphone Permission Denied';
    else statusText = 'Ready';
  }

  // --- Callbacks (remain the same, except promptForNextAction) ---

  const triggerHaptic = useCallback((type: Haptics.ImpactFeedbackStyle = Haptics.ImpactFeedbackStyle.Light) => {
    Haptics.impactAsync(type).catch(console.error);
  }, []);

  const triggerPlaybackUiUpdate = useCallback(() => {
    setPlaybackUiUpdate(v => v + 1);
  }, []);

  const verifyFileExists = useCallback(async (uri: string): Promise<boolean> => {
    if (!uri) return false;
    try { const fileInfo = await FileSystem.getInfoAsync(uri); return fileInfo.exists; }
    catch (error) { console.error("Error verifying file existence:", error); return false; }
  }, []);

  const unloadPlaybackSound = useCallback(async () => {
    console.log("Attempting to unload playback sound..");
    const sound = playbackStateRef.sound;
    if (sound) {
      try {
        await sound.unloadAsync();
        console.log("Playback sound unloaded successfully");
      } catch (e) {
        console.error("Error during sound unloading:", e);
      } finally {
        playbackStateRef.sound = null;
        playbackStateRef.status = 'idle';
        playbackStateRef.currentUri = null;
        playbackStateRef.error = null;
        triggerPlaybackUiUpdate();
      }
    } else {
      console.log("No playback sound to unload");
    }
  }, [playbackStateRef, triggerPlaybackUiUpdate]);

  const onPlaybackStatusUpdate = useCallback((status: AVPlaybackStatus) => {
    if (!playbackStateRef.sound) return;

    if (!status.isLoaded) {
      if (status.error) {
        console.error(`Playback Error: ${status.error}`);
        playbackStateRef.error = `Error: ${status.error}`;
        playbackStateRef.status = 'error';
        playbackStateRef.sound = null;
      } else {
        if (playbackStateRef.status !== 'idle' && playbackStateRef.status !== 'stopped') {
          playbackStateRef.status = 'stopped';
        }
      }
    } else {
      if (status.didJustFinish) {
        console.log("Playback finished naturally.");
        playbackStateRef.status = 'stopped';
        playbackStateRef.sound?.setPositionAsync(0).catch(console.error);
      } else if (status.isPlaying) {
        if (playbackStateRef.status !== 'playing') {
          playbackStateRef.status = 'playing';
          playbackStateRef.error = null;
        }
      } else {
        if (playbackStateRef.status === 'playing') {
          playbackStateRef.status = 'paused';
        }
      }
    }
    triggerPlaybackUiUpdate();
  }, [playbackStateRef, triggerPlaybackUiUpdate]);

  const handlePlayPause = useCallback(async () => {
    triggerHaptic();
    playbackStateRef.error = null;

    const currentSound = playbackStateRef.sound;
    const currentStatus = playbackStateRef.status;
    const targetUri = currentPromptRecordingUri;

    console.log("handlePlayPause called:", { URI: targetUri, currentStatus, hasSound: !!currentSound });

    if (currentStatus === 'playing' && currentSound) {
      console.log("Pausing playback");
      try { await currentSound.pauseAsync(); playbackStateRef.status = 'paused'; triggerPlaybackUiUpdate(); }
      catch (error: any) { console.error("Error pausing playback:", error); playbackStateRef.error = `Failed to pause: ${error.message}`; playbackStateRef.status = 'error'; triggerPlaybackUiUpdate(); }
    }
    else if (currentStatus === 'paused' && currentSound) {
      console.log("Resuming playback");
      try { await currentSound.playAsync(); playbackStateRef.status = 'playing'; triggerPlaybackUiUpdate(); }
      catch (error: any) { console.error("Error resuming playback:", error); playbackStateRef.error = `Failed to resume: ${error.message}`; playbackStateRef.status = 'error'; triggerPlaybackUiUpdate(); }
    }
    else if (targetUri && ['idle', 'stopped', 'error'].includes(currentStatus)) {
      console.log("Starting new playback for URI:", targetUri);
      await unloadPlaybackSound();
      playbackStateRef.status = 'loading'; playbackStateRef.currentUri = targetUri; triggerPlaybackUiUpdate();

      try {
        const fileExists = await verifyFileExists(targetUri);
        if (!fileExists) throw new Error(`Audio file doesn't exist at: ${targetUri}`);
        const { sound } = await Audio.Sound.createAsync({ uri: targetUri }, { shouldPlay: true, progressUpdateIntervalMillis: 100 }, onPlaybackStatusUpdate);
        playbackStateRef.sound = sound; playbackStateRef.status = 'playing'; playbackStateRef.error = null; triggerPlaybackUiUpdate();
      } catch (error: any) {
        console.error("Error creating/starting sound:", error); playbackStateRef.error = `Playback failed: ${error.message}`; playbackStateRef.status = 'error'; playbackStateRef.sound = null; playbackStateRef.currentUri = null; triggerPlaybackUiUpdate();
      }
    } else {
      console.log("Play/Pause ignored - Status:", currentStatus, "URI:", targetUri);
    }
  }, [currentPromptRecordingUri, playbackStateRef, unloadPlaybackSound, onPlaybackStatusUpdate, triggerHaptic, verifyFileExists, triggerPlaybackUiUpdate]);

  // --- Recording Callbacks (handleRecordPress, handleStopPress, loadRecordingForPrompt - no changes) ---
  const handleRecordPress = useCallback(() => {
    if (!participantDetails?.code) {
      Alert.alert("Error", "Please set up participant details before recording.");
      return;
    }

    if (!currentPrompt) {
      Alert.alert("Error", "No active prompt to record.");
      return;
    }

    if (playbackStateRef.status !== 'idle' && playbackStateRef.status !== 'stopped') {
      unloadPlaybackSound();
    }

    console.log("Starting new recording...");
    setCurrentPromptRecordingUri(null);
    setCurrentPromptRecordingDuration(undefined);
    triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
    startRecording();
  }, [participantDetails, currentPrompt, startRecording, unloadPlaybackSound, playbackStateRef, triggerHaptic]);

  const handleStopPress = useCallback(async () => {
    if (!participantDetails?.code) {
      console.error("Stop prevented: Missing participant details");
      Alert.alert("Error", "Participant details not found. Please set up participant details.");
      return;
    }

    if (!currentPrompt) {
      console.error("Stop prevented: Missing current prompt");
      Alert.alert("Error", "No active prompt found. Please restart the recording session.");
      return;
    }

    triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
    console.log("Attempting to stop recording...");

    try {
      const savedMetadata = await stopRecordingAndSave(
        participantDetails.code,
        currentPrompt.id,
        currentPrompt.text,
        participantDetails
      );

      if (savedMetadata) {
        console.log("Recording stopped and saved successfully:", savedMetadata.localUri);
        const fileExists = await verifyFileExists(savedMetadata.localUri);
        if (!fileExists) {
          throw new Error("Recording file not found after saving");
        }
        const savedToStorage = await savePendingRecording(savedMetadata);
        if (!savedToStorage) throw new Error("Failed to save recording metadata");

        setCurrentPromptRecordingUri(savedMetadata.localUri);
        setCurrentPromptRecordingDuration(savedMetadata.recordingDuration);
      } else {
        throw new Error("Could not save the recording");
      }
    } catch (error: any) {
      console.error("Error in handleStopPress:", error);
      Alert.alert("Recording Error", error.message || "An error occurred while saving the recording");
      setCurrentPromptRecordingUri(null);
      setCurrentPromptRecordingDuration(undefined);
    }
  }, [participantDetails, currentPrompt, stopRecordingAndSave, verifyFileExists, triggerHaptic]);

  const loadRecordingForPrompt = useCallback(async (promptId: string | undefined, code: string | undefined) => {
    if (!promptId || !code) {
      console.log(`[loadRecordingForPrompt] Invalid promptId (${promptId}) or code (${code})`);
      setCurrentPromptRecordingUri(null); setCurrentPromptRecordingDuration(undefined); return;
    }
    console.log(`[loadRecordingForPrompt] Loading for prompt ${promptId}, participant ${code}`);
    try {
      const allRecordings = await getPendingRecordings();
      console.log(`[loadRecordingForPrompt] Found ${allRecordings.length} total recordings`);
      const existingRecording = allRecordings.find(rec => rec.promptId === promptId && rec.participantCode === code);
      console.log(`[loadRecordingForPrompt] Found recording for this prompt:`, existingRecording ? "yes" : "no");
      if (existingRecording?.localUri) {
        const fileExists = await verifyFileExists(existingRecording.localUri);
        if (fileExists) {
          console.log(`[loadRecordingForPrompt] File exists: ${existingRecording.localUri}`);
          setCurrentPromptRecordingUri(existingRecording.localUri); setCurrentPromptRecordingDuration(existingRecording.recordingDuration);
        } else {
          console.warn(`[loadRecordingForPrompt] File doesn't exist at path: ${existingRecording.localUri}`);
          setCurrentPromptRecordingUri(existingRecording.localUri); setCurrentPromptRecordingDuration(existingRecording.recordingDuration);
        }
      } else {
        console.log(`[loadRecordingForPrompt] No recording found or URI is empty`);
        setCurrentPromptRecordingUri(null); setCurrentPromptRecordingDuration(undefined);
      }
    } catch (error) {
      console.error("[loadRecordingForPrompt] Error loading specific recording:", error);
      setCurrentPromptRecordingUri(null); setCurrentPromptRecordingDuration(undefined);
    }
  }, [verifyFileExists]);

  const handleRecordStopPress = useCallback(() => {
    if (!participantDetails?.code) {
      Alert.alert("Setup Required", "Please set up participant details before recording.");
      return;
    }

    if (!currentPrompt) {
      Alert.alert("Error", "No active prompt to record.");
      return;
    }

    if (recordingStatus === 'recording') {
      console.log("handleRecordStopPress: Status is recording, calling handleStopPress");
      handleStopPress();
    } else if (canRecord) {
      console.log("handleRecordStopPress: Status is not recording, calling handleRecordPress");
      handleRecordPress();
    } else {
      console.log("handleRecordStopPress: Neither recording nor canRecord is true.");
    }
  }, [recordingStatus, canRecord, handleStopPress, handleRecordPress, participantDetails, currentPrompt]);

  // --- Section/Navigation Callbacks ---
  const handleStartSection = useCallback(() => {
    triggerHaptic();
    setDisplayMode('prompt');
    const firstPromptId = currentSection?.prompts[0]?.id;
    if (firstPromptId && participantDetails?.code) {
      loadRecordingForPrompt(firstPromptId, participantDetails.code);
    } else {
      setCurrentPromptRecordingUri(null);
      setCurrentPromptRecordingDuration(undefined);
    }
  }, [currentSection, participantDetails, loadRecordingForPrompt, triggerHaptic]);

  // --- NEW: Function to show the confirmation prompt ---
  // const promptForNextAction = useCallback(() => {
  //   console.log("promptForNextAction called - Setting modal visible");
  //   setIsConfirmationModalVisible(true);
  //   console.log("isConfirmationModalVisible should be set to true now");
  // }, []);

  const handleNextPrompt = useCallback(() => {
    if (!canGoNext) {
      if (!hasRecordedCurrentPrompt && !isSessionComplete && displayMode === 'prompt') {
        Alert.alert("Recording Required", "Please record the current prompt before proceeding.");
        triggerHaptic(Haptics.ImpactFeedbackStyle.Heavy);
      }
      return;
    }

    triggerHaptic();
    unloadPlaybackSound();
    const currentSectionPrompts = currentSection?.prompts ?? [];
    const isLastPromptInSection = currentPromptInSectionIndex >= currentSectionPrompts.length - 1;

    if (isLastPromptInSection) {
      if (currentSectionIndex >= RECORDING_SECTIONS.length - 1) {
        setIsSessionComplete(true);
        Alert.alert("Session Complete!", "All prompts recorded. Go back to review and upload.");
      } else {
        triggerHaptic(Haptics.ImpactFeedbackStyle.Heavy);
        setShowConfetti(true);

        // Show modal after confetti
        setTimeout(() => {
          setShowConfetti(false);
          setIsSectionCompleteModalVisible(true);
        }, CONFETTI_DURATION);
      }
    } else {
      // Regular next prompt logic
      const nextPromptIndex = currentPromptInSectionIndex + 1;
      setCurrentPromptInSectionIndex(nextPromptIndex);
      setCurrentPromptRecordingUri(null);
      setCurrentPromptRecordingDuration(undefined);
      const nextPromptId = currentSection?.prompts[nextPromptIndex]?.id;
      if (nextPromptId && participantDetails?.code) {
        loadRecordingForPrompt(nextPromptId, participantDetails.code);
      } else {
        setCurrentPromptRecordingUri(null);
        setCurrentPromptRecordingDuration(undefined);
      }
    }
  }, [
    canGoNext, currentSection, currentSectionIndex, currentPromptInSectionIndex,
    displayMode, hasRecordedCurrentPrompt, isSessionComplete,
    triggerHaptic, unloadPlaybackSound, showModal, hideModal, router,
    participantDetails, loadRecordingForPrompt
  ]);

  // useEffect(() => {
  //   isMounted.current = true;

  //   return () => {
  //     isMounted.current = false;
  //     if (timeoutRef.current) {
  //       clearTimeout(timeoutRef.current);
  //     }
  //   };
  // }, []);

  // const handleModalError = useCallback((error: Error) => {
  //   console.error("Modal Error:", error);
  //   setIsConfirmationModalVisible(false);
  // }, []);

  const handlePreviousPrompt = useCallback(() => {
    if (!canGoPrevious) return;
    triggerHaptic();
    unloadPlaybackSound();

    if (currentPromptInSectionIndex > 0) {
      const prevPromptIndex = currentPromptInSectionIndex - 1;
      setCurrentPromptInSectionIndex(prevPromptIndex);
      setDisplayMode('prompt');
      const prevPromptId = currentSection?.prompts[prevPromptIndex]?.id;
      if (prevPromptId && participantDetails?.code) {
        loadRecordingForPrompt(prevPromptId, participantDetails.code);
      } else {
        setCurrentPromptRecordingUri(null); setCurrentPromptRecordingDuration(undefined);
      }
    } else if (currentSectionIndex > 0) {
      const prevSectionIndex = currentSectionIndex - 1;
      const prevSection = RECORDING_SECTIONS[prevSectionIndex];
      const lastPromptIndexInPrevSection = prevSection.prompts.length - 1;
      setCurrentSectionIndex(prevSectionIndex);
      setCurrentPromptInSectionIndex(lastPromptIndexInPrevSection);
      setDisplayMode('prompt');
      const prevPromptId = prevSection?.prompts[lastPromptIndexInPrevSection]?.id;
      if (prevPromptId && participantDetails?.code) {
        loadRecordingForPrompt(prevPromptId, participantDetails.code);
      } else {
        setCurrentPromptRecordingUri(null); setCurrentPromptRecordingDuration(undefined);
      }
    }
  }, [
    canGoPrevious, currentSectionIndex, currentPromptInSectionIndex, currentSection,
    participantDetails, triggerHaptic, unloadPlaybackSound, loadRecordingForPrompt
  ]);

  // --- Initialization (remains the same) ---
  const initializeScreen = useCallback(async (isActive: boolean) => {
    console.log("RecordScreen: Initializing screen...");
    setIsLoading(true);
    setInitializationError(null);
    try {
      // Load participant details first
      const details = await getParticipantDetails();

      if (!isActive) return;

      if (!details || !isValidCode(details.code)) {
        console.log("RecordScreen: No valid participant details found");
        Alert.alert(
          "Setup Required",
          "Please set up participant details before recording.",
          [{ text: "OK", onPress: () => router.replace("/") }]
        );
        setInitializationError("Participant setup required."); // Set error state
        setIsLoading(false);
        return;
      }

      console.log("RecordScreen: Participant details loaded:", details);
      setParticipantDetails(details);

      // Get the last recorded position
      const { sectionIndex, promptIndex } = await getLastRecordedPosition(details.code);
      setCurrentSectionIndex(sectionIndex);
      setCurrentPromptInSectionIndex(promptIndex);

      // Set the correct display mode
      setDisplayMode(promptIndex === 0 && sectionIndex < RECORDING_SECTIONS.length ? 'intro' : 'prompt');

      if (displayMode === 'prompt') {
        const startPrompt = RECORDING_SECTIONS[sectionIndex]?.prompts[promptIndex];
        if (startPrompt?.id) {
          loadRecordingForPrompt(startPrompt.id, details.code);
        }
      }

      setIsSessionComplete(false);
      setIsLoading(false);

    } catch (error) {
      console.error("RecordScreen: Initialization error:", error);
      if (isActive) {
        setInitializationError("Failed to load participant details");
        setIsLoading(false);
      }
    }
  }, [router, loadRecordingForPrompt]);

  // --- Effects (remain the same) ---
  useEffect(() => {
    // Mounted ref
    isMounted.current = true;
    return () => {
      isMounted.current = false;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      unloadPlaybackSound(); // Ensure sound is unloaded on unmount
    };
  }, [unloadPlaybackSound]);

  useEffect(() => {
    let newTitle = "Recording";
    if (isSessionComplete) newTitle = 'Session Complete';
    else if (displayMode === 'intro' && currentSection) newTitle = `Section: ${currentSection.title}`;
    else if (currentPrompt) newTitle = `Prompt ${globalPromptIndex + 1}/${totalPrompts}`;
    navigation.setOptions({ title: newTitle });
  }, [navigation, displayMode, currentSection, currentPrompt, globalPromptIndex, totalPrompts, isSessionComplete]);

  useEffect(() => {
    console.log(`RecordScreen: Prompt or Mode changed. Section: ${currentSectionIndex}, P: ${currentPromptInSectionIndex}, M: ${displayMode}`);
    unloadPlaybackSound();
    if (displayMode === 'prompt' && currentPrompt?.id && participantDetails?.code) {
      loadRecordingForPrompt(currentPrompt.id, participantDetails.code);
    } else {
      setCurrentPromptRecordingUri(null); setCurrentPromptRecordingDuration(undefined);
    }
  }, [currentPromptInSectionIndex, currentSectionIndex, displayMode, participantDetails, loadRecordingForPrompt, currentPrompt, unloadPlaybackSound]);

  useEffect(() => {
    if (recorderError) {
      Alert.alert('Recorder Error', recorderError);
    }
  }, [recorderError]);

  useFocusEffect(
    useCallback(() => {
      let isActive = true;
      console.log("RecordScreen: useFocusEffect - Running Initialization");
      initializeScreen(isActive);

      // Clear any existing timeout
      // if (timeoutRef.current) {
      //   clearTimeout(timeoutRef.current);
      // }

      // Reset only necessary states
      // const resetStates = () => {
      //   if (isActive) {
      //     // setIsConfirmationModalVisible(false);
      //     setShowConfetti(false);
      //     unloadPlaybackSound();
      //     setIsLoading(true);
      //     setInitializationError(null);
      //     setParticipantDetails(null);
      //     setCurrentPromptRecordingUri(null);
      //     setCurrentPromptRecordingDuration(undefined);
      //     playbackStateRef.status = 'idle';
      //     playbackStateRef.error = null;
      //     playbackStateRef.currentUri = null;
      //     playbackStateRef.sound = null;
      //     triggerPlaybackUiUpdate();
      //   }
      // };

      // resetStates();
      // initializeScreen(isActive);

      return () => {
        // console.log("RecordScreen: useFocusEffect - Cleanup");
        isActive = false;
        // if (timeoutRef.current) {
        //   clearTimeout(timeoutRef.current);
        // }
        // Don't reset section and prompt indices on cleanup
        // resetStates();
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
        <MaterialCommunityIcons name="alert-circle-outline" size={60} color={dangerColor} />
        <Text className="mt-4 text-lg text-center text-danger-dark dark:text-danger">Error Loading Session</Text>
        <Text className="mt-2 text-center text-neutral-600 dark:text-neutral-300">{initializationError}</Text>
        <Button title="Go Back" icon="arrow-left" onPress={() => router.back()} className="flex flex-row bg-primary mt-6 px-6 py-3 rounded-lg shadow" textClassName="text-white font-semibold" iconColor='white' />
      </ThemedSafeAreaView>
    );
  }

  // console.log("Rendering - Modal Visible:", isConfirmationModalVisible, "Display Mode:", displayMode); // Add log

  return (
    <ThemedSafeAreaView className="flex-1 bg-neutral-100 dark:bg-neutral-900">
      {/* Header */}
      <View className="flex-row items-center justify-between px-4 pt-2 pb-1">
        <Button icon="arrow-left" onPress={() => router.back()} title="" className="p-2" iconSize={28} iconColor={iconColor} />
        <Text className="text-center font-semibold text-neutral-600 dark:text-neutral-300">
          {/* Dynamic title setting seems to be missing, using navigation options instead is good */}
          {navigation?.getState().routes.find(r => r.name === 'record')?.params?.title || 'Recording'}
        </Text>
        <View style={{ width: 40 }} />
      </View>

      {/* Progress Bar */}
      <View className="h-1.5 bg-neutral-300 dark:bg-neutral-700 w-full overflow-hidden">
        <View style={{ width: `${progressPercent}%` }} className="h-full bg-primary dark:bg-primary-dark transition-all duration-300 ease-in-out" />
      </View>

      {/* --- Main Content Area --- */}
      <View className="flex-1 justify-center items-center">
        {displayMode === 'intro' && !isSessionComplete && currentSection ? (
          <SectionIntro
            sectionTitle={currentSection.title}
            sectionDescription={currentSection.description}
            sectionNumber={currentSectionIndex + 1}
            totalSections={RECORDING_SECTIONS.length}
            onStartSection={handleStartSection}
          />
        ) : isSessionComplete ? (
          <View className="p-6 items-center">
            <MaterialCommunityIcons name="party-popper" size={70} color="#10B981" />
            <Text className="text-2xl font-bold text-neutral-800 mt-4 text-center dark:text-neutral-100">Recording Session Complete!</Text>
            <Text className="text-lg text-neutral-600 mt-2 text-center dark:text-neutral-400">All prompts recorded. Go back home to review and upload.</Text>
            <Button title="Back to Home" icon="home-outline" onPress={() => router.back()} className="bg-primary dark:bg-primary-dark mt-8 px-8 py-3 rounded-lg shadow flex flex-row items-center justify-center" textClassName="text-white text-lg font-semibold ml-2" iconColor="white" />
          </View>
        ) : currentPrompt ? (
          <ScrollView className="w-full" contentContainerStyle={styles.scrollViewContent} showsVerticalScrollIndicator={false}>
            <PromptDisplay
              prompt={currentPrompt}
              currentNumber={globalPromptIndex + 1}
              totalNumber={totalPrompts}
              containerClassName="w-full items-center px-4 py-6"
              promptTextClassName="text-3xl md:text-4xl text-neutral-900 dark:text-neutral-100 leading-tight my-3 text-center font-medium"
              infoTextClassName="text-center text-sm text-neutral-500 dark:text-neutral-400 mb-1 font-medium"
              typeTextClassName="text-center text-xs font-bold uppercase tracking-wider text-primary dark:text-primary-light mb-3"
              meaningTextClassName="text-center text-base italic text-neutral-600 dark:text-neutral-400 mt-2" />
          </ScrollView>
        ) : (
          <ActivityIndicator size="small" color={primaryColor} />
        )}
      </View>

      {/* --- Bottom Controls Area --- */}
      {/* Ensure controls are disabled when modal is visible */}
      {!isSessionComplete && displayMode === 'prompt' && (
        <View className={`w-full items-center pt-3 pb-6 px-4 border-t border-neutral-200 dark:border-neutral-700 bg-neutral-100 dark:bg-neutral-900 `}
        // pointerEvents={isConfirmationModalVisible ? 'none' : 'auto'}
        >
          {/* Record/Stop Button */}
          <TouchableOpacity onPress={handleRecordStopPress} disabled={!canRecord && !isRecording} activeOpacity={0.7} className={`rounded-full w-20 h-20 items-center justify-center shadow-lg elevation-5 mb-3 ${isRecording ? 'bg-danger dark:bg-danger-dark' : 'bg-success dark:bg-success-dark'} ${(!canRecord && !isRecording) ? 'bg-neutral-400 dark:bg-neutral-600 opacity-60' : ''}`}>
            <MaterialCommunityIcons name={isRecording ? "stop-circle-outline" : "microphone"} size={40} color="white" />
          </TouchableOpacity>

          {/* Sound Wave & Status Text */}
          <View className="h-12 justify-center items-center mb-3 w-full">
            <SoundWaveAnimation isAnimating={isRecording || playbackStateRef.status === 'playing'} barColor={isRecording ? dangerColor : primaryColor} />
            <Text className={`text-center text-sm mt-1 h-5 capitalize ${recorderError || playbackStateRef.error ? 'text-danger-dark dark:text-danger-light' : 'text-neutral-600 dark:text-neutral-400'}`}>
              {statusText}
            </Text>
          </View>

          {/* Prev / Play / Next Controls */}
          <View className="flex-row justify-between w-full max-w-sm items-center px-2">
            <Button title="" icon="arrow-left-circle-outline" iconSize={40} onPress={handlePreviousPrompt} disabled={!canGoPrevious} className="p-2 rounded-full" iconColor={iconColor} disabledClassName="opacity-30" />
            {showReplay ? (
              <Button title="" icon={playbackStateRef.status === 'playing' ? "pause-circle-outline" : "play-circle-outline"} iconSize={56} onPress={handlePlayPause} disabled={isRecording || playbackStateRef.status === 'loading'} isLoading={playbackStateRef.status === 'loading'} className="p-2 rounded-full" iconColor={primaryColor} disabledClassName="opacity-30" />
            ) : (
              <View style={{ width: 60, height: 60 }} />
            )}
            <Button title="" icon="arrow-right-circle-outline" iconSize={40} onPress={handleNextPrompt} disabled={!canGoNext} className="p-2 rounded-full" iconColor={iconColor} disabledClassName="opacity-30" />
          </View>
        </View>
      )}

      {/* --- Confetti Layer (Wrapper View) --- */}
      {showConfetti && (
        <View style={styles.confettiLayer} pointerEvents="none">
          <ConfettiCannon ref={confettiRef} count={250} origin={{ x: screenWidth / 2, y: -20 }} fadeOut={true} explosionSpeed={500} fallSpeed={3500} colors={[primaryColor || '#4F46E5', successColor || '#10B981', '#F59E0B', '#EF4444']} autoStart={true} />
        </View>
      )}

      {/* --- Confirmation Modal --- */}
      {isSectionCompleteModalVisible && (
        <SectionCompleteDialog
          visible={isSectionCompleteModalVisible}
          onClose={() => setIsSectionCompleteModalVisible(false)}
          sectionNumber={currentSectionIndex + 1}
          sectionTitle={currentSection?.title || ''}
          onContinue={() => {
            setIsSectionCompleteModalVisible(false);
            const nextSectionIndex = currentSectionIndex + 1;
            if (nextSectionIndex < RECORDING_SECTIONS.length) {
              setCurrentSectionIndex(nextSectionIndex);
              setCurrentPromptInSectionIndex(0);
              setDisplayMode('intro');
              setCurrentPromptRecordingUri(null);
              setCurrentPromptRecordingDuration(undefined);
            } else {
              // This case should ideally be handled by the isSessionComplete logic earlier
              console.warn("Continue pressed on the last section, should have been caught earlier.");
              setIsSessionComplete(true); // Ensure session complete state is set
            }
          }}
          onGoBack={() => {
            setIsSectionCompleteModalVisible(false);
            router.back();
          }}
        />)
      }
    </ThemedSafeAreaView>
  );
}

// --- Styles ---
const styles = StyleSheet.create({
  scrollViewContent: { flexGrow: 1, justifyContent: 'center', paddingVertical: 20 },
  confettiLayer: { ...StyleSheet.absoluteFillObject, zIndex: 20, pointerEvents: 'none' }, // Added pointerEvents
});
