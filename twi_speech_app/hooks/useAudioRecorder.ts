import { requestMicrophonePermissions } from '@/lib/permissions';
import { saveRecordingFile } from '@/lib/storage';
import { ParticipantDetails, RecordingMetadata } from '@/types';
import { Audio, InterruptionModeAndroid, InterruptionModeIOS } from 'expo-av';
import { useCallback, useEffect, useRef, useState } from 'react';
import 'react-native-get-random-values';
import { v4 as uuidv4 } from 'uuid';

type RecordingStatus = 'idle' | 'requesting' | 'ready' | 'recording' | 'stopped' | 'error';

// Define explicit options (consider making these configurable if needed)
const commonAudioSettings = {
  sampleRate: 44100,
  numberOfChannels: 1,
  bitRate: 128000, // Reasonable quality for voice
};

// The problem is here - we need to properly format these options
const recordingOptions = {
  android: {
    extension: '.m4a',
    outputFormat: Audio.AndroidOutputFormat.MPEG_4,
    audioEncoder: Audio.AndroidAudioEncoder.AAC,
    ...commonAudioSettings,
  },
  ios: {
    extension: '.m4a',
    outputFormat: Audio.IOSOutputFormat.MPEG4AAC,
    ...commonAudioSettings,
  },
  web: {
    mimeType: 'audio/webm',
    bitsPerSecond: 128000,
  }
};

const fileExtension = '.m4a';
const defaultContentType = 'audio/mp4'; // MIME type for m4a

export function useAudioRecorder() {
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [status, setStatus] = useState<RecordingStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [lastRecordingUri, setLastRecordingUri] = useState<string | null>(null);
  const [lastRecordingDuration, setLastRecordingDuration] = useState<number | undefined>(undefined);

  const isMounted = useRef(true);

  // Request permissions on mount
  useEffect(() => {
    isMounted.current = true;
    console.log("useAudioRecorder: Mounting and requesting permissions...");

    let permissionGranted = false;

    const initializeRecorder = async () => {
      try {
        if (!isMounted.current) return;
        setStatus('requesting');
        permissionGranted = await requestMicrophonePermissions();
        if (!isMounted.current) return;

        if (permissionGranted) {
          setStatus('ready');
          console.log("useAudioRecorder: Permissions granted, status set to ready.");
        } else {
          setStatus('error');
          setError('Microphone permissions denied.');
          console.error("useAudioRecorder: Permissions denied.");
        }
      } catch (err: any) {
        console.error("useAudioRecorder: Error requesting permissions:", err);
        if (isMounted.current) {
          setStatus('error');
          setError(`Permission check failed: ${err.message}`);
        }
      }
    };

    initializeRecorder();

    return () => {
      console.log('useAudioRecorder: Unmounting...');
      isMounted.current = false;
      if (recording) {
        console.log('useAudioRecorder: Unmounting - stopping and unloading active recording.');
        recording.stopAndUnloadAsync().catch(e => console.error("Error stopping on unmount:", e));
        setRecording(null);
      }
    };
  }, []);

  // --- Start Recording ---
  const startRecording = useCallback(async () => {
    // Check if ready to record
    if (status !== 'ready' && status !== 'stopped') {
      console.warn(`useAudioRecorder: Cannot start recording in status: ${status}`);
      setError(`Cannot start recording right now (Status: ${status}). Please wait or check permissions.`);
      return;
    }

    // Clean up any existing recording object *before* starting new
    if (recording) {
      console.warn('useAudioRecorder: Cleaning up previous recording object...');
      try {
        await recording.stopAndUnloadAsync();
      } catch (e) {
        console.error("Error unloading previous recording:", e);
      } finally {
        if (isMounted.current) setRecording(null);
      }
    }

    // Reset state for the new recording attempt
    if (isMounted.current) {
      setError(null);
      setLastRecordingUri(null);
      setLastRecordingDuration(undefined);
    }

    try {
      console.log('useAudioRecorder: Setting Audio Mode for recording...');
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        interruptionModeIOS: InterruptionModeIOS.DoNotMix,
        interruptionModeAndroid: InterruptionModeAndroid.DoNotMix,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
        staysActiveInBackground: false,
      });
      console.log('useAudioRecorder: Audio Mode set.');

      console.log('useAudioRecorder: Creating new recording instance...');
      const newRecording = new Audio.Recording();

      // Use the correct format for recordingOptions
      console.log('useAudioRecorder: Preparing to record with options:', recordingOptions);
      await newRecording.prepareToRecordAsync(recordingOptions);

      console.log('useAudioRecorder: Starting recording...');
      await newRecording.startAsync();

      if (!isMounted.current) {
        console.log("useAudioRecorder: Unmounted during recording start, unloading.");
        await newRecording.stopAndUnloadAsync();
        return;
      }

      // Update state after successful start
      setRecording(newRecording);
      setStatus('recording');
      console.log('useAudioRecorder: Recording started successfully.');

    } catch (err: any) {
      console.error('useAudioRecorder: Failed during recording setup/start:', err);
      let message = err.message || 'An unknown error occurred during recording setup/start.';
      if (err.code) message += ` (Code: ${err.code})`;
      if (isMounted.current) {
        setError(`Failed to start recording: ${message}`);
        setStatus('error');
        setRecording(null);
      }
    }
  }, [status, recording]);

  // --- Stop Recording and Save ---
  const stopRecordingAndSave = useCallback(async (
    participantCode: string,
    promptId: string,
    promptText: string,
    participantDetails: ParticipantDetails | null
  ): Promise<RecordingMetadata | null> => {
    if (status !== 'recording' || !recording) {
      console.warn('useAudioRecorder: No active recording to stop.');
      return null;
    }
    console.log('useAudioRecorder: Stopping recording...');
    if (isMounted.current) setError(null);

    try {
      // Get status *before* unloading to capture duration
      const statusBeforeUnload = await recording.getStatusAsync();
      const durationMillis = statusBeforeUnload.isRecording ? statusBeforeUnload.durationMillis : undefined;
      console.log(`useAudioRecorder: Duration before unload: ${durationMillis}ms`);

      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      console.log('useAudioRecorder: Recording stopped and unloaded.');

      // Clear the recording object from state immediately after stopping/unloading
      if (isMounted.current) setRecording(null);

      if (!uri) {
        throw new Error("Recording URI is null after stopping.");
      }
      console.log('useAudioRecorder: Recording stopped, URI:', uri);

      const timestamp = Date.now();
      const uniqueId = uuidv4().substring(0, 8);
      const filename = `${participantCode}_${promptId}_${timestamp}_${uniqueId}${fileExtension}`;
      const contentType = defaultContentType;

      // Save the file from its temporary location to permanent app storage
      const permanentUri = await saveRecordingFile(uri, filename);

      if (permanentUri) {
        console.log('useAudioRecorder: Recording saved locally:', permanentUri);

        // Create metadata object, including participant details snapshot
        const metadata: RecordingMetadata = {
          id: uuidv4(),
          participantCode: participantCode,
          promptId: promptId,
          promptText: promptText,
          timestamp: timestamp,
          localUri: permanentUri,
          originalFilename: filename,
          contentType: contentType,
          uploaded: false,
          recordingDuration: durationMillis,
          dialect: participantDetails?.dialect,
          age_range: participantDetails?.age_range,
          gender: participantDetails?.gender,
        };
        console.log("[useAudioRecorder] Generated Recording Metadata:", JSON.stringify(metadata));

        // Update state with the details of the *last successfully saved* recording
        if (isMounted.current) {
          setLastRecordingUri(permanentUri);
          setLastRecordingDuration(durationMillis);
          setStatus('stopped');
        }
        return metadata;

      } else {
        throw new Error("Failed to save the recording file permanently.");
      }

    } catch (err: any) {
      console.error('useAudioRecorder: Failed to stop or save recording:', err);
      if (isMounted.current) {
        setError(`Failed to stop/save recording: ${err.message}`);
        setStatus('error');
        setRecording(null);
      }
      return null;
    }
  }, [status, recording]);

  return {
    startRecording,
    stopRecordingAndSave,
    recordingStatus: status,
    error,
    lastRecordingUri,
    lastRecordingDuration,
  };
}
