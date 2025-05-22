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
  savePendingRecording,
} from '@/lib/storage';
import { checkParticipantExists, uploadRecording } from '@/lib/api';
import * as Network from 'expo-network';
import { RecordingMetadata, ParticipantDetails, UploadResponse } from '@/types';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import SetupScreenContent from '@/components/SetupScreenContent';
import { ThemedSafeAreaView } from '@/components/ThemedSafeAreaView';
import { isValidCode, verifyFileExists } from '@/lib/utils';
import { useThemeColor } from '@/hooks/useThemeColor';
import { ParticipantSelector } from '@/components/ParticipantSelector';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Redirect } from "expo-router";
import { EXPECTED_TOTAL_RECORDINGS } from '@/constants/script';
import { Modal } from '@/components/Modal';

const WELCOME_SEEN_KEY = 'welcomeScreenSeen_v1';

interface SettingsScreenProps {
  allRecordings: RecordingMetadata[];
  playbackUiVersion: number;
  isDeletingId: string | null;
  isDeletingAll: boolean;
  uploadingItemId: string | null;
  canDeleteAllDisplayed: boolean;
  playbackState: {
    sound: Audio.Sound | null;
    status: PlaybackStatus;
    currentUri: string | null;
    isActive: boolean;
  };
  goToParticipantView: () => void;
  handlePlayRecording: (uri: string) => Promise<void>;
  handleDeleteRecording: (id: string, promptId: string) => void;
  handleDeleteAll: () => void;
  handleParticipantSelected: (participant: ParticipantDetails | null) => Promise<void>;
  handleCreateNewParticipant: () => void;
  participantDetails: ParticipantDetails | null;
  isRefreshing: boolean;
  onRefresh: () => Promise<void>;
  settingsTabIndex: number;
  setSettingsTabIndex: (index: number) => void;
}

interface ParticipantScreenProps {
  participantDetails: ParticipantDetails | null;
  displayedRecordings: RecordingMetadata[];
  isDeletingAll: boolean;
  playbackUiVersion: number;
  isDeletingId: string | null;
  uploadingItemId: string | null;
  handlePlayRecording: (uri: string) => Promise<void>;
  handleDeleteRecording: (id: string, promptId: string) => void;
  playbackState: {
    sound: Audio.Sound | null;
    status: PlaybackStatus;
    currentUri: string | null;
    isActive: boolean;
  };
  goToSettings: () => void;
  renderProgress: () => React.ReactNode;
  router: any; // Keep this as any for now since it's from expo-router
  isUploading: boolean;
  canDeleteAllDisplayed: boolean;
  handleDeleteAll: () => void;
  isRefreshing: boolean;
  onRefresh: () => Promise<void>;
  pendingToUploadCount: number;
  networkAvailable: boolean;
  canUpload: boolean;
  handleUploadAll: () => Promise<void>;
  uploadProgress: UploadProgressState;
}

type PlaybackStatus = 'idle' | 'loading' | 'playing' | 'error';
type ViewMode = 'participant' | 'settings' | 'select-participant' | 'create-participant' | 'edit-participant';
type UploadProgressState = { current: number; total: number } | null;

// Separate the settings screen into its own component to avoid conditional hooks
const SettingsScreen: React.FC<SettingsScreenProps> = ({
  allRecordings,
  playbackUiVersion,
  isDeletingId,
  isDeletingAll,
  uploadingItemId,
  canDeleteAllDisplayed,
  playbackState,
  goToParticipantView,
  handlePlayRecording,
  handleDeleteRecording,
  handleDeleteAll,
  handleParticipantSelected,
  handleCreateNewParticipant,
  participantDetails,
  isRefreshing,
  onRefresh,
  settingsTabIndex,
  setSettingsTabIndex,
}) => {
  const iconColor = useThemeColor({}, 'icon') || '#000';
  const dangerColor = useThemeColor({}, 'danger') || '#EF4444';
  const primaryColor = useThemeColor({}, 'tint') || '#4F46E5';
  const textColor = useThemeColor({}, 'text') || '#000';
  const secondaryTextColor = useThemeColor({ light: '#6B7280', dark: '#9CA3AF' }, 'text') || '#6B7280';

  return (
    <>
      <View className="bg-white dark:bg-neutral-800 border-b border-neutral-200 dark:border-neutral-700">
        <View className="px-4 pt-4 pb-0">
          <View className="flex-row items-center justify-between mb-4 px-1">
            <Button
              title=""
              icon="arrow-left-circle-outline"
              iconSize={26}
              onPress={goToParticipantView}
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
          extraData={[playbackUiVersion, isDeletingId, isDeletingAll, uploadingItemId]}
          renderItem={({ item }) => (
            <RecordingListItem
              recording={item}
              onPlay={handlePlayRecording}
              onDelete={() => handleDeleteRecording(item.id, item.promptId)}
              isPlaying={playbackState.currentUri === item.localUri && playbackState.status === 'playing'}
              isDeleting={isDeletingId === item.id || isDeletingAll}
              isUploadingNow={uploadingItemId === item.id}
              showParticipantCode={true}
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

// Separate participant screen component
const ParticipantScreen: React.FC<ParticipantScreenProps> = ({
  participantDetails,
  displayedRecordings,
  isDeletingAll,
  playbackUiVersion,
  isDeletingId,
  uploadingItemId,
  handlePlayRecording,
  handleDeleteRecording,
  playbackState,
  goToSettings,
  renderProgress,
  router,
  isUploading,
  canDeleteAllDisplayed,
  handleDeleteAll,
  isRefreshing,
  onRefresh,
  pendingToUploadCount,
  networkAvailable,
  canUpload,
  handleUploadAll,
  uploadProgress,
}) => {
  const iconColor = useThemeColor({}, 'icon') || '#000';
  const dangerColor = useThemeColor({}, 'danger') || '#EF4444';
  const textColor = useThemeColor({}, 'text') || '#000';
  const secondaryTextColor = useThemeColor({ light: '#6B7280', dark: '#9CA3AF' }, 'text') || '#6B7280';
  const primaryColor = useThemeColor({}, 'tint') || '#4F46E5';

  return (
    <>
      {isDeletingAll && (
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
                title=""
                icon="cog-outline"
                iconSize={26}
                onPress={goToSettings}
                className="p-1"
                iconColor={iconColor || undefined}
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
                    title="Edit"
                    icon="account-edit-outline"
                    iconColor={primaryColor || undefined}
                    iconSize={16}
                    onPress={() => router.push({
                      pathname: '/edit-participant',
                      params: { participantCode: participantDetails.code }
                    })}
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
                {/* Render Progress Bar */}
                <View className="mt-4">
                  {renderProgress()}
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
            <Text className={`text-xs font-medium ${networkAvailable ? "text-success" : "text-danger-dark dark:text-danger-light"}`}>
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
          className="bg-primary dark:bg-primary-dark w-full flex-row items-center justify-center py-3 rounded-lg"
          textClassName="text-white text-lg font-semibold ml-2 dark:text-neutral-100"
          iconColor="white"
          disabledClassName="bg-primary-light dark:bg-primary/50 opacity-70"
        />
        {isUploading && uploadProgress && uploadProgress.total > 0 && (
          <View className="mt-2 h-1.5 bg-neutral-200 dark:bg-neutral-600 rounded-full overflow-hidden flex-1">
            <View
              className="h-full bg-primary dark:bg-primary-dark rounded-full"
              style={{ width: `${(uploadProgress.current / uploadProgress.total) * 100}%` }}
            />
          </View>
        )}
      </View>
    </>
  );
};

// Main component
export default function HomeScreen() {
  const router = useRouter();
  const navigation = useNavigation();

  // --- Theme Colors ---
  const iconColor = useThemeColor({}, 'icon') || '#000';
  const dangerColor = useThemeColor({}, 'danger') || '#EF4444';
  const primaryColor = useThemeColor({}, 'tint') || '#4F46E5';
  const textColor = useThemeColor({}, 'text') || '#000';
  const secondaryTextColor = useThemeColor({ light: '#6B7280', dark: '#9CA3AF' }, 'text') || '#6B7280';
  const successColor = useThemeColor({}, 'success') || '#10B981';
  const warningColor = useThemeColor({}, 'warning') || '#F59E0B';

  // --- State Management ---
  // App Flow States
  const [isCheckingWelcome, setIsCheckingWelcome] = useState(true);
  const [hasSeenWelcome, setHasSeenWelcome] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>('participant');
  const [isSetupMode, setIsSetupMode] = useState(false);
  const [settingsTabIndex, setSettingsTabIndex] = useState(0);

  // Data States
  const [participantDetails, setParticipantDetails] = useState<ParticipantDetails | null>(null);
  const [allRecordings, setAllRecordings] = useState<RecordingMetadata[]>([]);
  const [participantCount, setParticipantCount] = useState(0);

  // Network and UI States
  const [networkAvailable, setNetworkAvailable] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isDeletingId, setIsDeletingId] = useState<string | null>(null);
  const [isDeletingAll, setIsDeletingAll] = useState(false);
  const [uploadingItemId, setUploadingItemId] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<UploadProgressState>(null);

  // Audio Playback Management
  const [playbackUiVersion, setPlaybackUiVersion] = useState(0);
  const playbackState = useRef({
    sound: null as Audio.Sound | null,
    status: 'idle' as PlaybackStatus,
    currentUri: null as string | null,
    isActive: false,
  }).current;

  // --- Utility Functions ---
  const updatePlaybackUI = useCallback(() => {
    setPlaybackUiVersion(prev => prev + 1);
  }, []);

  const triggerHaptic = useCallback((type: Haptics.ImpactFeedbackStyle = Haptics.ImpactFeedbackStyle.Light) => {
    Haptics.impactAsync(type).catch(console.error);
  }, []);

  // --- Network Management ---
  const checkNetwork = useCallback(async () => {
    try {
      const state = await Network.getNetworkStateAsync();
      const isConnected = !!(state?.isConnected) && !!(state?.isInternetReachable);
      setNetworkAvailable(isConnected);
    } catch (error) {
      console.error("HomeScreen: Error checking network:", error);
      setNetworkAvailable(false);
    }
  }, []);

  // --- Data Loading Functions ---
  const loadAllRecordings = useCallback(async () => {
    console.log("HomeScreen: Loading ALL recordings from storage...");
    try {
      const storedRecordings = await getPendingRecordings();
      if (Array.isArray(storedRecordings)) {
        const recordingsWithResetStatus = storedRecordings.map(rec => ({
          ...rec,
          uploadStatus: rec.uploadStatus === 'uploading' ? 'pending' : (rec.uploadStatus ?? 'pending')
        }));
        recordingsWithResetStatus.sort((a, b) => b.timestamp - a.timestamp);
        setAllRecordings(recordingsWithResetStatus);
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

  const loadInitialData = useCallback(async (options: { forceSetup?: boolean } = {}) => {
    console.log("HomeScreen: Loading initial data...");
    setIsLoading(true);

    try {
      let shouldEnterSetup = false;

      if (options?.forceSetup) {
        console.log("HomeScreen: Forcing setup mode.");
        shouldEnterSetup = true;
        setParticipantDetails(null);
      } else {
        const fetchedDetails = await getParticipantDetails();
        if (fetchedDetails && isValidCode(fetchedDetails.code)) {
          console.log("HomeScreen: Found valid participant details:", fetchedDetails);
          setParticipantDetails(fetchedDetails);
        } else {
          console.log("HomeScreen: No valid participant found, entering setup mode.");
          setParticipantDetails(null);
          shouldEnterSetup = true;
        }
      }

      // Always load participant count
      await loadParticipantCount();

      if (shouldEnterSetup) {
        setIsSetupMode(true);
        const allExistingParticipants = await getAllParticipants();
        if (Array.isArray(allExistingParticipants) && allExistingParticipants.length > 0) {
          console.log("HomeScreen: Existing participants found, showing selection screen.");
          setViewMode('select-participant');
        } else {
          console.log("HomeScreen: No participants exist, showing creation screen.");
          setViewMode('create-participant');
        }
      } else {
        console.log("HomeScreen: Setting view mode to participant.");
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

  // --- Audio Playback Management ---


  const unloadPlaybackSound = useCallback(async () => {
    if (playbackState.sound) {
      try {
        await playbackState.sound.unloadAsync();
      } catch (e) {
        console.error("Unload error:", e);
      }
      playbackState.sound = null;
      playbackState.status = 'idle';
      playbackState.currentUri = null;
      playbackState.isActive = false;
      updatePlaybackUI();
    }
  }, [playbackState, updatePlaybackUI]);

  const handlePlayRecording = useCallback(async (uri: string) => {
    triggerHaptic();

    // If already playing this URI, stop it
    if (playbackState.status === 'playing' && playbackState.currentUri === uri) {
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

    // For local files, check if file exists
    if (!uri.startsWith('http://') && !uri.startsWith('https://')) {
      const fileExists = await verifyFileExists(uri);
      if (!fileExists) {
        Alert.alert("Playback Error", "Audio file not found.");
        return;
      }
    }

    // Set refs to loading state
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

  // --- Data Management Functions ---
  const onRefresh = useCallback(async () => {
    setIsRefreshing(true);
    triggerHaptic();

    try {
      // First load all local recordings
      await loadAllRecordings();
      await loadParticipantCount();
      await checkNetwork();

      // Then, if we have a participant and we're online, check for server recordings
      if (participantDetails?.code && networkAvailable) {
        console.log(`Checking server for recordings for participant ${participantDetails.code}...`);

        // Import from checkParticipantExists from api.ts
        const serverData = await checkParticipantExists(participantDetails.code);

        if (serverData && serverData.recordings && serverData.recordings.length > 0) {
          console.log(`Found ${serverData.recordings.length} recordings on server for ${participantDetails.code}`);

          // Compare server recordings with local ones to find missing recordings
          const localRecordings = allRecordings.filter(
            rec => rec.participantCode === participantDetails.code
          );

          const localPromptIds = new Set(localRecordings.map(rec => rec.promptId));

          // Find recordings on server that aren't in local storage
          const missingRecordings = serverData.recordings.filter(
            serverRec => !localPromptIds.has(serverRec.prompt_id)
          );

          if (missingRecordings.length > 0) {
            console.log(`Found ${missingRecordings.length} recordings on server that are not in local storage`);

            // Ask user if they want to import these recordings
            Alert.alert(
              "Server Recordings Found",
              `Found ${missingRecordings.length} recordings on the server that are not on this device. Import them?`,
              [
                {
                  text: "No",
                  style: "cancel"
                },
                {
                  text: "Yes, Import",
                  onPress: async () => {
                    // Show importing indicator
                    setIsRefreshing(true);

                    let importedCount = 0;

                    try {
                      for (const serverRec of missingRecordings) {
                        // Create a unique ID for each imported recording
                        const uniqueId = `import_${serverRec.id || Math.random().toString(36).substring(2, 15)}`;

                        // Create local metadata record that uses the server URL
                        const localRecording: RecordingMetadata = {
                          id: uniqueId,
                          participantCode: serverRec.participant_code,
                          promptId: serverRec.prompt_id,
                          promptText: serverRec.prompt_text || serverRec.prompt_id.replace(/_/g, ' '),
                          timestamp: new Date(serverRec.uploaded_at || Date.now()).getTime(),
                          localUri: serverRec.file_url, // Use the remote URL directly
                          originalFilename: serverRec.filename_original || `${serverRec.prompt_id}.m4a`,
                          contentType: serverRec.content_type || 'audio/mp4',
                          uploaded: true, // Mark as already uploaded since we're using the server URL
                          recordingDuration: serverRec.recording_duration || 0,
                          dialect: serverData.participant.dialect || participantDetails.dialect,
                          age_range: serverData.participant.age_range || participantDetails.age_range,
                          gender: serverData.participant.gender || participantDetails.gender,
                          uploadStatus: 'pending'
                        };

                        // Save to local storage
                        await savePendingRecording(localRecording);
                        importedCount++;
                      }

                      // Reload recordings after import
                      await loadAllRecordings();

                      // Update progress for participant
                      const allParticipantRecordings = allRecordings.filter(
                        rec => rec.participantCode === participantDetails.code
                      );

                      // Update the participant's progress
                      if (allParticipantRecordings.length > 0) {
                        const updatedDetails: ParticipantDetails = {
                          ...participantDetails,
                          progress: {
                            total_recordings: allParticipantRecordings.length,
                            total_required: EXPECTED_TOTAL_RECORDINGS,
                            is_complete: allParticipantRecordings.length >= EXPECTED_TOTAL_RECORDINGS
                          }
                        };

                        await saveParticipantDetails(updatedDetails);
                        setParticipantDetails(updatedDetails);
                      }

                      Alert.alert(
                        "Import Complete",
                        `Successfully imported ${importedCount} recordings from the server.`
                      );

                    } catch (error) {
                      console.error("Error importing server recordings:", error);
                      Alert.alert(
                        "Import Error",
                        `Error importing recordings: ${error}`
                      );
                    } finally {
                      setIsRefreshing(false);
                    }
                  }
                }
              ]
            );
          } else {
            console.log("No new recordings found on server to import");
          }

          // Update participant details if needed
          if (serverData.participant) {
            // Check if we need to update local participant details with server data
            const serverProgress = serverData.participant.progress;
            const localProgress = participantDetails.progress;

            // If server reports more recordings or completion status differs, update
            if (serverProgress && (
              (serverProgress.total_recordings > localProgress.total_recordings) ||
              (serverProgress.is_complete && !localProgress.is_complete)
            )) {

              console.log("Updating participant progress from server data");

              const updatedDetails: ParticipantDetails = {
                ...participantDetails,
                dialect: serverData.participant.dialect || participantDetails.dialect,
                age_range: serverData.participant.age_range || participantDetails.age_range,
                gender: serverData.participant.gender || participantDetails.gender,
                progress: serverProgress
              };

              await saveParticipantDetails(updatedDetails);
              setParticipantDetails(updatedDetails);
            }
          }
        }
      }
    } catch (error) {
      console.error("Error during refresh with server check:", error);
    } finally {
      setIsRefreshing(false);
    }
  }, [
    loadAllRecordings,
    loadParticipantCount,
    checkNetwork,
    triggerHaptic,
    participantDetails,
    networkAvailable,
    allRecordings
  ]);

  const handleDeleteRecording = useCallback((id: string, promptId: string) => {
    if (isDeletingAll || isUploading) return;

    triggerHaptic(Haptics.ImpactFeedbackStyle.Medium);
    Alert.alert(
      "Delete Recording?",
      `Are you sure you want to delete the recording for prompt "${promptId?.replace(/_/g, ' ') || 'Unknown'}"? This cannot be undone.`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete", style: "destructive",
          onPress: async () => {
            setIsDeletingId(id);
            try {
              const deleted = await deletePendingRecordingById(id);
              if (deleted) {
                setAllRecordings(prev => prev.filter(rec => rec.id !== id));
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
    // Get the displayedRecordings directly here based on the current viewMode
    const activeRecordings = viewMode === 'settings'
      ? allRecordings
      : (participantDetails?.code
        ? allRecordings.filter(rec => rec.participantCode === participantDetails.code)
        : []);

    if (isDeletingAll || isUploading || activeRecordings.length === 0) return;

    triggerHaptic(Haptics.ImpactFeedbackStyle.Heavy);

    const isParticipantView = viewMode === 'participant' && participantDetails?.code;
    const count = activeRecordings.length;
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
    viewMode, allRecordings, participantDetails, isDeletingAll, isUploading,
    triggerHaptic, unloadPlaybackSound, loadAllRecordings
  ]);

  const handleUploadComplete = useCallback(async (response: UploadResponse) => {
    if (!participantDetails) return;

    if (response.progress) {
      const updatedDetails: ParticipantDetails = {
        ...participantDetails,
        progress: response.progress
      };
      setParticipantDetails(updatedDetails);
      await saveParticipantDetails(updatedDetails);
    }
  }, [participantDetails]);

  const handleUploadAll = useCallback(async () => {
    if (!participantDetails?.code || viewMode === 'settings' || isUploading) {
      return;
    }

    const displayedRecordings = allRecordings.filter(rec =>
      rec.participantCode === participantDetails.code
    );

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
            failCount++;
            setAllRecordings(prev => prev.map(rec =>
              rec.id === recordingMeta.id ? { ...rec, uploadStatus: 'failed' } : rec
            ));
            continue;
          }

          const uploadResult = await uploadRecording(recordingMeta);

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
      Alert.alert("Upload Error", "An unexpected error occurred during the upload process.");
    } finally {
      setIsUploading(false);
      setUploadingItemId(null);
      setUploadProgress(null);
    }
  }, [
    participantDetails, viewMode, isUploading, allRecordings, networkAvailable,
    triggerHaptic, checkNetwork, verifyFileExists, handleUploadComplete
  ]);

  // --- UI Navigation Functions ---
  const handleSetupComplete = useCallback(async (details: ParticipantDetails) => {
    setIsLoading(true);
    try {
      if (!details || !isValidCode(details.code)) {
        throw new Error("Invalid participant code provided.");
      }

      // Save the participant details
      const savedSuccessfully = await saveParticipantDetails(details);
      if (!savedSuccessfully) {
        throw new Error("Failed to save participant details to storage.");
      }

      // Reload all recordings to get the imported ones
      await loadAllRecordings();

      // Count recordings for this participant to update progress
      const participantRecordings = allRecordings.filter(
        rec => rec.participantCode === details.code
      );

      // Only update if we found recordings
      if (participantRecordings.length > 0) {
        const updatedDetails: ParticipantDetails = {
          ...details,
          progress: {
            total_recordings: participantRecordings.length,
            total_required: EXPECTED_TOTAL_RECORDINGS,
            is_complete: participantRecordings.length >= EXPECTED_TOTAL_RECORDINGS
          }
        };

        // Save the updated progress
        await saveParticipantDetails(updatedDetails);
        setParticipantDetails(updatedDetails);
      } else {
        setParticipantDetails(details);
      }

      // Complete setup
      await loadParticipantCount();
      setIsSetupMode(false);
      setViewMode('participant');

    } catch (error: any) {
      Alert.alert("Error", `Failed to finalize setup: ${error.message || 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  }, [allRecordings, loadAllRecordings, loadParticipantCount]);

  const handleParticipantSelected = useCallback(async (participant: ParticipantDetails | null) => {
    if (participant) {
      const saved = await saveParticipantDetails(participant);
      if (saved) {
        setParticipantDetails(participant);
        setIsSetupMode(false);
        setViewMode('participant');
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

  const goToParticipantView = useCallback(() => {
    setViewMode('participant');
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

  useEffect(() => {
    loadInitialData();
    return () => { unloadPlaybackSound(); };
  }, [loadInitialData, unloadPlaybackSound]);

  useFocusEffect(
    useCallback(() => {
      if (!isSetupMode) {
        loadInitialData();
        unloadPlaybackSound();
      }
      return () => { unloadPlaybackSound(); };
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

  // --- Derived State using useMemo ---
  const displayedRecordings = useMemo(() => {
    if (viewMode === 'settings') return allRecordings;

    if (viewMode === 'participant' && participantDetails?.code) {
      return allRecordings.filter(rec => rec.participantCode === participantDetails.code);
    }
    return [];
  }, [allRecordings, viewMode, participantDetails]);

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

  // --- Render Progress UI ---
  const renderProgress = useCallback(() => {
    if (viewMode !== 'participant' || !participantDetails) return null;

    const localTotal = displayedRecordings.length;
    const localRequired = EXPECTED_TOTAL_RECORDINGS;
    const isLocallyComplete = localTotal >= localRequired;

    // Calculate uploaded count
    const uploadedCount = displayedRecordings.filter(rec => rec.uploaded).length;

    const localPercentage = localRequired > 0
      ? Math.min(100, Math.round((localTotal / localRequired) * 100))
      : 0;

    const uploadedPercentage = localRequired > 0
      ? Math.min(100, Math.round((uploadedCount / localRequired) * 100))
      : 0;

    // Server progress
    const serverProgress = participantDetails.progress;
    const isServerComplete = serverProgress?.is_complete ?? false;

    const displayPercentage = localPercentage;

    let progressText = `${localTotal} / ${localRequired} Recorded`;
    let uploadedText = '';
    if (uploadedCount > 0 && !isUploading) {
      uploadedText = ` (${uploadedCount} Uploaded)`;
    }

    return (
      <View>
        {/* Progress Text */}
        <View className="flex-row justify-between mb-1 px-1">
          <Text style={{ color: secondaryTextColor }} className="text-sm">
            {progressText} <Text className='text-success/80'>{uploadedText}</Text>
          </Text>
          <Text style={{ color: isLocallyComplete ? primaryColor : successColor }} className="text-sm font-bold">
            {displayPercentage}%
          </Text>
        </View>

        {/* Progress Bar Container */}
        <View className="h-3 bg-neutral-200 dark:bg-neutral-700 rounded-full overflow-hidden relative">
          {/* Local Progress Bar */}
          <View
            className="absolute top-0 left-0 bottom-0 bg-primary dark:bg-primary-dark rounded-full"
            style={{ width: `${localPercentage}%` }}
          />
          {/* Uploaded Progress Bar */}
          {uploadedPercentage > 0 && (
            <View
              className="absolute top-0 left-0 bottom-0 bg-success dark:bg-success-dark rounded-full"
              style={{ width: `${uploadedPercentage}%` }}
            />
          )}
        </View>

        {/* Completion Text */}
        {isLocallyComplete && !isUploading && (
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
      </View>
    );
  }, [
    viewMode, participantDetails, displayedRecordings, isUploading,
    secondaryTextColor, primaryColor, successColor, warningColor
  ]);

  // --- Component Renders ---
  if (isCheckingWelcome || isLoading) {
    return (
      <ThemedSafeAreaView className="flex-1 justify-center items-center">
        <ActivityIndicator size="large" color={primaryColor} />
      </ThemedSafeAreaView>
    );
  }

  if (!hasSeenWelcome) {
    return <Redirect href="/welcome" />;
  }

  // Setup mode content
  if (isSetupMode) {
    if (viewMode === 'select-participant') {
      return (
        <ThemedSafeAreaView className="flex-1">
          <ParticipantSelector
            currentParticipant={participantDetails}
            onSelectParticipant={handleParticipantSelected}
            onCreateNewParticipant={handleCreateNewParticipant}
          />
        </ThemedSafeAreaView>
      );
    } else if (viewMode === 'create-participant') {
      return (
        <ThemedSafeAreaView className="flex-1">
          <SetupScreenContent
            onSetupComplete={handleSetupComplete}
            isNewParticipant={true}
            onCancel={() => {
              setIsSetupMode(false);
              setViewMode('participant');
              if (!participantDetails) {
                loadInitialData();
              }
            }}
          />
        </ThemedSafeAreaView>
      );
    } else if (viewMode === 'edit-participant') {
      return (
        <ThemedSafeAreaView className="flex-1">
          <SetupScreenContent
            onSetupComplete={handleSetupComplete}
            initialDetails={participantDetails}
            onCancel={() => {
              setViewMode('participant');
              setIsSetupMode(false);
            }}
          />
        </ThemedSafeAreaView>
      );
    }

    // Default fallback for setup mode
    return (
      <ThemedSafeAreaView className="flex-1 justify-center items-center">
        <ActivityIndicator size="large" color={primaryColor} />
        <Text className="mt-4 text-center" style={{ color: textColor }}>Loading setup...</Text>
      </ThemedSafeAreaView>
    );
  }

  // Settings mode content
  if (viewMode === 'settings') {
    return (
      <ThemedSafeAreaView className="flex-1">
        <SettingsScreen
          allRecordings={allRecordings}
          playbackUiVersion={playbackUiVersion}
          isDeletingId={isDeletingId}
          isDeletingAll={isDeletingAll}
          uploadingItemId={uploadingItemId}
          canDeleteAllDisplayed={canDeleteAllDisplayed}
          playbackState={playbackState}
          goToParticipantView={goToParticipantView}
          handlePlayRecording={handlePlayRecording}
          handleDeleteRecording={handleDeleteRecording}
          handleDeleteAll={handleDeleteAll}
          handleParticipantSelected={handleParticipantSelected}
          handleCreateNewParticipant={handleCreateNewParticipant}
          participantDetails={participantDetails}
          isRefreshing={isRefreshing}
          onRefresh={onRefresh}
          settingsTabIndex={settingsTabIndex}
          setSettingsTabIndex={setSettingsTabIndex}
        />
      </ThemedSafeAreaView>
    );
  }

  // Participant mode content
  return (
    <ThemedSafeAreaView className="flex-1">
      <ParticipantScreen
        participantDetails={participantDetails}
        displayedRecordings={displayedRecordings}
        isDeletingAll={isDeletingAll}
        playbackUiVersion={playbackUiVersion}
        isDeletingId={isDeletingId}
        uploadingItemId={uploadingItemId}
        handlePlayRecording={handlePlayRecording}
        handleDeleteRecording={handleDeleteRecording}
        playbackState={playbackState}
        goToSettings={goToSettings}
        renderProgress={renderProgress}
        router={router}
        isUploading={isUploading}
        canDeleteAllDisplayed={canDeleteAllDisplayed}
        handleDeleteAll={handleDeleteAll}
        isRefreshing={isRefreshing}
        onRefresh={onRefresh}
        pendingToUploadCount={pendingToUploadCount}
        networkAvailable={networkAvailable}
        canUpload={canUpload}
        handleUploadAll={handleUploadAll}
        uploadProgress={uploadProgress}
      />
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
