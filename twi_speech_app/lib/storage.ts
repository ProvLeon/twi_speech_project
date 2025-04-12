import { ParticipantDetails, RecordingMetadata } from '@/types';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system';
import { isValidCode } from './utils';

// --- Constants ---
const PARTICIPANT_DETAILS_KEY = 'participantDetails_v2'; // Current participant
const ALL_PARTICIPANTS_KEY = 'allParticipants_v1'; // New key for storing all participants
const PENDING_RECORDINGS_KEY = 'pendingRecordings_v2';
const recordingsDir = FileSystem.documentDirectory + 'recordings/';

// --- Helper Functions ---
const ensureDirExists = async () => {
  try {
    const dirInfo = await FileSystem.getInfoAsync(recordingsDir);
    if (!dirInfo.exists) {
      console.log("[storage] Recordings directory doesn't exist, creating...");
      await FileSystem.makeDirectoryAsync(recordingsDir, { intermediates: true });
    }
  } catch (error) {
    console.error("[storage] Failed to ensure recordings directory exists:", error);
    // Decide if this is critical enough to throw
  }
};

const safeJsonParse = <T>(jsonString: string | null, defaultValue: T): T => {
  if (jsonString === null) return defaultValue;
  try {
    return JSON.parse(jsonString) as T;
  } catch (e) {
    console.error('[storage] Failed to parse JSON:', e, 'Data:', jsonString?.substring(0, 100)); // Log snippet
    return defaultValue;
  }
};

// --- Multi-Participant Management ---

// Get list of all saved participants
export const getAllParticipants = async (): Promise<ParticipantDetails[]> => {
  try {
    const jsonValue = await AsyncStorage.getItem(ALL_PARTICIPANTS_KEY);
    const participants = safeJsonParse<ParticipantDetails[]>(jsonValue, []);
    // Filter to ensure only valid participants are returned
    return participants.filter(p => p && isValidCode(p.code));
  } catch (e) {
    console.error('[storage] Failed to fetch all participants:', e);
    return [];
  }
};

// Add a participant to the list (or update if exists)
export const saveParticipantToList = async (details: ParticipantDetails): Promise<boolean> => {
  try {
    if (!isValidCode(details.code)) {
      console.error('[storage] Attempted to save invalid participant code:', details.code);
      return false;
    }

    const allParticipants = await getAllParticipants();

    // Check if this participant already exists (by code)
    const existingIndex = allParticipants.findIndex(p => p.code === details.code);

    if (existingIndex >= 0) {
      // Update existing participant
      allParticipants[existingIndex] = details;
    } else {
      // Add new participant
      allParticipants.push(details);
    }

    // Save updated list
    const jsonValue = JSON.stringify(allParticipants);
    await AsyncStorage.setItem(ALL_PARTICIPANTS_KEY, jsonValue);
    console.log(`[storage] Saved participant ${details.code} to participant list`);
    return true;
  } catch (e) {
    console.error('[storage] Failed to save participant to list:', e);
    return false;
  }
};

// Delete a participant from the list
export const deleteParticipantFromList = async (participantCode: string): Promise<boolean> => {
  try {
    const allParticipants = await getAllParticipants();
    const filteredParticipants = allParticipants.filter(p => p.code !== participantCode);

    // If no change in length, participant wasn't found
    if (allParticipants.length === filteredParticipants.length) {
      console.warn(`[storage] Participant ${participantCode} not found for deletion from list`);
      return false;
    }

    // Save updated list
    const jsonValue = JSON.stringify(filteredParticipants);
    await AsyncStorage.setItem(ALL_PARTICIPANTS_KEY, jsonValue);
    console.log(`[storage] Removed participant ${participantCode} from list`);

    // If this was the current participant, clear current participant
    const currentParticipant = await getParticipantDetails();
    if (currentParticipant?.code === participantCode) {
      await AsyncStorage.removeItem(PARTICIPANT_DETAILS_KEY);
      console.log(`[storage] Cleared current participant (${participantCode})`);
    }

    return true;
  } catch (e) {
    console.error(`[storage] Failed to delete participant ${participantCode} from list:`, e);
    return false;
  }
};

// Get a specific participant's details by code
export const getParticipantByCode = async (code: string): Promise<ParticipantDetails | null> => {
  try {
    const allParticipants = await getAllParticipants();
    const participant = allParticipants.find(p => p.code === code);
    return participant || null;
  } catch (e) {
    console.error(`[storage] Failed to get participant by code ${code}:`, e);
    return null;
  }
};

// --- Current Participant Details ---
export const saveParticipantDetails = async (details: ParticipantDetails): Promise<boolean> => {
  try {
    if (!details || typeof details.code !== 'string' || !isValidCode(details.code)) { // Use validator
      console.error('[storage] Attempted to save invalid participant details:', details);
      return false;
    }

    // Save to current participant
    const jsonValue = JSON.stringify(details);
    await AsyncStorage.setItem(PARTICIPANT_DETAILS_KEY, jsonValue);

    // Also add/update in all participants list
    await saveParticipantToList(details);

    console.log('[storage] Participant details saved successfully:', details);
    return true;
  } catch (e) {
    console.error('[storage] Failed to save participant details', e);
    return false;
  }
};

export const getParticipantDetails = async (): Promise<ParticipantDetails | null> => {
  try {
    const jsonValue = await AsyncStorage.getItem(PARTICIPANT_DETAILS_KEY);
    const details = safeJsonParse<ParticipantDetails | null>(jsonValue, null);
    if (details && isValidCode(details.code)) { // Use validator
      return details;
    }
    if (jsonValue) { // Log if data existed but was invalid
      console.warn('[storage] Participant details found but failed validation:', details);
    }
    return null;
  } catch (e) {
    console.error('[storage] Failed to fetch participant details', e);
    return null;
  }
};

export const getParticipantCode = async (): Promise<string | null> => {
  const details = await getParticipantDetails();
  return details?.code ?? null;
};

// --- Pending Recordings Metadata ---
export const getPendingRecordings = async (): Promise<RecordingMetadata[]> => {
  try {
    const jsonValue = await AsyncStorage.getItem(PENDING_RECORDINGS_KEY);
    const recordings = safeJsonParse<RecordingMetadata[]>(jsonValue, []);
    // Add a filter step to remove potentially invalid entries during load
    return recordings.filter(rec => rec && rec.id && rec.localUri && rec.participantCode && rec.promptId);
  } catch (e) {
    console.error('[storage] Failed to fetch pending recordings key from AsyncStorage', e);
    return [];
  }
};

// Get recordings for a specific participant
export const getRecordingsForParticipant = async (participantCode: string): Promise<RecordingMetadata[]> => {
  try {
    const allRecordings = await getPendingRecordings();
    return allRecordings.filter(rec => rec.participantCode === participantCode);
  } catch (e) {
    console.error(`[storage] Failed to get recordings for participant ${participantCode}:`, e);
    return [];
  }
};

// Saves/Overwrites recording metadata
export const savePendingRecording = async (newRecordingMeta: RecordingMetadata): Promise<boolean> => {
  try {
    // Add more robust validation
    if (!newRecordingMeta || !newRecordingMeta.id || !newRecordingMeta.participantCode || !newRecordingMeta.promptId || !newRecordingMeta.localUri || !newRecordingMeta.originalFilename || !newRecordingMeta.promptText) {
      console.error("[storage] Attempted to save invalid/incomplete recording metadata:", newRecordingMeta);
      return false;
    }

    const currentRecordings = await getPendingRecordings(); // Already filtered for basic validity
    let updatedRecordings = [...currentRecordings];
    const existingIndex = updatedRecordings.findIndex(
      (rec) => rec.promptId === newRecordingMeta.promptId && rec.participantCode === newRecordingMeta.participantCode
    );

    if (existingIndex !== -1) {
      const oldRecordingMeta = updatedRecordings[existingIndex];
      console.log(`[storage] Overwriting previous recording for prompt ${newRecordingMeta.promptId} by ${newRecordingMeta.participantCode}. Old URI: ${oldRecordingMeta.localUri}`);
      // Ensure deleteLocalRecording is awaited and handles errors gracefully
      await deleteLocalRecording(oldRecordingMeta.localUri);
      updatedRecordings[existingIndex] = newRecordingMeta;
    } else {
      updatedRecordings.push(newRecordingMeta);
    }

    const jsonValue = JSON.stringify(updatedRecordings);
    await AsyncStorage.setItem(PENDING_RECORDINGS_KEY, jsonValue);
    console.log(`[storage] Saved/Updated metadata for recording ID ${newRecordingMeta.id}`);
    return true;
  } catch (e) {
    console.error('[storage] Failed to save pending recording metadata', e);
    return false;
  }
};

// Deletes metadata and file
export const deletePendingRecordingById = async (recordingId: string): Promise<boolean> => {
  let recordingToDelete: RecordingMetadata | undefined;
  try {
    const currentRecordings = await getPendingRecordings();
    recordingToDelete = currentRecordings.find(rec => rec.id === recordingId);

    if (!recordingToDelete) {
      console.warn(`[storage] Recording ID ${recordingId} not found for deletion.`);
      return false;
    }

    const updatedRecordings = currentRecordings.filter(rec => rec.id !== recordingId);
    const jsonValue = JSON.stringify(updatedRecordings);
    await AsyncStorage.setItem(PENDING_RECORDINGS_KEY, jsonValue);
    console.log(`[storage] Removed metadata for recording ID ${recordingId}`);

    // Delete the file *after* metadata update is confirmed (or attempt both regardless)
    await deleteLocalRecording(recordingToDelete.localUri);

    return true;
  } catch (e) {
    console.error(`[storage] Failed to delete recording ${recordingId}`, e);
    // Attempt to delete file even if metadata removal failed? (Optional)
    if (recordingToDelete?.localUri) {
      await deleteLocalRecording(recordingToDelete.localUri);
    }
    return false;
  }
};

// Updates only the 'uploaded' status
export const updateRecordingUploadedStatus = async (recordingId: string, uploaded: boolean): Promise<boolean> => {
  try {
    const currentRecordings = await getPendingRecordings();
    let found = false;
    const updatedRecordings = currentRecordings.map((rec) => {
      if (rec.id === recordingId) {
        found = true;
        return { ...rec, uploaded: uploaded };
      }
      return rec;
    });

    if (!found) {
      console.warn(`[storage] Recording ID ${recordingId} not found for status update.`);
      return false;
    }

    const jsonValue = JSON.stringify(updatedRecordings);
    await AsyncStorage.setItem(PENDING_RECORDINGS_KEY, jsonValue);
    console.log(`[storage] Updated upload status for ${recordingId} to ${uploaded}`);
    return true;
  } catch (e) {
    console.error('[storage] Failed to update recording upload status', e);
    return false;
  }
};

// --- File System Operations ---
export const saveRecordingFile = async (tempUri: string, filename: string): Promise<string | null> => {
  try {
    await ensureDirExists();
    const permanentUri = recordingsDir + filename;
    console.log(`[storage] Moving recording from ${tempUri} to ${permanentUri}`);
    await FileSystem.moveAsync({ from: tempUri, to: permanentUri });
    console.log('[storage] File moved successfully');
    return permanentUri;
  } catch (e) {
    console.error('[storage] Failed to move recording file', e);
    try { await FileSystem.deleteAsync(tempUri, { idempotent: true }); } catch (deleteError) { /* Log only */ console.error('[storage] Cleanup delete failed:', deleteError); }
    return null;
  }
};

export const deleteLocalRecording = async (localUri: string | null | undefined): Promise<void> => {
  if (!localUri) return; // Silently ignore invalid URIs
  try {
    const fileInfo = await FileSystem.getInfoAsync(localUri);
    if (fileInfo.exists) {
      await FileSystem.deleteAsync(localUri, { idempotent: true });
      console.log(`[storage] Deleted local file: ${localUri}`);
    } else {
      // console.log(`[storage] Local file already deleted: ${localUri}`); // Less noisy log
    }
  } catch (e) {
    console.error(`[storage] Failed to delete local file ${localUri}:`, e);
  }
};

// --- Bulk Deletion ---
export const deleteAllDeviceRecordings = async (): Promise<{ deletedCount: number; failedCount: number }> => {
  let deletedFiles = 0;
  let failedFiles = 0;
  const currentRecordings = await getPendingRecordings();
  const totalCount = currentRecordings.length;

  if (totalCount === 0) return { deletedCount: 0, failedCount: 0 };
  console.log(`[storage] Attempting to delete all ${totalCount} device recordings...`);

  // Try deleting all files
  const deletePromises = currentRecordings.map(rec =>
    deleteLocalRecording(rec.localUri).then(() => true).catch(() => false)
  );
  const fileResults = await Promise.all(deletePromises);
  deletedFiles = fileResults.filter(Boolean).length;
  failedFiles = totalCount - deletedFiles;

  // Attempt to clear all metadata
  try {
    await AsyncStorage.removeItem(PENDING_RECORDINGS_KEY);
    console.log(`[storage] Cleared all recording metadata. File deletions - Success: ${deletedFiles}, Failed: ${failedFiles}`);
  } catch (e) {
    console.error('[storage] Failed to clear metadata during deleteAllDeviceRecordings:', e);
    // If metadata clearing fails, the result is less certain, but report file outcomes
  }

  // Return count based on successful file deletions as primary metric
  return { deletedCount: deletedFiles, failedCount: failedFiles };
};

export const deleteAllRecordingsForParticipant = async (participantCodeToDelete: string): Promise<{ deletedCount: number; failedCount: number }> => {
  let deletedFiles = 0;
  let failedFiles = 0;
  let participantRecordings: RecordingMetadata[] = [];
  let remainingRecordings: RecordingMetadata[] = [];

  try {
    const allRecordings = await getPendingRecordings();
    participantRecordings = allRecordings.filter(rec => rec.participantCode === participantCodeToDelete);
    remainingRecordings = allRecordings.filter(rec => rec.participantCode !== participantCodeToDelete);
    const totalCount = participantRecordings.length;

    if (totalCount === 0) {
      console.log(`[storage] No recordings found for participant ${participantCodeToDelete} to delete.`);
      return { deletedCount: 0, failedCount: 0 };
    }
    console.log(`[storage] Attempting to delete ${totalCount} recordings for participant ${participantCodeToDelete}...`);

    // Delete files for this participant
    const deletePromises = participantRecordings.map(rec =>
      deleteLocalRecording(rec.localUri).then(() => true).catch(() => false)
    );
    const fileResults = await Promise.all(deletePromises);
    deletedFiles = fileResults.filter(Boolean).length;
    failedFiles = totalCount - deletedFiles;

    // Save the filtered list (excluding the deleted participant's) back
    const jsonValue = JSON.stringify(remainingRecordings);
    await AsyncStorage.setItem(PENDING_RECORDINGS_KEY, jsonValue);

    console.log(`[storage] Deleted recordings for participant ${participantCodeToDelete}. Successful file deletes: ${deletedFiles}, Failed: ${failedFiles}`);
    return { deletedCount: deletedFiles, failedCount: failedFiles };

  } catch (e) {
    console.error(`[storage] Failed during deleteAllRecordingsForParticipant (${participantCodeToDelete}):`, e);
    // Return file deletion attempt results even if metadata save fails
    return { deletedCount: deletedFiles, failedCount: failedFiles || participantRecordings.length }; // Assume all failed if metadata save error occurred
  }
};

// Delete participant and all their recordings
export const deleteParticipantWithRecordings = async (participantCode: string): Promise<boolean> => {
  try {
    // First delete all recordings
    const { deletedCount, failedCount } = await deleteAllRecordingsForParticipant(participantCode);
    console.log(`[storage] Deleted ${deletedCount} recordings for participant ${participantCode} (${failedCount} failed)`);

    // Then remove from participant list
    const removed = await deleteParticipantFromList(participantCode);
    if (!removed) {
      console.warn(`[storage] Could not remove participant ${participantCode} from list`);
    }

    return true;
  } catch (e) {
    console.error(`[storage] Failed to delete participant with recordings: ${participantCode}`, e);
    return false;
  }
};
