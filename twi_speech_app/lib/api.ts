import { UPLOAD_AUDIO_ENDPOINT } from '@/constants/api';
import { RecordingMetadata } from '@/types';
import * as FileSystem from 'expo-file-system';
import * as Network from 'expo-network';

export const uploadRecording = async (
  recordingMeta: RecordingMetadata
): Promise<boolean> => {
  const networkState = await Network.getNetworkStateAsync();
  if (!networkState.isConnected || !networkState.isInternetReachable) {
    console.log('Upload skipped: No internet connection.');
    return false; // Indicate upload skipped/failed due to network
  }

  const fileUri = recordingMeta.localUri;
  const uploadUrl = UPLOAD_AUDIO_ENDPOINT;

  // Check if file exists before attempting upload
  try {
    const fileInfo = await FileSystem.getInfoAsync(fileUri);
    if (!fileInfo.exists) {
      console.error(`[api] Upload failed: File not found at ${fileUri}`);
      return false; // File doesn't exist, cannot upload
    }
  } catch (error) {
    console.error(`[api] Error checking file info for ${fileUri}:`, error);
    return false; // Error accessing file system
  }

  const formData = new FormData();

  // --- Append Metadata Fields (Match backend Form fields) ---
  // Required fields:
  formData.append('participant_code', recordingMeta.participantCode);
  formData.append('prompt_id', recordingMeta.promptId);
  // Make sure promptText exists and is not null/undefined
  formData.append('prompt_text', recordingMeta.promptText ?? ''); // Send empty string if null/undefined

  // Optional fields (append only if they have a non-empty value):
  if (recordingMeta.dialect && recordingMeta.dialect.trim()) {
    formData.append('dialect', recordingMeta.dialect);
  }
  if (recordingMeta.age_range && recordingMeta.age_range.trim()) {
    formData.append('age_range', recordingMeta.age_range);
  }
  if (recordingMeta.gender && recordingMeta.gender.trim()) {
    formData.append('gender', recordingMeta.gender);
  }
  if (recordingMeta.session_id && recordingMeta.session_id.trim()) {
    formData.append('session_id', recordingMeta.session_id);
  }
  // --- End Metadata Fields ---

  // --- Append File ---
  // Use the permanent file URI and ensure type is correct
  formData.append('file', {
    uri: fileUri,
    name: recordingMeta.originalFilename,
    type: recordingMeta.contentType || 'audio/mp4', // Provide a default if somehow missing
  } as any); // Type assertion often needed for RN FormData
  // --- End Append File ---

  console.log(`[api] Attempting to upload ${recordingMeta.originalFilename} for prompt ${recordingMeta.promptId}`);
  // console.log('[api] Uploading metadata:', JSON.stringify(Object.fromEntries(formData.entries()))); // Log carefully

  try {
    const response = await fetch(uploadUrl, {
      method: 'POST',
      body: formData,
      headers: {
        // IMPORTANT: Do NOT manually set Content-Type for FormData,
        // the browser/fetch API handles it with the correct boundary.
        // 'Content-Type': 'multipart/form-data', // <--- REMOVE/COMMENT OUT
      },
    });

    const responseText = await response.text(); // Read response body for debugging
    console.log(`[api] Upload response status for ${recordingMeta.id}: ${response.status}`);
    // console.log('[api] Upload response body:', responseText); // Log full body if needed

    if (response.ok) { // Status 200-299 (expecting 201 Created)
      console.log(`[api] Upload successful for ${recordingMeta.id}`);
      try {
        const responseData = JSON.parse(responseText);
        console.log('[api] Upload response data:', responseData);
        // Optionally use responseData (e.g., returned DB IDs) if needed
      } catch (parseError) {
        console.warn('[api] Upload response was OK, but body was not valid JSON:', parseError);
      }
      return true;
    } else {
      console.error(`[api] Upload failed for ${recordingMeta.id}: ${response.status} ${responseText}`);
      // Optionally handle specific error codes (e.g., 422 for validation errors)
      return false;
    }
  } catch (error) {
    console.error(`[api] Upload network error for ${recordingMeta.id}:`, error);
    return false;
  }
};
