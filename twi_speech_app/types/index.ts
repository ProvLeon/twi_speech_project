export interface ScriptPrompt {
  id: string;
  type: 'scripted' | 'spontaneous';
  text: string; // The main prompt text (Twi for scripted, instructions for spontaneous)
  meaning?: string; // Optional: English meaning/translation of the 'text' field
}

export interface RecordingSection {
  id: string;
  title: string;
  description: string;
  prompts: ScriptPrompt[];
}


export interface RecordingProgress {
  total_recordings: number;
  total_required: number;
  is_complete: boolean;
}

// NEW: Define structure for participant details
export interface ParticipantDetails {
  code: string;
  dialect?: string;
  age_range?: string;
  gender?: string;
  progress: RecordingProgress;
}


export interface RecordingMetadata {
  id: string;
  participantCode: string; // Keep code here for individual recording context
  promptText: string;
  promptId: string;
  timestamp: number;
  localUri: string;
  originalFilename: string;
  contentType: string;
  uploaded: boolean;
  recordingDuration?: number; // Added: Duration in milliseconds

  // Optional metadata (populated from ParticipantDetails)
  dialect?: string;
  age_range?: string;
  gender?: string;
  session_id?: string; // Optional session identifier if needed later
}

export interface UploadResponse {
  message: string;
  file_url: string;
  recording_db_id: string;
  speaker_db_id: string;
  participant_code: string;
  prompt_id: string;
  progress: RecordingProgress; // Backend sends the updated progress
}
