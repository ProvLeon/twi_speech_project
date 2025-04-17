export const isValidCode = (code?: string | null): boolean => {
  // Check if code is a string and not null/empty
  if (typeof code !== 'string' || !code) {
    return false;
  }
  // Regex: Starts with "TWI_Speaker_", followed by exactly 3 digits ($ anchors to end)
  return /^(TWI_Speaker_)[0-9]{3}$/.test(code.trim());
}

// Keep isValidText as is, or adjust length check if needed
export const isValidText = (text?: string) => !!text && text.trim().length > 0;

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
