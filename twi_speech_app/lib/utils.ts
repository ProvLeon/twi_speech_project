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
