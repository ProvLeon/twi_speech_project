import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Alert, ActivityIndicator, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { Button } from '@/components/Button';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { ParticipantDetails } from '@/types';
import { isValidCode } from '@/lib/utils';
import { ThemedView } from './ThemedView';
import { useThemeColor } from '@/hooks/useThemeColor';
import { getAllParticipants } from '@/lib/storage';

interface SetupScreenContentProps {
  onSetupComplete: (details: ParticipantDetails) => Promise<void>;
  initialDetails?: ParticipantDetails | null;
  isNewParticipant?: boolean; // Add this prop to distinguish between edit and create
  onCancel?: () => void; // Add this prop for cancellation
}

export default function SetupScreenContent({
  onSetupComplete,
  initialDetails,
  isNewParticipant = false,
  onCancel
}: SetupScreenContentProps) {
  // State for form inputs, initialize with initialDetails if provided
  const [code, setCode] = useState(initialDetails?.code || '');
  const [dialect, setDialect] = useState(initialDetails?.dialect || '');
  const [ageRange, setAgeRange] = useState(initialDetails?.age_range || '');
  const [gender, setGender] = useState(initialDetails?.gender || '');
  const [existingCodes, setExistingCodes] = useState<string[]>([]);

  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingCodes, setIsLoadingCodes] = useState(isNewParticipant); // Only load if creating new participant
  const [codeError, setCodeError] = useState('');
  const [dialectError, setDialectError] = useState('');
  const [ageRangeError, setAgeRangeError] = useState('');
  const [genderError, setGenderError] = useState('');

  const inputBgColor = useThemeColor({}, 'background');
  const textColor = useThemeColor({}, 'text');
  const placeholderColor = useThemeColor({ light: '#9CA3AF', dark: '#6B7280' }, 'text');
  const borderColor = useThemeColor({ light: '#D1D5DB', dark: '#4B5563' }, 'border');
  const errorBorderColor = useThemeColor({ light: '#EF4444', dark: '#DC2626' }, 'danger');

  // Load existing participant codes when creating a new participant
  useEffect(() => {
    if (isNewParticipant) {
      const loadExistingCodes = async () => {
        try {
          setIsLoadingCodes(true);
          const participants = await getAllParticipants();
          setExistingCodes(participants.map(p => p.code));
        } catch (error) {
          console.error("Error loading existing codes:", error);
          // Don't set any error UI, just log
        } finally {
          setIsLoadingCodes(false);
        }
      };

      loadExistingCodes();
    }
  }, [isNewParticipant]);

  // Effect to update form if initialDetails change
  useEffect(() => {
    if (initialDetails) {
      setCode(initialDetails.code || '');
      setDialect(initialDetails.dialect || '');
      setAgeRange(initialDetails.age_range || '');
      setGender(initialDetails.gender || '');
    }
  }, [initialDetails]);

  const validateInputs = (): boolean => {
    let isValid = true;
    setCodeError('');
    setDialectError('');
    setAgeRangeError('');
    setGenderError('');

    // Validate Participant Code (Required)
    if (!isValidCode(code)) {
      setCodeError('Code format: TWI_Speaker_ followed by exactly 3 digits');
      isValid = false;
    }

    // When creating new participant, check for duplicate codes
    if (isNewParticipant && isValidCode(code) && existingCodes.includes(code)) {
      setCodeError('This participant code already exists');
      isValid = false;
    }

    // Optional fields: only validate if non-empty and just whitespace
    if (dialect && dialect.trim().length === 0) {
      setDialectError('Dialect cannot be only whitespace');
      isValid = false;
    }
    if (ageRange && ageRange.trim().length === 0) {
      setAgeRangeError('Age Range cannot be only whitespace');
      isValid = false;
    }
    if (gender && gender.trim().length === 0) {
      setGenderError('Gender cannot be only whitespace');
      isValid = false;
    }

    return isValid;
  };

  const handleSave = async () => {
    if (!validateInputs()) {
      Alert.alert('Validation Error', 'Please correct the highlighted fields.');
      return;
    }

    setIsLoading(true);
    setCodeError('');

    // Trim values before creating the details object
    const participantDetails: ParticipantDetails = {
      code: code.trim(),
      dialect: dialect.trim() || undefined,
      age_range: ageRange.trim() || undefined,
      gender: gender.trim() || undefined,
    };

    try {
      await onSetupComplete(participantDetails);
    } catch (error: any) {
      console.error("[SetupScreenContent] Error during setup save callback:", error);
      Alert.alert('Save Error', `Could not save participant details. ${error.message || 'Unknown error'}`);
      setIsLoading(false);
    }
  };

  const getInputStyle = (hasError: boolean) => `
    border-2 rounded-lg p-3 text-base
    dark:text-white dark:placeholder-neutral-400
    focus:border-primary focus:ring-1 focus:ring-primary
    ${hasError ? 'border-danger dark:border-danger-dark' : 'border-neutral-300 dark:border-neutral-600'}
  `;

  return (
    <ThemedView className="flex-1">
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={{ flex: 1 }}
      >
        <ScrollView
          contentContainerStyle={{ flexGrow: 1, justifyContent: 'center' }}
          className="p-6"
          keyboardShouldPersistTaps="handled"
        >
          {onCancel && <View className="absolute top-0 right-0">
            <Button
              title=""
              onPress={onCancel}
              className="flex-1 mr-2 p-2 rounded-lg bg-neutral-300 dark:bg-neutral-700 items-center"
              textClassName="text-neutral-800 dark:text-neutral-100 text-lg font-medium"
              icon="close"
            />
          </View>}
          <View className="items-center mb-8">
            <MaterialCommunityIcons
              name={isNewParticipant ? "account-plus" : "account-edit-outline"}
              size={64}
              color="#4F46E5"
            />
            <Text className="text-3xl font-bold mt-4 text-neutral-800 text-center dark:text-neutral-100">
              {isNewParticipant ? 'Create Participant' : 'Update Participant'}
            </Text>
            <Text className="text-lg text-neutral-600 mt-1.5 text-center dark:text-neutral-300">
              {isNewParticipant
                ? 'Create a new participant profile.'
                : 'Update participant details.'}
            </Text>
          </View>

          {/* Participant Code */}
          <Text className="text-sm font-semibold mb-1 text-neutral-700 uppercase tracking-wider dark:text-neutral-400">
            Participant Code *
          </Text>
          <TextInput
            className={getInputStyle(!!codeError)}
            style={{
              backgroundColor: inputBgColor,
              color: textColor,
              borderColor: codeError ? errorBorderColor : borderColor
            }}
            placeholder="e.g., TWI_Speaker_001"
            value={code}
            onChangeText={(text) => setCode(text)}
            onBlur={validateInputs}
            autoCapitalize="none"
            autoCorrect={false}
            placeholderTextColor={placeholderColor}
            returnKeyType="next"
            editable={isNewParticipant} // Only editable when creating a new participant
          />
          <View className="h-5 mt-0.5 mb-2.5">
            {codeError ? <Text className="text-danger text-xs dark:text-danger-light">{codeError}</Text> : null}
            {isLoadingCodes && <Text className="text-xs italic" style={{ color: placeholderColor }}>Checking existing codes...</Text>}
          </View>

          {/* Dialect */}
          <Text className="text-sm font-semibold mb-1 text-neutral-700 uppercase tracking-wider dark:text-neutral-400">
            Dialect (e.g., Asante, Fante)
          </Text>
          <TextInput
            className={getInputStyle(!!dialectError)}
            style={{ backgroundColor: inputBgColor, color: textColor, borderColor: dialectError ? errorBorderColor : borderColor }}
            placeholder="Optional"
            value={dialect}
            onChangeText={(text) => setDialect(text)}
            onBlur={validateInputs}
            autoCapitalize="words"
            placeholderTextColor={placeholderColor}
            returnKeyType="next"
          />
          <View className="h-5 mt-0.5 mb-2.5">
            {dialectError ? <Text className="text-danger text-xs dark:text-danger-light">{dialectError}</Text> : null}
          </View>

          {/* Age Range */}
          <Text className="text-sm font-semibold mb-1 text-neutral-700 uppercase tracking-wider dark:text-neutral-400">
            Age Range (e.g., 18-25, 26-35)
          </Text>
          <TextInput
            className={getInputStyle(!!ageRangeError)}
            style={{ backgroundColor: inputBgColor, color: textColor, borderColor: ageRangeError ? errorBorderColor : borderColor }}
            placeholder="Optional"
            value={ageRange}
            onChangeText={(text) => setAgeRange(text)}
            onBlur={validateInputs}
            placeholderTextColor={placeholderColor}
            returnKeyType="next"
          />
          <View className="h-5 mt-0.5 mb-2.5">
            {ageRangeError ? <Text className="text-danger text-xs dark:text-danger-light">{ageRangeError}</Text> : null}
          </View>

          {/* Gender */}
          <Text className="text-sm font-semibold mb-1 text-neutral-700 uppercase tracking-wider dark:text-neutral-400">
            Gender (e.g., Male, Female, Other)
          </Text>
          <TextInput
            className={getInputStyle(!!genderError)}
            style={{ backgroundColor: inputBgColor, color: textColor, borderColor: genderError ? errorBorderColor : borderColor }}
            placeholder="Optional"
            value={gender}
            onChangeText={(text) => setGender(text)}
            onBlur={validateInputs}
            autoCapitalize="words"
            placeholderTextColor={placeholderColor}
            returnKeyType="done"
            onSubmitEditing={handleSave}
          />
          <View className="h-5 mt-0.5 mb-4">
            {genderError ? <Text className="text-danger text-xs dark:text-danger-light">{genderError}</Text> : null}
          </View>

          {/* Buttons */}
          <View className="flex-row justify-between mt-2">
            {/* {onCancel && (
              <Button
                title="Cancel"
                onPress={onCancel}
                className="flex-1 mr-2 py-3.5 px-6 rounded-lg bg-neutral-200 dark:bg-neutral-700 items-center"
                textClassName="text-neutral-800 dark:text-neutral-100 text-lg font-medium"
              />
            )} */}

            <Button
              title={isNewParticipant ? "Create Participant" : "Update Details"}
              onPress={handleSave}
              disabled={isLoading || isLoadingCodes}
              isLoading={isLoading}
              className={`flex-1 ${onCancel ? 'ml-2' : ''} py-3.5 px-6 rounded-lg bg-primary dark:bg-primary-dark items-center shadow-md`}
              disabledClassName="bg-primary-light opacity-70 dark:bg-primary/50"
              textClassName="text-white text-lg font-bold dark:text-neutral-100"
              icon={isNewParticipant ? "account-plus" : "check-circle-outline"}
              iconColor="white"
            />
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </ThemedView>
  );
}
