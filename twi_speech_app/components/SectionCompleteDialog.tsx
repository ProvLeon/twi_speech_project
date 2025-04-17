import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Modal } from './Modal';
import { Button } from './Button';
import { useThemeColor } from '@/hooks/useThemeColor';

interface SectionCompleteDialogProps {
  visible: boolean;
  onClose: () => void;
  sectionNumber: number;
  sectionTitle: string;
  onContinue: () => void;
  onGoBack: () => void;
}

export const SectionCompleteDialog: React.FC<SectionCompleteDialogProps> = ({
  visible,
  onClose,
  sectionNumber,
  sectionTitle,
  onContinue,
  onGoBack,
}) => {
  const primaryColor = useThemeColor({}, 'tint');
  const textColor = useThemeColor({}, 'text');
  const secondaryTextColor = useThemeColor({}, 'secondaryText');

  return (
    <Modal visible={visible} onClose={onClose}>
      <View style={styles.container}>
        {/* <Button
          title=""
          icon="close"
          onPress={onGoBack}
          className="absolute bg-neutral-300 dark:bg-neutral-800 mb-3 p-3 top-0 right-0 rounded-lg items-center justify-center flex"
          textClassName="text-neutral-800 dark:text-neutral-100 font-semibold"
        /> */}
        <MaterialCommunityIcons
          name="check-circle-outline"
          size={56}
          color={primaryColor}
          style={styles.icon}
        />

        <Text style={[styles.title, { color: textColor }]}>
          Section Complete!
        </Text>

        <Text style={[styles.message, { color: secondaryTextColor }]}>
          {`You have completed Section ${sectionNumber}:\n`}<Text className="font-extrabold">{`${sectionTitle}`}</Text> {'\n\nClick next to go to the next section?'}
        </Text>

        <View style={styles.buttonContainer}>
          {/* <Button
            title="Go Back Home"
            icon="arrow-left"
            onPress={onGoBack}
            className="w-full bg-neutral-200 dark:bg-neutral-700 mb-3 py-3"
            textClassName="text-neutral-800 dark:text-neutral-100 font-semibold"
          /> */}

          <Button
            title="Continue"
            icon="arrow-right"
            iconPosition="right"
            onPress={onContinue}
            className="w-full bg-primary dark:bg-primary-dark py-3 rounded-lg"
            textClassName="text-white font-semibold"
          />
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
  },
  icon: {
    marginBottom: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 8,
  },
  message: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 24,
  },
  buttonContainer: {
    width: '100%',
  },
});
