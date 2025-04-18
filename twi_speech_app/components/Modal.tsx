import React from 'react';
import { Modal as RNModal, View, StyleSheet, TouchableOpacity, Dimensions, Platform, Text } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { ThemedView } from './ThemedView';
import { useThemeColor } from '@/hooks/useThemeColor';
import { Button } from './Button'

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface ModalProps {
  visible: boolean;
  onClose: () => void;
  children: React.ReactNode;
  showCloseButton?: boolean;
}

export const Modal: React.FC<ModalProps> = ({
  visible,
  onClose,
  children,
  showCloseButton = true,
}) => {
  const backgroundColor = useThemeColor({}, 'background');
  const iconColor = useThemeColor({}, 'text');

  return (
    <RNModal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onClose}
    // statusBarTranslucent={Platform.OS === 'android'}
    >
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0, 0, 0, 0.5)', padding: 20 }}>
        <ThemedView style={styles.content}>
          {showCloseButton && (
            <Button
              title=""
              icon="close"
              onPress={onClose}
              className="absolute bg-neutral-300 dark:bg-neutral-800 mb-3 p-3 top-0 right-0 rounded-lg items-center justify-center flex m-4"
              textClassName="text-neutral-800 dark:text-neutral-100 font-semibold"
            />
          )}
          {children}
        </ThemedView>
      </View>
    </RNModal>
  );
};

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  content: {
    width: Math.min(SCREEN_WIDTH - 40, 400),
    borderRadius: 16,
    padding: 20,
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.25,
        shadowRadius: 4,
      },
      android: {
        elevation: 5,
      },
    }),
  },
  closeButton: {
    position: 'absolute',
    right: 12,
    top: 12,
    zIndex: 1,
  },
});
