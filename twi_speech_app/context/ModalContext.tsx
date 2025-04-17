import React, { createContext, useContext, useState, useCallback } from 'react';
import { Modal, View, StyleSheet, Dimensions, Platform, Animated } from 'react-native';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

interface ModalContextType {
  showModal: (content: React.ReactNode) => void;
  hideModal: () => void;
}

const ModalContext = createContext<ModalContextType>({
  showModal: () => {},
  hideModal: () => {},
});

export const ModalProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [modalContent, setModalContent] = useState<React.ReactNode | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const fadeAnim = React.useRef(new Animated.Value(0)).current;

  const showModal = useCallback((content: React.ReactNode) => {
    setModalContent(content);
    setIsVisible(true);
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 200,
      useNativeDriver: true,
    }).start();
  }, [fadeAnim]);

  const hideModal = useCallback(() => {
    Animated.timing(fadeAnim, {
      toValue: 0,
      duration: 200,
      useNativeDriver: true,
    }).start(() => {
      setIsVisible(false);
      setModalContent(null);
    });
  }, [fadeAnim]);

  return (
    <ModalContext.Provider value={{ showModal, hideModal }}>
      {children}
      <Modal
        transparent
        visible={isVisible}
        animationType="none"
        onRequestClose={hideModal}
        statusBarTranslucent={Platform.OS === 'android'}
      >
        <Animated.View
          style={[
            styles.backdrop,
            {
              opacity: fadeAnim
            }
          ]}
        >
          <View style={styles.modalWrapper}>
            {modalContent}
          </View>
        </Animated.View>
      </Modal>
    </ModalContext.Provider>
  );
};

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalWrapper: {
    width: Math.min(SCREEN_WIDTH - 32, 400),
    maxHeight: SCREEN_HEIGHT - 80,
    margin: 16,
    borderRadius: 16,
    overflow: 'hidden',
  }
});

export const useModal = () => useContext(ModalContext);
