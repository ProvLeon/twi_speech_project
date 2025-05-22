import { Audio } from 'expo-av';
import { Alert, Linking, Platform } from 'react-native';

export async function requestMicrophonePermissions(): Promise<boolean> {
  console.log('Requesting microphone permissions...');
  const { status } = await Audio.requestPermissionsAsync();

  if (status === 'granted') {
    console.log('Microphone permissions granted.');
    return true;
  } else {
    console.log('Microphone permissions denied.');
    Alert.alert(
      'Permissions Required',
      'This app needs microphone access to record audio. Please grant permissions in your settings.',
      [
        { text: 'Cancel', style: 'cancel' },
        // Link to app settings if possible (might not work reliably across all platforms/versions)
        { text: 'Open Settings', onPress: () => Linking.openSettings() },
      ]
    );
    return false;
  }
}
