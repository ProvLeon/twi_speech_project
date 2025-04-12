import React, { useId, useState } from 'react';
import { View, Text, StyleSheet, Image, StatusBar, useColorScheme } from 'react-native';
import { useRouter } from 'expo-router';
import { ThemedSafeAreaView } from '@/components/ThemedSafeAreaView';
import { useThemeColor } from '@/hooks/useThemeColor';
import { Button } from '@/components/Button';
import AsyncStorage from '@react-native-async-storage/async-storage';

const WELCOME_SEEN_KEY = 'welcomeScreenSeen_v1';

export default function WelcomeScreen() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const primaryColor = useThemeColor({}, 'tint');
  const textColor = useThemeColor({}, 'text');
  const secondaryTextColor = useThemeColor({ light: '#6B7280', dark: '#9CA3AF' }, 'text');
  const backgroundColor = useThemeColor({}, 'background');

  const darkImage = require('@/assets/images/welcome-illustration_dark.png');
  const lightImage = require('@/assets/images/welcome-illustration_light.png');
  const theme = useColorScheme();
  const welcomeImage = theme === 'dark' ? darkImage : lightImage;

  const handleContinue = async () => {
    setIsLoading(true);
    try {
      // Mark welcome as seen and redirect to walkthrough
      await AsyncStorage.setItem(WELCOME_SEEN_KEY, 'true');
      router.replace('/walkthrough');
    } catch (error) {
      console.error('Error saving welcome screen state:', error);
      setIsLoading(false);
    }
  };

  return (
    <>
      <StatusBar barStyle="dark-content" backgroundColor={backgroundColor} />
      <ThemedSafeAreaView className="flex-1">
        <View className="flex-1 justify-between py-10 px-6">
          {/* Top section with logo and welcome text */}
          <View className="items-center pt-10">
            <Image
              source={require('@/assets/images/icon.png')}
              style={styles.logo}
              resizeMode="contain"
            />
            <Text className="text-3xl font-bold text-center mt-6" style={{ color: textColor }}>
              Welcome to Twi Speech
            </Text>
            <Text className="text-base text-center mt-3 max-w-xs mx-auto" style={{ color: secondaryTextColor }}>
              Help build speech recognition technology for the Twi language
            </Text>
          </View>

          {/* Middle illustration */}
          <View className="items-center justify-center flex-1 rounded-full">
            <Image
              source={welcomeImage}
              style={styles.illustration}
              resizeMode="contain"
              className='opacity-70 dark:opacity-60'
            />
          </View>

          {/* Bottom section with button */}
          <View className="w-full">
            <Button
              title="Continue"
              onPress={handleContinue}
              isLoading={isLoading}
              className="bg-primary dark:bg-primary-dark py-4 rounded-full w-full"
              textClassName="text-white text-center text-lg font-semibold"
              disabledClassName="opacity-70"
            />

            <Text className="text-xs text-center mt-6" style={{ color: secondaryTextColor }}>
              By continuing, you agree to our Terms of Service and Privacy Policy.
            </Text>
          </View>
        </View>
      </ThemedSafeAreaView>
    </>
  );
}

const styles = StyleSheet.create({
  logo: {
    width: 100,
    height: 100,
  },
  illustration: {
    width: '100%',
    height: "100%",
  }
});
