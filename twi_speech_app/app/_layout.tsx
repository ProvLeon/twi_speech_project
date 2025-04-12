import React, { useEffect, useState } from 'react';
import { SplashScreen, Stack } from "expo-router";
import "./global.css";
import { useFonts } from "expo-font";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { StatusBar } from 'expo-status-bar';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { View } from 'react-native';

const WELCOME_SEEN_KEY = 'welcomeScreenSeen_v1';

export default function RootLayout() {
  const [fontsLoaded] = useFonts({
    SpaceMono: require('@/assets/fonts/SpaceMono-Regular.ttf'),
  });

  useEffect(() => {
    if (fontsLoaded) {
      SplashScreen.hideAsync();
    }
  }, [fontsLoaded]);

  if (!fontsLoaded) return null;

  // Return the Stack immediately without any conditional redirects
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <Stack screenOptions={{
        headerShown: false,
      }}>
        <Stack.Screen
          name="welcome"
          options={{
            title: "Welcome",
            headerShown: false,
            gestureEnabled: false
          }}
        />
        <Stack.Screen
          name="index"
          options={{
            title: "Twi Recorder Home"
          }}
        />
        <Stack.Screen
          name="record"
          options={{
            title: "Record Prompt",
            headerBackTitle: "Home"
          }}
        />

      </Stack>
      <StatusBar style="auto" />
    </GestureHandlerRootView>
  );
}
