import React, { useState, useRef } from 'react';
import { View, Text, StyleSheet, FlatList, Image, Dimensions, Animated, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import { ThemedSafeAreaView } from '@/components/ThemedSafeAreaView';
import { useThemeColor } from '@/hooks/useThemeColor';
import { Button } from '@/components/Button';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const { width } = Dimensions.get('window');

const slides = [
  {
    id: '1',
    title: 'Record Twi Speech',
    description: 'Make a valuable contribution to language technology by recording Twi phrases and sentences.',
    image: require('@/assets/images/walkthrough-1.png'),
    icon: 'microphone'
  },
  {
    id: '2',
    title: 'Privacy First',
    description: 'Your data is handled securely. Participate anonymously with only the information you choose to share.',
    image: require('@/assets/images/walkthrough-2.png'),
    icon: 'shield-check'
  },
  {
    id: '3',
    title: 'Easy to Use',
    description: 'Simple interface designed for efficient recording. See your history and upload progress.',
    image: require('@/assets/images/walkthrough-3.png'),
    icon: 'gesture-tap'
  },
  {
    id: '4',
    title: 'Ready to Start',
    description: 'Join us in making technology more accessible for Twi speakers around the world.',
    image: require('@/assets/images/walkthrough-4.png'),
    icon: 'rocket-launch'
  }
];

export default function WalkthroughScreen() {
  const router = useRouter();
  const [currentIndex, setCurrentIndex] = useState(0);
  const scrollX = useRef(new Animated.Value(0)).current;
  const flatListRef = useRef<FlatList>(null);

  const primaryColor = useThemeColor({}, 'tint');
  const textColor = useThemeColor({}, 'text');
  const secondaryTextColor = useThemeColor({ light: '#6B7280', dark: '#9CA3AF' }, 'text');
  const backgroundColor = useThemeColor({}, 'background');

  const viewableItemsChanged = useRef(({ viewableItems }) => {
    if (viewableItems[0]) {
      setCurrentIndex(viewableItems[0].index);
    }
  }).current;

  const viewConfig = useRef({ viewAreaCoveragePercentThreshold: 50 }).current;

  const handleNext = () => {
    if (currentIndex < slides.length - 1) {
      flatListRef.current?.scrollToIndex({ index: currentIndex + 1 });
    } else {
      handleGetStarted();
    }
  };

  const handleSkip = () => {
    flatListRef.current?.scrollToIndex({ index: slides.length - 1 });
  };

  const handleGetStarted = () => {
    router.replace('/');
  };

  return (
    <ThemedSafeAreaView className="flex-1">
      <View className="flex-1">
        {/* Skip button */}
        {currentIndex < slides.length - 1 && (
          <TouchableOpacity
            onPress={handleSkip}
            className="absolute top-4 right-4 z-10 py-2 px-4"
            style={{ backgroundColor: `${primaryColor}20` }}
            activeOpacity={0.7}
          >
            <Text style={{ color: primaryColor }} className="font-medium">Skip</Text>
          </TouchableOpacity>
        )}

        {/* Main content */}
        <FlatList
          data={slides}
          renderItem={({ item }) => (
            <View style={{ width }} className="flex-1 items-center px-5 pt-10 pb-8 bg-gradient-to-b from-primary/5 to-transparent">
              {/* Icon with gradient background */}
              <View className="items-center justify-center mb-8 mt-8">
                <View
                  className="w-20 h-20 rounded-full items-center justify-center mb-4 bg-gradient-to-br from-primary/30 to-primary/10"
                >
                  <MaterialCommunityIcons name={item.icon as any} size={48} color={primaryColor} />
                </View>
              </View>

              {/* Illustration with subtle gradient overlay */}
              <View className="items-center justify-center flex-1 w-full relative">
                <View className="absolute inset-0 bg-gradient-to-r from-transparent via-primary/5 to-transparent rounded-3xl" />
                <Image
                  source={item.image}
                  style={item.id === '1' ? styles.image : {}}
                  resizeMode="contain"
                  className={` ${item.id === '1' ? '' : 'h-[150px]'} rounded-xl`}
                />
              </View>

              {/* Content area */}
              <View className="items-center w-full max-w-sm mt-8 p-4 bg-white/80 dark:bg-gray-800/80 rounded-xl backdrop-blur-sm">
                <Text className="text-2xl font-bold text-center mb-3 text-primary dark:text-primary-light">
                  {item.title}
                </Text>
                <Text className="text-base text-center" style={{ color: secondaryTextColor }}>
                  {item.description}
                </Text>
              </View>
            </View>
          )}
          horizontal
          showsHorizontalScrollIndicator={false}
          pagingEnabled
          bounces={false}
          keyExtractor={(item) => item.id}
          onScroll={Animated.event(
            [{ nativeEvent: { contentOffset: { x: scrollX } } }],
            { useNativeDriver: false }
          )}
          onViewableItemsChanged={viewableItemsChanged}
          viewabilityConfig={viewConfig}
          ref={flatListRef}
        />

        {/* Pagination */}
        <View className="flex-row justify-center mb-4">
          {slides.map((_, i) => {
            const inputRange = [(i - 1) * width, i * width, (i + 1) * width];

            const dotWidth = scrollX.interpolate({
              inputRange,
              outputRange: [10, 24, 10],
              extrapolate: 'clamp',
            });

            const opacity = scrollX.interpolate({
              inputRange,
              outputRange: [0.3, 1, 0.3],
              extrapolate: 'clamp',
            });

            return (
              <Animated.View
                key={`dot-${i}`}
                style={[
                  styles.dot,
                  {
                    width: dotWidth,
                    opacity,
                    backgroundColor: primaryColor,
                  },
                ]}
              />
            );
          })}
        </View>

        {/* Bottom button */}
        <View className="px-5 mb-8">
          <Button
            title={currentIndex === slides.length - 1 ? "Get Started" : "Next"}
            onPress={handleNext}
            icon={currentIndex === slides.length - 1 ? "check" : "arrow-right"}
            iconPosition="right"
            className="flex flex-row bg-primary dark:bg-primary-dark py-4 rounded-full w-full center items-center justify-center"
            textClassName="text-white text-center text-lg font-semibold mr-2"
            iconColor="white"
            iconSize={22}
          />
        </View>
      </View>
    </ThemedSafeAreaView>
  );
}

const styles = StyleSheet.create({
  image: {
    width: width * 0.8,
    height: width * 0.8,
  },
  dot: {
    height: 10,
    borderRadius: 5,
    marginHorizontal: 4,
  }
});
