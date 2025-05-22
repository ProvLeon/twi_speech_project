import React, { useEffect, useRef } from 'react';
import { View, StyleSheet } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withRepeat,
  withSequence,
  Easing,
  cancelAnimation,
  interpolate,
  Extrapolation,
  withDelay,
} from 'react-native-reanimated';

interface SoundWaveAnimationProps {
  isAnimating: boolean; // Tied to recording status
  barColor?: string;
  barWidth?: number;
  barHeightMin?: number;
  barHeightMax?: number;
  gap?: number;
  numberOfBars?: number;
  animationDurationBase?: number; // Base duration for one cycle (up/down)
}

const DEFAULT_BAR_COLOR = '#4F46E5'; // Primary color from your config

export const SoundWaveAnimation: React.FC<SoundWaveAnimationProps> = ({
  isAnimating,
  barColor = DEFAULT_BAR_COLOR,
  barWidth = 5, // Slightly wider bars
  barHeightMin = 5, // Minimum height when idle
  barHeightMax = 25, // Maximum height when animating
  gap = 4, // Slightly larger gap
  numberOfBars = 7, // More bars for a smoother look
  animationDurationBase = 500, // Base duration for a full up/down cycle
}) => {
  // Create shared values for each bar's height factor (0 to 1)
  const animatedValues = useRef<Animated.SharedValue<number>[]>(
    Array.from({ length: numberOfBars }, () => useSharedValue(0))
  ).current;

  useEffect(() => {
    if (isAnimating) {
      animatedValues.forEach((val, index) => {
        // Introduce randomness and stagger for a more organic feel
        const randomFactor = Math.random() * 0.4 + 0.8; // 0.8 to 1.2
        const duration = (animationDurationBase * randomFactor) / 2; // Duration for up or down phase
        const delay = (index * animationDurationBase) / numberOfBars / 3; // Stagger start times

        // Animate up and down repeatedly with slight variations
        val.value = withDelay(
          delay,
          withRepeat(
            withSequence(
              // Animate up
              withTiming(1, {
                duration: duration,
                easing: Easing.bezier(0.25, 0.1, 0.25, 1), // Smoother easing
              }),
              // Animate down
              withTiming(0, {
                duration: duration * (Math.random() * 0.3 + 0.85), // Slightly vary down duration
                easing: Easing.bezier(0.42, 0, 0.58, 1), // Smoother easing
              })
            ),
            -1, // Repeat indefinitely
            true // Reverse direction on repeat
          )
        );
      });
    } else {
      // Smoothly animate all bars down to the minimum height when not animating
      animatedValues.forEach((val) => {
        cancelAnimation(val);
        val.value = withTiming(0, {
          duration: animationDurationBase / 3, // Faster transition to idle
          easing: Easing.out(Easing.ease),
        });
      });
    }

    // Cleanup function to cancel animations on unmount
    return () => {
      animatedValues.forEach((val) => cancelAnimation(val));
    };
    // Depend on isAnimating to restart/stop animation
  }, [isAnimating, animatedValues, animationDurationBase, numberOfBars]);

  return (
    <View style={styles.container} accessibilityRole="progressbar" accessibilityLabel="Sound wave animation">
      {animatedValues.map((animatedValue, index) => {
        // Create the animated style for each bar - this avoids reading .value during render
        const animatedStyle = useAnimatedStyle(() => {
          // Interpolate the height based on the animated value (0 to 1)
          const height = interpolate(
            animatedValue.value,
            [0, 1],
            [barHeightMin, barHeightMax],
            Extrapolation.CLAMP // Prevent height going outside min/max bounds
          );

          // Optionally, interpolate opacity for a subtle fade effect
          const opacity = interpolate(
            animatedValue.value,
            [0, 0.2, 1], // Start fading in slightly above min height
            [0.6, 0.8, 1], // Opacity from 0.6 to 1
            Extrapolation.CLAMP
          );

          return {
            height,
            opacity: isAnimating ? opacity : 0.6, // Keep minimum opacity when idle
          };
        });

        return (
          <Animated.View
            key={index}
            style={[
              styles.bar,
              {
                width: barWidth,
                backgroundColor: barColor,
                marginLeft: index > 0 ? gap : 0,
              },
              animatedStyle, // Apply the calculated animated styles
            ]}
          />
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'flex-end', // Align bars to the bottom
    justifyContent: 'center',
    height: 35, // Increased height slightly to accommodate taller bars + breathing room
    minWidth: 80, // Give it some minimum width
  },
  bar: {
    borderRadius: 3, // Keep rounded corners
  },
});
