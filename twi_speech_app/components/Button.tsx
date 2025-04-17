import React from 'react';
import { TouchableOpacity, Text, ViewStyle, TextStyle, ActivityIndicator, View } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';

interface ButtonProps {
  title: string;
  onPress: () => void;
  style?: ViewStyle;
  textStyle?: TextStyle;
  disabled?: boolean;
  isLoading?: boolean;
  icon?: keyof typeof MaterialCommunityIcons.glyphMap | null;
  iconSize?: number;
  iconColor?: string;
  iconPosition?: 'left' | 'right';
  className?: string; // This is the style for the TouchableOpacity itself
  disabledClassName?: string;
  textClassName?: string;
  // NEW: Optional className for the inner content wrapper
  contentClassName?: string;
}

export const Button: React.FC<ButtonProps> = ({
  title,
  onPress,
  style,
  textStyle,
  disabled = false,
  isLoading = false,
  icon = null,
  iconSize = 20,
  iconColor = 'white',
  iconPosition = 'left',
  // Default className for the outer TouchableOpacity
  className = 'py-3 px-5 bg-blue-600 rounded-lg flex-row items-center justify-center',
  disabledClassName = 'opacity-60 bg-gray-400',
  textClassName = 'text-white text-center text-base font-medium',
  // Default className for the inner View
  contentClassName = 'flex-row items-center justify-center', // Added justify-center here too
}) => {
  const effectiveClassName = `${className} ${disabled || isLoading ? disabledClassName : ''}`;
  const finalIconColor = iconColor || (textStyle?.color as string) || '#000000';

  const IconComponent = icon ? (
    <MaterialCommunityIcons
      name={icon}
      size={iconSize}
      color={finalIconColor}
      style={{
        marginRight: title && iconPosition === 'left' ? 8 : 0,
        marginLeft: title && iconPosition === 'right' ? 8 : 0,
      }}
    />
  ) : null;

  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={disabled || isLoading}
      className={effectiveClassName} // Outer TouchableOpacity centers the inner View
      style={style}
      activeOpacity={0.7}
    >
      {isLoading ? (
        <View className="absolute inset-0 flex items-center justify-center">
          <ActivityIndicator size="small" color={finalIconColor} />
        </View>
      ) : (
        // Inner View now uses contentClassName, which defaults to including justify-center
        <View className={contentClassName}>
          {iconPosition === 'left' && IconComponent}
          {title ? (
            <Text className={textClassName} style={textStyle}>
              {title}
            </Text>
          ) : null}
          {iconPosition === 'right' && IconComponent}
        </View>
      )}
    </TouchableOpacity>
  );
};
