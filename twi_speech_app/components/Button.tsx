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
  className?: string;
  disabledClassName?: string;
  textClassName?: string;
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
  iconColor = 'white', // Default icon color for text buttons
  iconPosition = 'left',
  className = 'py-3 px-5 bg-blue-600 rounded-lg flex-row items-center justify-center',
  disabledClassName = 'opacity-60 bg-gray-400',
  textClassName = 'text-white text-center text-base font-medium',
}) => {
  const effectiveClassName = `${className} ${disabled || isLoading ? disabledClassName : ''}`;

  // Use a default color if none is provided specifically for the icon
  const finalIconColor = iconColor || (textStyle?.color as string) || '#000000'; // Default to black if no color context

  const IconComponent = icon ? (
    <MaterialCommunityIcons
      name={icon}
      size={iconSize}
      color={finalIconColor}
      className={finalIconColor}
      style={{
        // Add margin only if there is a title next to it
        marginRight: title && iconPosition === 'left' ? 8 : 0,
        marginLeft: title && iconPosition === 'right' ? 8 : 0,
      }}
    />
  ) : null;

  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={disabled || isLoading}
      className={effectiveClassName}
      style={style}
      activeOpacity={0.7}
    >
      {isLoading ? (
        // Center the activity indicator
        <View className="absolute inset-0 flex items-center justify-center">
          <ActivityIndicator size="small" color={finalIconColor} />
        </View>
      ) : (
        <>
          {iconPosition === 'left' && IconComponent}
          {/* Render Text only if title is not empty */}
          {title ? (
            <Text className={textClassName} style={textStyle}>
              {title}
            </Text>
          ) : null}
          {iconPosition === 'right' && IconComponent}
        </>
      )}
    </TouchableOpacity>
  );
};
