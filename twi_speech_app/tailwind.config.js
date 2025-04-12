/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,jsx,ts,tsx}", "./components/**/*.{js,jsx,ts,tsx}"],
  presets: [require("nativewind/preset")],
  theme: {
    extend: {
      colors: {
        primary: { light: '#E0E7FF', DEFAULT: '#4F46E5', dark: '#3730A3' },
        secondary: { light: '#CCFBF1', DEFAULT: '#14B8A6', dark: '#0F766E' },
        success: { light: '#D1FAE5', DEFAULT: '#10B981', dark: '#047857' },
        danger: { light: '#FEE2E2', DEFAULT: '#EF4444', dark: '#B91C1C' },
        warning: { light: '#FEF3C7', DEFAULT: '#F59E0B', dark: '#B45309' },
        neutral: {
          50: '#F9FAFB', 100: '#F3F4F6', 200: '#E5E7EB', 300: '#D1D5DB',
          400: '#9CA3AF', 500: '#6B7280', 600: '#4B5563', 700: '#374151',
          800: '#1F2937', 900: '#111827',
        },
      },
    },
  },
  plugins: [],
};
