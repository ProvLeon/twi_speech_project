// constants/api.ts

// --- FIND YOUR LOCAL IP ADDRESS ---
// macOS: System Settings > Network > Wi-Fi/Ethernet > Details... > TCP/IP > IP Address (e.g., 192.168.1.100)
// Windows: Open Command Prompt, type `ipconfig`, look for IPv4 Address under your active network adapter.
// Linux: `ip addr show`

// const YOUR_COMPUTER_IP = '192.168.83.117'; // <-- !!! REPLACE THIS !!!

// Use localhost for web/simulator if possible, IP for physical devices/some emulators
// export const API_BASE_URL = Platform.OS === 'web' || Platform.OS === 'ios' // Simulator often maps localhost
//   ? 'http://127.0.0.1:8000'
//   : `http://${YOUR_COMPUTER_IP}:8000`;

// For deployed backend:
export const API_BASE_URL = 'https://twi-speech-backend-data.onrender.com';

export const UPLOAD_AUDIO_ENDPOINT = `${API_BASE_URL}/upload/audio`;
