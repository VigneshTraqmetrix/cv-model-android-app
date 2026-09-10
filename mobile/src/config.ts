/**
 * Backend base URL.
 *
 * During development with Expo Go on a physical Android device, "localhost"
 * refers to the phone itself, not your dev machine -- point this at your
 * machine's LAN IP (e.g. http://192.168.1.20:8000) or an ngrok/tunnel URL.
 * The Android emulator's special alias for the host machine is
 * http://10.0.2.2:8000.
 *
 * For a production build, replace this with your deployed API's HTTPS URL.
 */
export const API_BASE_URL = "http://10.0.2.2:8000";

export const REQUEST_TIMEOUT_MS = 30000;
