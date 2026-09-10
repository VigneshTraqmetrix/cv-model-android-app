import { useRef, useState } from "react";
import { Image, Linking, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as ImageManipulator from "expo-image-manipulator";
import type { NativeStackScreenProps } from "@react-navigation/native-stack";

import type { RootStackParamList } from "@/types";

type Props = NativeStackScreenProps<RootStackParamList, "Camera">;

// Cap the longest edge before upload -- phone cameras produce 12MP+ photos
// that are unnecessarily slow to upload and don't help detection accuracy
// past a point. 1600px keeps individual rice grains comfortably resolvable.
const MAX_UPLOAD_DIMENSION = 1600;

export default function CameraScreen({ route, navigation }: Props) {
  const { mode } = route.params;
  const [permission, requestPermission] = useCameraPermissions();
  const [previewUri, setPreviewUri] = useState<string | null>(null);
  const [capturing, setCapturing] = useState(false);
  const cameraRef = useRef<CameraView>(null);

  // --- Permission states -----------------------------------------------------
  if (!permission) {
    return <View style={styles.center} />; // permission status still loading
  }

  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.permissionTitle}>Camera access needed</Text>
        <Text style={styles.permissionBody}>
          {permission.canAskAgain
            ? "We need camera access to capture a photo of the cones or rice grains."
            : "Camera access was denied. Enable it in Settings to use this feature."}
        </Text>
        <TouchableOpacity
          style={styles.primaryButton}
          onPress={() => (permission.canAskAgain ? requestPermission() : Linking.openSettings())}
        >
          <Text style={styles.primaryButtonText}>
            {permission.canAskAgain ? "Grant Permission" : "Open Settings"}
          </Text>
        </TouchableOpacity>
      </View>
    );
  }

  // --- Capture -----------------------------------------------------------
  const handleCapture = async () => {
    if (!cameraRef.current || capturing) return;
    setCapturing(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.9 });
      if (photo) setPreviewUri(photo.uri);
    } catch (err) {
      console.warn("Failed to capture photo", err);
    } finally {
      setCapturing(false);
    }
  };

  const handleConfirm = async () => {
    if (!previewUri) return;
    // Downscale + re-encode before upload; also normalizes orientation.
    const manipulated = await ImageManipulator.manipulateAsync(
      previewUri,
      [{ resize: { width: MAX_UPLOAD_DIMENSION } }],
      { compress: 0.85, format: ImageManipulator.SaveFormat.JPEG }
    );
    navigation.replace("Processing", {
      mode,
      photo: { uri: manipulated.uri, width: manipulated.width, height: manipulated.height },
    });
  };

  // --- Preview (confirm / retake) -----------------------------------------
  if (previewUri) {
    return (
      <View style={styles.container}>
        <Image source={{ uri: previewUri }} style={styles.previewImage} resizeMode="contain" />
        <View style={styles.previewActions}>
          <TouchableOpacity style={styles.secondaryButton} onPress={() => setPreviewUri(null)}>
            <Text style={styles.secondaryButtonText}>Retake</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.primaryButton} onPress={handleConfirm}>
            <Text style={styles.primaryButtonText}>Use Photo</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // --- Live camera ---------------------------------------------------------
  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing="back" />
      <View style={styles.modeBanner}>
        <Text style={styles.modeBannerText}>
          {mode === "rice" ? "🌾 Rice mode: get close, spread grains out" : "🔶 Cone mode: keep all cones in frame"}
        </Text>
      </View>
      <View style={styles.captureBar}>
        <TouchableOpacity
          style={[styles.shutterButton, capturing && styles.shutterButtonDisabled]}
          onPress={handleCapture}
          disabled={capturing}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 32,
    backgroundColor: "#1F2933",
  },
  camera: { flex: 1 },
  modeBanner: {
    position: "absolute",
    top: 48,
    alignSelf: "center",
    backgroundColor: "rgba(0,0,0,0.55)",
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  modeBannerText: { color: "#fff", fontWeight: "600" },
  captureBar: {
    position: "absolute",
    bottom: 40,
    alignSelf: "center",
  },
  shutterButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: "#fff",
    borderWidth: 4,
    borderColor: "rgba(255,255,255,0.4)",
  },
  shutterButtonDisabled: { opacity: 0.5 },
  previewImage: { flex: 1, backgroundColor: "#000" },
  previewActions: {
    flexDirection: "row",
    justifyContent: "space-between",
    padding: 20,
    backgroundColor: "#111",
  },
  primaryButton: {
    backgroundColor: "#3E7BFA",
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 10,
    marginTop: 20,
  },
  primaryButtonText: { color: "#fff", fontWeight: "700", fontSize: 16 },
  secondaryButton: {
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#fff",
  },
  secondaryButtonText: { color: "#fff", fontWeight: "700", fontSize: 16 },
  permissionTitle: { color: "#fff", fontSize: 20, fontWeight: "700", marginBottom: 12 },
  permissionBody: { color: "#CBD2D9", fontSize: 14, textAlign: "center", lineHeight: 20 },
});
