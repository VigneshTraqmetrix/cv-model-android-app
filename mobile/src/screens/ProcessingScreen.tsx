import { useEffect, useState } from "react";
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import type { NativeStackScreenProps } from "@react-navigation/native-stack";

import { analyzeImage, ApiError } from "@/services/api";
import type { RootStackParamList } from "@/types";

type Props = NativeStackScreenProps<RootStackParamList, "Processing">;

export default function ProcessingScreen({ route, navigation }: Props) {
  const { mode, photo } = route.params;
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      setErrorMessage(null);
      try {
        const result = await analyzeImage(photo, mode);
        if (cancelled) return;

        if (result.status === "blurry" || result.status === "error") {
          setErrorMessage(result.message);
          return;
        }
        // "ok" and "no_detections" both render on the Results screen -- the
        // latter just shows a friendly empty state instead of a crash.
        navigation.replace("Results", { mode, photo, result });
      } catch (err) {
        if (cancelled) return;
        setErrorMessage(err instanceof ApiError ? err.message : "Something went wrong. Please try again.");
      }
    };

    run();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (errorMessage) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorTitle}>Couldn't analyze photo</Text>
        <Text style={styles.errorBody}>{errorMessage}</Text>
        <TouchableOpacity
          style={styles.retryButton}
          onPress={() => navigation.replace("Camera", { mode })}
        >
          <Text style={styles.retryButtonText}>Retake Photo</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color="#3E7BFA" />
      <Text style={styles.loadingText}>
        {mode === "rice" ? "Counting rice grains..." : "Detecting cones..."}
      </Text>
      <Text style={styles.loadingHint}>This usually takes a few seconds</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
    backgroundColor: "#F7F9FA",
  },
  loadingText: { marginTop: 20, fontSize: 16, fontWeight: "600", color: "#1F2933" },
  loadingHint: { marginTop: 6, fontSize: 13, color: "#9AA5B1" },
  errorTitle: { fontSize: 18, fontWeight: "700", color: "#CF1124", marginBottom: 8 },
  errorBody: { fontSize: 14, color: "#616E7C", textAlign: "center", lineHeight: 20, marginBottom: 24 },
  retryButton: {
    backgroundColor: "#3E7BFA",
    paddingVertical: 14,
    paddingHorizontal: 32,
    borderRadius: 10,
  },
  retryButtonText: { color: "#fff", fontWeight: "700", fontSize: 16 },
});
