import { useState } from "react";
import { Dimensions, Image, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import type { NativeStackScreenProps } from "@react-navigation/native-stack";

import DetectionOverlay from "@/components/DetectionOverlay";
import SummaryBar from "@/components/SummaryBar";
import type { RootStackParamList } from "@/types";

type Props = NativeStackScreenProps<RootStackParamList, "Results">;

const SCREEN_WIDTH = Dimensions.get("window").width;

export default function ResultsScreen({ route, navigation }: Props) {
  const { mode, photo, result } = route.params;
  const [displayHeight, setDisplayHeight] = useState(0);

  // Fit the photo to the screen width and compute the matching height from
  // its original aspect ratio, so DetectionOverlay can scale boxes exactly
  // to how the image is actually being rendered.
  const displayWidth = SCREEN_WIDTH;
  const aspectRatio = result.image_width / result.image_height;
  const computedHeight = displayWidth / aspectRatio;

  return (
    <SafeAreaView style={styles.container} edges={["bottom"]}>
      <View style={[styles.imageWrapper, { width: displayWidth, height: computedHeight }]}>
        <Image
          source={{ uri: photo.uri }}
          style={{ width: displayWidth, height: computedHeight }}
          resizeMode="contain"
          onLayout={() => setDisplayHeight(computedHeight)}
        />
        <DetectionOverlay
          detections={result.detections}
          imageWidth={result.image_width}
          imageHeight={result.image_height}
          displayWidth={displayWidth}
          displayHeight={displayHeight || computedHeight}
        />
      </View>

      {result.status === "no_detections" && (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateTitle}>No objects detected</Text>
          <Text style={styles.emptyStateBody}>
            Make sure the {mode === "rice" ? "grains" : "cones"} are well-lit, in focus, and
            contrast clearly against the background, then try again.
          </Text>
        </View>
      )}

      <SummaryBar riceCount={result.rice_count} coneCount={result.cone_count} />

      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.scanAgainButton}
          onPress={() => navigation.navigate("Camera", { mode })}
        >
          <Text style={styles.scanAgainText}>Scan Again</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.homeButton}
          onPress={() => navigation.popToTop()}
        >
          <Text style={styles.homeButtonText}>Done</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0B0F14" },
  imageWrapper: { alignSelf: "center", backgroundColor: "#000" },
  emptyState: { padding: 20, backgroundColor: "#2B0F12" },
  emptyStateTitle: { color: "#FFB4B4", fontWeight: "700", fontSize: 15, marginBottom: 4 },
  emptyStateBody: { color: "#E8C7C7", fontSize: 13, lineHeight: 18 },
  actions: { flexDirection: "row", padding: 16, gap: 12 },
  scanAgainButton: {
    flex: 1,
    backgroundColor: "#3E7BFA",
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
  },
  scanAgainText: { color: "#fff", fontWeight: "700", fontSize: 15 },
  homeButton: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#3E4C59",
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
  },
  homeButtonText: { color: "#CBD2D9", fontWeight: "700", fontSize: 15 },
});
