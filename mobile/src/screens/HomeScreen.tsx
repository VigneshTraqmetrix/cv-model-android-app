import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import type { NativeStackScreenProps } from "@react-navigation/native-stack";

import type { RootStackParamList } from "@/types";

type Props = NativeStackScreenProps<RootStackParamList, "Home">;

/**
 * Rice grains and cones are photographed at very different distances/zoom
 * (macro close-up vs. standing back), so the user picks which pipeline to
 * run before opening the camera rather than us guessing from one shot.
 */
export default function HomeScreen({ navigation }: Props) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>What do you want to count?</Text>
      <Text style={styles.subtitle}>
        Choose a mode, then capture a clear, well-lit photo of the objects laid out
        on a plain, contrasting background.
      </Text>

      <TouchableOpacity
        style={[styles.card, styles.riceCard]}
        onPress={() => navigation.navigate("Camera", { mode: "rice" })}
      >
        <Text style={styles.cardEmoji}>🌾</Text>
        <Text style={styles.cardTitle}>Count Rice Grains</Text>
        <Text style={styles.cardHint}>Close-up shot, grains spread out, minimal overlap</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.card, styles.coneCard]}
        onPress={() => navigation.navigate("Camera", { mode: "cone" })}
      >
        <Text style={styles.cardEmoji}>🔶</Text>
        <Text style={styles.cardTitle}>Count Cones</Text>
        <Text style={styles.cardHint}>Wider shot, cones fully visible and unobstructed</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F7F9FA",
    padding: 24,
    justifyContent: "center",
  },
  title: {
    fontSize: 24,
    fontWeight: "700",
    color: "#1F2933",
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: "#616E7C",
    marginBottom: 32,
    lineHeight: 20,
  },
  card: {
    borderRadius: 16,
    padding: 24,
    marginBottom: 16,
    borderWidth: 1,
  },
  riceCard: {
    backgroundColor: "#EFFAF1",
    borderColor: "#3EBD6D",
  },
  coneCard: {
    backgroundColor: "#FFF3E8",
    borderColor: "#E8760C",
  },
  cardEmoji: {
    fontSize: 32,
    marginBottom: 8,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#1F2933",
    marginBottom: 4,
  },
  cardHint: {
    fontSize: 13,
    color: "#616E7C",
  },
});
