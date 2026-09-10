import { StyleSheet, Text, View } from "react-native";

interface Props {
  riceCount: number;
  coneCount: number;
}

export default function SummaryBar({ riceCount, coneCount }: Props) {
  return (
    <View style={styles.container}>
      <View style={styles.stat}>
        <Text style={styles.value}>{coneCount}</Text>
        <Text style={styles.label}>Total Cones</Text>
      </View>
      <View style={styles.divider} />
      <View style={styles.stat}>
        <Text style={styles.value}>{riceCount}</Text>
        <Text style={styles.label}>Total Rice Grains</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    backgroundColor: "#1F2933",
    paddingVertical: 16,
    paddingHorizontal: 12,
    alignItems: "center",
  },
  stat: { flex: 1, alignItems: "center" },
  value: { color: "#fff", fontSize: 26, fontWeight: "800" },
  label: { color: "#9AA5B1", fontSize: 12, marginTop: 2, textTransform: "uppercase", letterSpacing: 0.5 },
  divider: { width: 1, height: 36, backgroundColor: "#3E4C59" },
});
