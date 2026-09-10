import { StatusBar } from "expo-status-bar";
import RootNavigator from "@/navigation";

export default function App() {
  return (
    <>
      <StatusBar style="light" />
      <RootNavigator />
    </>
  );
}
