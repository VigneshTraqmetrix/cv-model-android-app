import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";

import HomeScreen from "@/screens/HomeScreen";
import CameraScreen from "@/screens/CameraScreen";
import ProcessingScreen from "@/screens/ProcessingScreen";
import ResultsScreen from "@/screens/ResultsScreen";
import type { RootStackParamList } from "@/types";

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function RootNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Home"
        screenOptions={{
          headerStyle: { backgroundColor: "#1F2933" },
          headerTintColor: "#FFFFFF",
          headerTitleStyle: { fontWeight: "600" },
        }}
      >
        <Stack.Screen name="Home" component={HomeScreen} options={{ title: "Cone & Rice Counter" }} />
        <Stack.Screen name="Camera" component={CameraScreen} options={{ title: "Capture", headerShown: false }} />
        <Stack.Screen
          name="Processing"
          component={ProcessingScreen}
          options={{ title: "Analyzing", headerBackVisible: false, gestureEnabled: false }}
        />
        <Stack.Screen name="Results" component={ResultsScreen} options={{ title: "Results" }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
